"""Pluggable sliding-window rate limiting helpers."""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Protocol

from fastapi import Request

from app.config import get_settings


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int


class RateLimiterBackend(Protocol):
    """Backend contract for the per-scope sliding-window limiter.

    Implementations must be safe to share across concurrent
    requests. ``check`` records a hit when it allows the
    request and returns the decision for the current window.
    ``clear`` drops all recorded state; backends whose entries
    expire on their own may implement it as a no-op.
    """

    def check(
        self,
        *,
        scope: str,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitDecision: ...

    def clear(self) -> None: ...


class InMemoryRateLimiter:
    """Thread-safe per-scope/per-key sliding-window limiter."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.RLock()
        self._hits: dict[tuple[str, str], Deque[float]] = defaultdict(deque)

    def check(self, *, scope: str, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        if limit <= 0 or window_seconds <= 0:
            return RateLimitDecision(allowed=True, retry_after_seconds=0)

        now = self._clock()
        cutoff = now - float(window_seconds)
        bucket_key = (scope, key)

        with self._lock:
            bucket = self._hits[bucket_key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= int(limit):
                retry = max(1, int(math.ceil(float(window_seconds) - (now - bucket[0]))))
                return RateLimitDecision(allowed=False, retry_after_seconds=retry)

            bucket.append(now)
            return RateLimitDecision(allowed=True, retry_after_seconds=0)

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()


class RedisBackend:
    """Sliding-window limiter backed by Redis sorted sets.

    Each ``(scope, key)`` pair owns one sorted set: members are
    unique request markers and scores are request timestamps.
    The window is enforced by trimming scores at or below the
    cutoff (``ZREMRANGEBYSCORE``) before counting (``ZCARD``),
    so the state is shared across replicas and survives
    restarts until the key's TTL expires.
    """

    _KEY_PREFIX = "ratelimit"

    def __init__(
        self,
        redis_url: str = "",
        *,
        client: Any | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self._clock = clock
        if client is not None:
            self._client = client
            return
        url = (redis_url or os.getenv("REDIS_URL", "")).strip()
        if not url:
            raise RuntimeError(
                "REDIS_URL must be set when STUDY_RATE_LIMIT_BACKEND=redis"
            )
        try:
            import redis as redis_module
        except ImportError as exc:
            raise RuntimeError(
                "STUDY_RATE_LIMIT_BACKEND=redis requires the 'redis' "
                "package. Install it with: uv add redis"
            ) from exc
        self._client = redis_module.Redis.from_url(url, decode_responses=True)

    def ping(self) -> None:
        """Validate connectivity; raises when Redis is unreachable."""
        self._client.ping()

    def _bucket(self, scope: str, key: str) -> str:
        return f"{self._KEY_PREFIX}:{scope}:{key}"

    def check(self, *, scope: str, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        if limit <= 0 or window_seconds <= 0:
            return RateLimitDecision(allowed=True, retry_after_seconds=0)

        now = self._clock()
        cutoff = now - float(window_seconds)
        bucket = self._bucket(scope, key)

        self._client.zremrangebyscore(bucket, "-inf", cutoff)
        if self._client.zcard(bucket) >= int(limit):
            oldest = self._client.zrange(bucket, 0, 0, withscores=True)
            oldest_score = float(oldest[0][1]) if oldest else now
            retry = max(1, int(math.ceil(float(window_seconds) - (now - oldest_score))))
            return RateLimitDecision(allowed=False, retry_after_seconds=retry)

        self._client.zadd(bucket, {f"{now:.6f}:{uuid.uuid4().hex}": now})
        self._client.expire(bucket, int(window_seconds))
        return RateLimitDecision(allowed=True, retry_after_seconds=0)

    def clear(self) -> None:
        """No-op: every bucket carries a TTL and expires on its own."""
        return None


def create_rate_limiter(settings: Any | None = None) -> RateLimiterBackend:
    """Build the configured rate-limiter backend.

    ``STUDY_RATE_LIMIT_BACKEND`` selects the implementation:
    ``memory`` (default) keeps per-replica state, while
    ``redis`` shares sliding-window state across replicas via
    ``REDIS_URL``.
    """
    settings = settings or get_settings()
    backend_name = str(
        getattr(settings, "rate_limit_backend", "memory") or "memory"
    ).strip().lower()
    if backend_name in {"", "memory", "inmemory", "in-memory"}:
        return InMemoryRateLimiter()
    if backend_name == "redis":
        return RedisBackend(redis_url=str(getattr(settings, "redis_url", "") or ""))
    raise ValueError(f"Unknown STUDY_RATE_LIMIT_BACKEND: {backend_name!r}")


def _trusts_proxy_headers() -> bool:
    """Only a deployment behind a controlled proxy may set X-Forwarded-For.

    Honouring the header unconditionally lets any caller pick its own rate
    limit bucket by sending an arbitrary header, which makes the limiter
    trivially bypassable.
    """
    return bool(getattr(get_settings(), "trusted_proxy_enabled", False))


def request_ip(request: Request) -> str:
    """Resolve the client IP used as a rate-limit key.

    Proxy headers are only consulted when ``trusted_proxy_enabled`` is set;
    otherwise the transport-level peer address is the only trustworthy
    source.
    """
    if _trusts_proxy_headers():
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            first = forwarded.split(",", 1)[0].strip()
            if first:
                return first
        real_ip = request.headers.get("x-real-ip", "").strip()
        if real_ip:
            return real_ip
    client = request.client.host if request.client else ""
    return client or "unknown"


_TOPICS_PREFIX = "/api/topics/"


def _classify_topic_detail_scope(norm_path: str, norm_method: str) -> str | None:
    """Return the ``llm`` scope for the LLM-backed topic GET/PUT routes.

    ``GET /api/topics/{id}`` and ``GET|PUT /api/topics/{id}/preferences`` both
    call ``resolve_topic_ai_settings`` -> ``TopicLanguageAdvisor.advise_topic``.
    ``GET /api/topics`` (the list route) and ``GET /api/topics/{id}/videos`` are
    deliberately excluded: neither reaches the advisor.
    """
    if not norm_path.startswith(_TOPICS_PREFIX):
        return None
    if norm_method not in {"GET", "PUT"}:
        return None
    segments = [part for part in norm_path[len(_TOPICS_PREFIX) :].split("/") if part]
    if len(segments) == 1:
        return "llm" if norm_method == "GET" else None
    if len(segments) == 2 and segments[1] == "preferences":
        return "llm"
    return None


def classify_rate_limit_scope(path: str, method: str) -> str | None:
    norm_path = (path or "").strip().lower()
    norm_method = (method or "").strip().upper()

    oauth_paths = {
        "/api/auth/github",
        "/api/auth/github/callback",
        "/api/auth/google",
        "/api/auth/google/callback",
        "/api/user-settings/google/connect",
        "/api/user-settings/google/callback",
    }
    if norm_path in oauth_paths and norm_method == "GET":
        return "oauth"

    if norm_path in {"/api/auth/me", "/api/auth/logout"}:
        return "session"

    if norm_path.startswith("/api/llm/"):
        return "llm"
    if norm_path.startswith("/api/questions/") and norm_method == "POST":
        return "llm"
    if norm_path == "/api/chat/follow-up" and norm_method == "POST":
        return "llm"
    if norm_path.startswith("/api/interview-sessions") and norm_method == "POST":
        return "llm"
    if norm_path in {"/api/topics/custom", "/api/topics/custom/stream"} and norm_method == "POST":
        return "llm"
    if norm_path.endswith("/content/generate/stream") and norm_path.startswith("/api/topics/"):
        return "llm"
    # Topic detail and topic preferences both resolve the language profile,
    # which runs TopicLanguageAdvisor.advise_topic — a real LLM call — whenever
    # the cached profile is missing or stale. Both are GET/PUT, so the
    # POST-scoped rules above can never see them.
    topic_detail_scope = _classify_topic_detail_scope(norm_path, norm_method)
    if topic_detail_scope:
        return topic_detail_scope
    # Explicit progress saves are LLM-backed (each one runs a summary call), so
    # they share the llm budget. The autosave beacon is deliberately left
    # unlimited: rate limiting it would reject the request before the handler
    # runs and lose the durable raw write for a beacon the browser never awaits.
    if norm_path.startswith("/api/progress/") and norm_path.endswith("/save"):
        return "llm"
    return None
