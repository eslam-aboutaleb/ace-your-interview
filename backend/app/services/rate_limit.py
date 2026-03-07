"""In-memory sliding-window rate limiting helpers."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

from fastapi import Request


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int


class InMemoryRateLimiter:
    """Thread-safe per-scope/per-key sliding-window limiter."""

    def __init__(self):
        self._lock = threading.RLock()
        self._hits: dict[tuple[str, str], Deque[float]] = defaultdict(deque)

    def check(self, *, scope: str, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        if limit <= 0 or window_seconds <= 0:
            return RateLimitDecision(allowed=True, retry_after_seconds=0)

        now = time.monotonic()
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


def request_ip(request: Request) -> str:
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
    return None
