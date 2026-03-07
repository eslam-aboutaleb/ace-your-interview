"""Topic-section video recommendations with global quota gate."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from app.config import Settings
from app.services.learning_store import LearningStore

logger = logging.getLogger(__name__)

_FEATURE_KEY = "topic_videos"
_ALLOWED_STATUSES = {"enabled", "disabled_quota_exhausted", "probe", "disabled_config"}
_QUOTA_REASONS = {
    "quotaexceeded",
    "dailylimitexceeded",
    "dailylimitexceededunreg",
    "ratelimitexceeded",
    "userratelimitexceeded",
}


class VideoRecommender:
    """Fetch and cache YouTube recommendations for a topic section."""

    def __init__(self, store: LearningStore, settings: Settings):
        self.store = store
        self.settings = settings

    @staticmethod
    def _next_pacific_midnight_utc_iso(now_utc: datetime | None = None) -> str:
        now = now_utc or datetime.now(UTC)
        pacific = ZoneInfo("America/Los_Angeles")
        now_pt = now.astimezone(pacific)
        next_midnight_pt = (now_pt + timedelta(days=1)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        return next_midnight_pt.astimezone(UTC).isoformat()

    @staticmethod
    def _parse_iso(value: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value or ""))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @staticmethod
    def _parse_duration_seconds(value: str) -> int:
        text = str(value or "").strip().upper()
        if not text:
            return 0
        match = re.match(r"^P(?:\d+D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", text)
        if not match:
            return 0
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return (hours * 3600) + (minutes * 60) + seconds

    @staticmethod
    def _error_reasons(payload: dict) -> set[str]:
        reasons: set[str] = set()
        err = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(err, dict):
            return reasons
        errors = err.get("errors")
        if not isinstance(errors, list):
            return reasons
        for item in errors:
            if not isinstance(item, dict):
                continue
            reason = str(item.get("reason", "")).strip().lower()
            if reason:
                reasons.add(reason)
        return reasons

    @staticmethod
    def _is_quota_error(status_code: int, payload: dict) -> bool:
        if int(status_code) == 429:
            return True
        if int(status_code) != 403:
            return False
        reasons = VideoRecommender._error_reasons(payload)
        return any(reason in _QUOTA_REASONS for reason in reasons)

    def _config_enabled(self) -> bool:
        return bool(self.settings.enable_topic_videos and self.settings.youtube_api_key.strip())

    def _status_payload(
        self,
        *,
        status: str,
        disabled_until: str = "",
        reason: str = "",
    ) -> dict:
        current = status if status in _ALLOWED_STATUSES else "disabled_config"
        enabled = current in {"enabled", "probe"}
        return {
            "enabled": enabled,
            "status": current,
            "disabled_until": disabled_until,
            "reason": reason,
        }

    def get_status(self) -> dict:
        if not self.settings.enable_topic_videos:
            return self._status_payload(status="disabled_config", reason="feature_flag_off")
        if not self.settings.youtube_api_key.strip():
            return self._status_payload(status="disabled_config", reason="youtube_api_key_missing")

        state = self.store.get_feature_state(feature_key=_FEATURE_KEY)
        if state is None:
            state = self.store.set_feature_state(feature_key=_FEATURE_KEY, status="enabled")

        status = str(state.get("status", "enabled")).strip().lower()
        disabled_until = str(state.get("disabled_until", ""))
        reason = str(state.get("reason", ""))

        if status == "disabled_quota_exhausted":
            disabled_until_dt = self._parse_iso(disabled_until)
            if disabled_until_dt is not None and datetime.now(UTC) >= disabled_until_dt:
                state = self.store.set_feature_state(
                    feature_key=_FEATURE_KEY,
                    status="probe",
                    reason="quota_reset_probe",
                )
                status = str(state.get("status", "probe"))
                disabled_until = str(state.get("disabled_until", ""))
                reason = str(state.get("reason", ""))

        return self._status_payload(
            status=status,
            disabled_until=disabled_until,
            reason=reason,
        )

    async def get_status_async(self) -> dict:
        return await self.store.run_async(self.get_status)

    def _mark_quota_exhausted(self, *, reason: str) -> dict:
        disabled_until = self._next_pacific_midnight_utc_iso()
        state = self.store.set_feature_state(
            feature_key=_FEATURE_KEY,
            status="disabled_quota_exhausted",
            disabled_until=disabled_until,
            reason=reason,
        )
        self.store.record_feature_event(
            user_id="system",
            feature_key=_FEATURE_KEY,
            event_name="video_panel_hidden_quota",
            metadata={
                "reason": reason,
                "disabled_until": disabled_until,
            },
        )
        return self._status_payload(
            status=str(state.get("status", "disabled_quota_exhausted")),
            disabled_until=disabled_until,
            reason=reason,
        )

    def _mark_reenabled(self) -> None:
        state = self.store.set_feature_state(
            feature_key=_FEATURE_KEY,
            status="enabled",
            disabled_until="",
            reason="",
        )
        self.store.record_feature_event(
            user_id="system",
            feature_key=_FEATURE_KEY,
            event_name="video_feature_reenabled",
            metadata={
                "status": state.get("status", "enabled"),
            },
        )

    @staticmethod
    def _cache_key(*, topic_id: str, section_heading: str, preferred_language: str, limit: int) -> str:
        blob = "|".join(
            [
                str(topic_id or "").strip().lower(),
                str(section_heading or "").strip().lower(),
                str(preferred_language or "").strip().lower(),
                str(int(limit)),
            ]
        )
        digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()
        return f"topic_videos:{digest}"

    @staticmethod
    def _build_query(*, topic_title: str, section_heading: str, preferred_language: str) -> str:
        parts = [
            topic_title.strip(),
            section_heading.strip(),
            "interview",
            "tutorial",
        ]
        if preferred_language.strip():
            parts.append(preferred_language.strip())
        return " ".join(part for part in parts if part)[:220]

    @staticmethod
    def _is_low_signal_title(title: str) -> bool:
        lowered = str(title or "").strip().lower()
        return "shorts" in lowered or lowered.startswith("#shorts")

    async def _youtube_request(self, *, endpoint: str, params: dict) -> tuple[int, dict]:
        url = f"https://www.googleapis.com/youtube/v3/{endpoint}"
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(url, params=params)
        payload = response.json() if response.text.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
        return response.status_code, payload

    async def _fetch_videos(
        self,
        *,
        topic_title: str,
        section_heading: str,
        preferred_language: str,
        limit: int,
    ) -> tuple[list[dict], str | None]:
        query = self._build_query(
            topic_title=topic_title,
            section_heading=section_heading,
            preferred_language=preferred_language,
        )
        search_params = {
            "key": self.settings.youtube_api_key.strip(),
            "part": "snippet",
            "type": "video",
            "maxResults": max(6, min(12, int(limit) * 3)),
            "q": query,
            "safeSearch": "moderate",
            "videoEmbeddable": "true",
        }
        status_code, search_payload = await self._youtube_request(endpoint="search", params=search_params)
        if status_code >= 400:
            if self._is_quota_error(status_code, search_payload):
                return [], "quota_exhausted"
            logger.warning("topic_videos_search_failed status=%s payload=%s", status_code, search_payload)
            return [], "search_failed"

        ids: list[str] = []
        live_flags: dict[str, str] = {}
        for item in search_payload.get("items", []):
            if not isinstance(item, dict):
                continue
            id_obj = item.get("id")
            if not isinstance(id_obj, dict):
                continue
            video_id = str(id_obj.get("videoId", "")).strip()
            if not video_id or video_id in ids:
                continue
            snippet = item.get("snippet") if isinstance(item.get("snippet"), dict) else {}
            live_flags[video_id] = str(snippet.get("liveBroadcastContent", "none"))
            ids.append(video_id)
            if len(ids) >= max(6, min(16, int(limit) * 5)):
                break

        if not ids:
            return [], None

        details_params = {
            "key": self.settings.youtube_api_key.strip(),
            "part": "snippet,contentDetails",
            "id": ",".join(ids),
            "maxResults": len(ids),
        }
        status_code, details_payload = await self._youtube_request(endpoint="videos", params=details_params)
        if status_code >= 400:
            if self._is_quota_error(status_code, details_payload):
                return [], "quota_exhausted"
            logger.warning("topic_videos_details_failed status=%s payload=%s", status_code, details_payload)
            return [], "details_failed"

        by_id: dict[str, dict] = {}
        for item in details_payload.get("items", []):
            if not isinstance(item, dict):
                continue
            video_id = str(item.get("id", "")).strip()
            if video_id:
                by_id[video_id] = item

        results: list[dict] = []
        seen: set[str] = set()
        for video_id in ids:
            item = by_id.get(video_id)
            if not item:
                continue
            snippet = item.get("snippet") if isinstance(item.get("snippet"), dict) else {}
            content_details = (
                item.get("contentDetails") if isinstance(item.get("contentDetails"), dict) else {}
            )
            title = str(snippet.get("title", "")).strip()
            channel = str(snippet.get("channelTitle", "")).strip()
            if not title or self._is_low_signal_title(title):
                continue
            if live_flags.get(video_id, "none") in {"live", "upcoming"}:
                continue
            duration_seconds = self._parse_duration_seconds(str(content_details.get("duration", "")))
            if duration_seconds <= 0:
                continue
            if duration_seconds < 240 or duration_seconds > 2100:
                continue

            dedup_key = f"{title.lower()}::{channel.lower()}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            thumbs = snippet.get("thumbnails") if isinstance(snippet.get("thumbnails"), dict) else {}
            thumb_url = ""
            for k in ("high", "medium", "default"):
                node = thumbs.get(k)
                if isinstance(node, dict):
                    thumb_url = str(node.get("url", "")).strip()
                if thumb_url:
                    break

            results.append(
                {
                    "video_id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "channel": channel,
                    "duration_seconds": duration_seconds,
                    "thumbnail_url": thumb_url,
                    "published_at": str(snippet.get("publishedAt", "")).strip(),
                    "source": "youtube",
                }
            )
            if len(results) >= int(limit):
                break

        return results[: int(limit)], None

    async def recommend(
        self,
        *,
        topic_id: str,
        topic_title: str,
        section_heading: str,
        preferred_language: str,
        limit: int,
        force_refresh: bool,
    ) -> dict:
        normalized_limit = max(1, min(5, int(limit or self.settings.topic_videos_default_limit or 3)))
        status = await self.get_status_async()
        status_name = str(status.get("status", "disabled_config"))

        if not status.get("enabled"):
            return {
                **status,
                "topic_id": topic_id,
                "section_index": 0,
                "section_heading": section_heading,
                "videos": [],
                "source": "",
                "cached": False,
            }

        cache_key = self._cache_key(
            topic_id=topic_id,
            section_heading=section_heading,
            preferred_language=preferred_language,
            limit=normalized_limit,
        )

        if not force_refresh and status_name != "probe":
            cached = await self.store.run_async(
                self.store.get_topic_video_cache,
                cache_key=cache_key,
                max_age_hours=int(self.settings.topic_videos_cache_ttl_hours or 168),
            )
            if cached is not None:
                payload = cached.get("payload") if isinstance(cached.get("payload"), dict) else {}
                videos = payload.get("videos") if isinstance(payload.get("videos"), list) else []
                return {
                    **status,
                    "topic_id": topic_id,
                    "section_index": 0,
                    "section_heading": section_heading,
                    "videos": videos,
                    "source": str(payload.get("source", "youtube")),
                    "cached": True,
                }

        videos, error_code = await self._fetch_videos(
            topic_title=topic_title,
            section_heading=section_heading,
            preferred_language=preferred_language,
            limit=normalized_limit,
        )
        if error_code == "quota_exhausted" and self.settings.topic_videos_disable_on_quota:
            disabled = await self.store.run_async(
                self._mark_quota_exhausted,
                reason="youtube_quota_exhausted",
            )
            return {
                **disabled,
                "topic_id": topic_id,
                "section_index": 0,
                "section_heading": section_heading,
                "videos": [],
                "source": "youtube",
                "cached": False,
            }

        if status_name == "probe" and error_code is None:
            await self.store.run_async(self._mark_reenabled)
            status = await self.get_status_async()

        response_payload = {
            "videos": videos,
            "source": "youtube",
        }
        await self.store.run_async(
            self.store.upsert_topic_video_cache,
            cache_key=cache_key,
            topic_id=topic_id,
            section_heading=section_heading,
            preferred_language=preferred_language,
            payload=response_payload,
        )

        return {
            **status,
            "topic_id": topic_id,
            "section_index": 0,
            "section_heading": section_heading,
            "videos": videos,
            "source": "youtube",
            "cached": False,
            "reason": "" if error_code is None else error_code,
        }
