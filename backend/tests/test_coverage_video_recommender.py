"""Coverage for app/services/video_recommender.py.

``httpx.AsyncClient`` is always replaced with a recording fake, so no socket is
opened. The existing ``tests/test_video_recommender.py`` covers the quota /
probe state machine via a stubbed ``_fetch_videos``; these tests cover the pure
helpers, the YouTube search+details parsing, and the cache and config gates.
"""

import asyncio
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from app.services import video_recommender as vr
from app.services.learning_store import LearningStore
from app.services.video_recommender import VideoRecommender


def _settings(**overrides):
    base = {
        "enable_topic_videos": True,
        "youtube_api_key": "yt-key",
        "topic_videos_default_limit": 3,
        "topic_videos_cache_ttl_hours": 168,
        "topic_videos_disable_on_quota": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeResponse:
    def __init__(self, status_code, payload, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else "body"

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Records requests and replays a queue of ``_FakeResponse`` objects."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, params=None):
        self.requests.append((url, dict(params or {})))
        if not self._responses:
            raise AssertionError(f"unexpected extra request to {url}")
        return self._responses.pop(0)


def _fake(responses):
    """A reusable fake client instance so tests can inspect ``requests``."""

    return _FakeAsyncClient(responses)


def _patch_client(responses_or_fake):
    """Patch ``httpx.AsyncClient`` (as the recommender sees it) with a fake.

    Accepts either a list of responses, in which case a fresh fake is built, or
    an existing ``_FakeAsyncClient`` the test wants to inspect afterwards.
    """

    fake = (
        responses_or_fake
        if isinstance(responses_or_fake, _FakeAsyncClient)
        else _FakeAsyncClient(responses_or_fake)
    )
    return patch.object(vr.httpx, "AsyncClient", return_value=fake)


def _search_item(video_id, title="A Talk", live="none", channel="Chan"):
    return {
        "id": {"videoId": video_id},
        "snippet": {
            "title": title,
            "channelTitle": channel,
            "liveBroadcastContent": live,
        },
    }


def _details_item(video_id, title="A Talk", duration="PT10M", channel="Chan", thumbs=None):
    snippet = {
        "title": title,
        "channelTitle": channel,
        "publishedAt": "2026-01-02T03:04:05Z",
        "thumbnails": thumbs if thumbs is not None else {},
    }
    return {
        "id": video_id,
        "snippet": snippet,
        "contentDetails": {"duration": duration},
    }


class _Store:
    """Temporary-file LearningStore, one per test."""

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        self.store = LearningStore(os.path.join(self._td.name, "learning.db"))
        return self.store

    def __exit__(self, *exc_info):
        self._td.cleanup()
        return False


class VideoRecommenderHelperTests(unittest.TestCase):
    def test_parse_duration_handles_the_iso_8601_subset_youtube_emits(self):
        self.assertEqual(VideoRecommender._parse_duration_seconds("PT10M"), 600)
        self.assertEqual(VideoRecommender._parse_duration_seconds("pt1h2m3s"), 3723)
        self.assertEqual(VideoRecommender._parse_duration_seconds("PT45S"), 45)
        self.assertEqual(VideoRecommender._parse_duration_seconds("P1DT2H"), 7200)

    def test_parse_duration_returns_zero_for_junk(self):
        self.assertEqual(VideoRecommender._parse_duration_seconds(""), 0)
        self.assertEqual(VideoRecommender._parse_duration_seconds(None), 0)
        self.assertEqual(VideoRecommender._parse_duration_seconds("10:30"), 0)
        self.assertEqual(VideoRecommender._parse_duration_seconds("P1X"), 0)

    def test_parse_iso_normalises_to_utc_and_assumes_utc_when_naive(self):
        naive = VideoRecommender._parse_iso("2026-03-04T05:06:07")
        self.assertEqual(naive.tzinfo, UTC)
        self.assertEqual(naive.hour, 5)

        offset = VideoRecommender._parse_iso("2026-03-04T05:06:07+02:00")
        self.assertEqual(offset.astimezone(UTC).hour, 3)

        self.assertIsNone(VideoRecommender._parse_iso("not-a-date"))
        self.assertIsNone(VideoRecommender._parse_iso(""))

    def test_error_reasons_only_reads_a_well_formed_error_block(self):
        payload = {"error": {"errors": [{"reason": "quotaExceeded"}, {"reason": "dailyLimitExceeded"}]}}
        self.assertEqual(
            VideoRecommender._error_reasons(payload),
            {"quotaexceeded", "dailylimitexceeded"},
        )

        self.assertEqual(VideoRecommender._error_reasons({}), set())
        self.assertEqual(VideoRecommender._error_reasons({"error": "nope"}), set())
        self.assertEqual(VideoRecommender._error_reasons({"error": {"errors": "nope"}}), set())
        self.assertEqual(
            VideoRecommender._error_reasons({"error": {"errors": ["skip-me", {"reason": "  "}, 7]}}),
            set(),
        )

    def test_is_quota_error_covers_429_and_the_403_reason_allowlist(self):
        self.assertTrue(VideoRecommender._is_quota_error(429, {}))
        self.assertTrue(
            VideoRecommender._is_quota_error(403, {"error": {"errors": [{"reason": "quotaExceeded"}]}})
        )
        self.assertTrue(
            VideoRecommender._is_quota_error(403, {"error": {"errors": [{"reason": "userRateLimitExceeded"}]}})
        )
        # 403 without a quota reason is an ordinary auth failure.
        self.assertFalse(
            VideoRecommender._is_quota_error(403, {"error": {"errors": [{"reason": "keyInvalid"}]}})
        )
        self.assertFalse(VideoRecommender._is_quota_error(500, {}))
        self.assertFalse(VideoRecommender._is_quota_error(404, {}))

    def test_cache_key_is_case_and_whitespace_insensitive_but_limit_sensitive(self):
        base = VideoRecommender._cache_key(
            topic_id="01-http", section_heading="Basics", preferred_language="en", limit=3
        )
        self.assertTrue(base.startswith("topic_videos:"))
        self.assertEqual(
            base,
            VideoRecommender._cache_key(
                topic_id=" 01-HTTP ", section_heading="BASICS", preferred_language="EN ", limit=3
            ),
        )
        self.assertNotEqual(
            base,
            VideoRecommender._cache_key(
                topic_id="01-http", section_heading="Basics", preferred_language="en", limit=4
            ),
        )

    def test_build_query_appends_language_only_when_present_and_is_truncated(self):
        self.assertEqual(
            VideoRecommender._build_query(
                topic_title="HTTP", section_heading="Basics", preferred_language="en"
            ),
            "HTTP Basics interview tutorial en",
        )
        self.assertEqual(
            VideoRecommender._build_query(
                topic_title="HTTP", section_heading="Basics", preferred_language="   "
            ),
            "HTTP Basics interview tutorial",
        )
        long_query = VideoRecommender._build_query(
            topic_title="x" * 400, section_heading="y", preferred_language=""
        )
        self.assertEqual(len(long_query), 220)

    def test_low_signal_titles_are_detected(self):
        self.assertTrue(VideoRecommender._is_low_signal_title("60 Second Shorts of HTTP"))
        self.assertTrue(VideoRecommender._is_low_signal_title("#Shorts"))
        self.assertFalse(VideoRecommender._is_low_signal_title("HTTP Retry Patterns"))
        self.assertFalse(VideoRecommender._is_low_signal_title(""))

    def test_next_pacific_midnight_is_the_following_local_midnight(self):
        # 2026-03-04T09:00Z is 01:00 PST -> next local midnight is the 5th.
        iso = VideoRecommender._next_pacific_midnight_utc_iso(
            datetime(2026, 3, 4, 9, 0, tzinfo=UTC)
        )
        self.assertEqual(iso, "2026-03-05T08:00:00+00:00")

    def test_status_payload_rejects_unknown_statuses_and_flags_probe_as_enabled(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            self.assertEqual(
                recommender._status_payload(status="bogus")["status"], "disabled_config"
            )
            probe = recommender._status_payload(status="probe")
            self.assertTrue(probe["enabled"])
            self.assertEqual(
                recommender._status_payload(status="disabled_quota_exhausted")["enabled"], False
            )


class VideoRecommenderStatusTests(unittest.TestCase):
    def test_feature_flag_off_short_circuits_before_touching_the_store(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings(enable_topic_videos=False))
            status = recommender.get_status()
            self.assertEqual(status["status"], "disabled_config")
            self.assertEqual(status["reason"], "feature_flag_off")
            self.assertIsNone(store.get_feature_state(feature_key="topic_videos"))

    def test_blank_api_key_is_reported_separately_from_the_flag(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings(youtube_api_key="   "))
            status = recommender.get_status()
            self.assertEqual(status["status"], "disabled_config")
            self.assertEqual(status["reason"], "youtube_api_key_missing")

    def test_first_status_read_seeds_the_feature_state_row(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            self.assertIsNone(store.get_feature_state(feature_key="topic_videos"))
            status = recommender.get_status()
            self.assertEqual(status["status"], "enabled")
            self.assertTrue(status["enabled"])
            self.assertIsNotNone(store.get_feature_state(feature_key="topic_videos"))

    def test_unparseable_disabled_until_keeps_the_quota_disable_in_place(self):
        with _Store() as store:
            store.set_feature_state(
                feature_key="topic_videos",
                status="disabled_quota_exhausted",
                disabled_until="not-a-timestamp",
                reason="youtube_quota_exhausted",
            )
            recommender = VideoRecommender(store, _settings())
            status = recommender.get_status()
            # A corrupt expiry must not be read as "expired" and re-enabled.
            self.assertEqual(status["status"], "disabled_quota_exhausted")
            self.assertFalse(status["enabled"])

    def test_config_disabled_when_everything_else_is_fine(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            self.assertTrue(recommender._config_enabled())
            self.assertFalse(
                VideoRecommender(store, _settings(enable_topic_videos=False))._config_enabled()
            )


class VideoRecommenderFetchTests(unittest.TestCase):
    def test_search_and_details_are_merged_into_watch_urls(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            fake = _fake(
                [
                    _FakeResponse(200, {"items": [_search_item("v1"), _search_item("v2")]}),
                    _FakeResponse(
                        200,
                        {
                            "items": [
                                _details_item("v1", title="HTTP Retry Patterns", duration="PT10M"),
                                _details_item("v2", title="Caching Deep Dive", duration="PT20M"),
                            ]
                        },
                    ),
                ]
            )
            with _patch_client(fake):
                videos, error = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="en",
                        limit=5,
                    )
                )

            self.assertIsNone(error)
            self.assertEqual([v["video_id"] for v in videos], ["v1", "v2"])
            self.assertEqual(videos[0]["url"], "https://www.youtube.com/watch?v=v1")
            self.assertEqual(videos[0]["duration_seconds"], 600)
            self.assertEqual(videos[0]["source"], "youtube")
            self.assertEqual(videos[0]["published_at"], "2026-01-02T03:04:05Z")

            search_url, search_params = fake.requests[0]
            self.assertEqual(search_url, "https://www.googleapis.com/youtube/v3/search")
            self.assertEqual(search_params["key"], "yt-key")
            self.assertEqual(search_params["maxResults"], 12)  # min(12, 5*3)
            self.assertEqual(search_params["q"], "HTTP Basics interview tutorial en")

            details_url, details_params = fake.requests[1]
            self.assertEqual(details_url, "https://www.googleapis.com/youtube/v3/videos")
            self.assertEqual(details_params["id"], "v1,v2")
            self.assertEqual(details_params["part"], "snippet,contentDetails")

    def test_search_failure_that_is_not_a_quota_error_reports_search_failed(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            with _patch_client(
                [_FakeResponse(500, {"error": {"message": "backendError"}})]
            ):
                videos, error = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                    )
                )
            self.assertEqual(videos, [])
            self.assertEqual(error, "search_failed")

    def test_quota_error_on_search_is_reported_as_quota_exhausted(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            payload = {"error": {"errors": [{"reason": "quotaExceeded"}]}}
            with _patch_client([_FakeResponse(403, payload)]):
                videos, error = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                    )
                )
            self.assertEqual(videos, [])
            self.assertEqual(error, "quota_exhausted")

    def test_details_failure_is_reported_separately_from_search_failure(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            with _patch_client(
                [
                    _FakeResponse(200, {"items": [_search_item("v1")]}),
                    _FakeResponse(404, {"error": {"message": "notFound"}}),
                ]
            ):
                videos, error = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                    )
                )
            self.assertEqual(videos, [])
            self.assertEqual(error, "details_failed")

    def test_quota_error_on_the_details_call_is_still_quota_exhausted(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            with _patch_client(
                [
                    _FakeResponse(200, {"items": [_search_item("v1")]}),
                    _FakeResponse(429, {}),
                ]
            ):
                videos, error = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                    )
                )
            self.assertEqual(videos, [])
            self.assertEqual(error, "quota_exhausted")

    def test_search_without_usable_ids_skips_the_details_call(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            fake = _fake([_FakeResponse(200, {"items": []})])
            with _patch_client(fake):
                videos, error = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                    )
                )
            self.assertEqual(videos, [])
            # No ids survived parsing, so nothing was normalised to an error.
            self.assertIsNone(error)
            self.assertEqual(len(fake.requests), 1)

    def test_malformed_search_items_are_skipped(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            payload = {
                "items": [
                    "not-a-dict",
                    {"no_id_key": True},
                    {"id": {"videoId": "   "}},
                    {"id": {"videoId": "dup"}, "snippet": {}},
                    {"id": {"videoId": "dup"}, "snippet": {}},
                    {"id": {"videoId": "keep"}, "snippet": "not-a-dict"},
                ]
            }
            with _patch_client(
                [
                    _FakeResponse(200, payload),
                    _FakeResponse(200, {"items": [_details_item("keep", duration="PT9M")]}),
                ]
            ):
                videos, error = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                    )
                )
            self.assertIsNone(error)
            self.assertEqual([v["video_id"] for v in videos], ["keep"])

    def test_empty_body_is_treated_as_an_empty_payload(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            fake = _fake([_FakeResponse(200, [], text="   ")])
            with _patch_client(fake):
                status_code, payload = asyncio.run(
                    recommender._youtube_request(endpoint="search", params={"key": "k"})
                )
            self.assertEqual(status_code, 200)
            self.assertEqual(payload, {})

    def test_non_dict_json_body_is_coerced_to_an_empty_payload(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            fake = _fake([_FakeResponse(200, ["a", "list"], text="[]")])
            with _patch_client(fake):
                status_code, payload = asyncio.run(
                    recommender._youtube_request(endpoint="search", params={"key": "k"})
                )
            self.assertEqual(status_code, 200)
            self.assertEqual(payload, {})

    def test_low_signal_live_and_out_of_range_videos_are_dropped(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            search_items = [
                _search_item("short", title="HTTP #Shorts"),
                _search_item("live", live="live"),
                _search_item("soon", live="upcoming"),
                _search_item("tiny", title="Too Short"),
                _search_item("huge", title="Way Too Long"),
                _search_item("zero", title="No Duration"),
                _search_item("good", title="HTTP Retry Patterns"),
                _search_item("dup", title="HTTP Retry Patterns", channel="Chan"),
            ]
            details = {
                "items": [
                    _details_item("short", title="HTTP #Shorts"),
                    _details_item("live", title="Live Now"),
                    _details_item("soon", title="Coming Up"),
                    _details_item("tiny", title="Too Short", duration="PT2M"),
                    _details_item("huge", title="Way Too Long", duration="PT60M"),
                    _details_item("zero", title="No Duration", duration=""),
                    _details_item("good", title="HTTP Retry Patterns", duration="PT10M"),
                    _details_item("dup", title="HTTP Retry Patterns", duration="PT11M"),
                    {"id": "missing", "snippet": None, "contentDetails": None},
                ]
            }
            with _patch_client(
                [_FakeResponse(200, {"items": search_items}), _FakeResponse(200, details)]
            ):
                videos, error = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=5,
                    )
                )
            self.assertIsNone(error)
            # "dup" repeats title+channel of "good", so de-duplication keeps one.
            self.assertEqual([v["video_id"] for v in videos], ["good"])

    def test_search_ids_are_capped_by_limit(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            search_items = [_search_item(f"v{i}") for i in range(1, 21)]
            details = {
                "items": [
                    _details_item(f"v{i}", title=f"Talk {i}", duration="PT10M")
                    for i in range(1, 7)
                ]
            }
            fake = _fake(
                [_FakeResponse(200, {"items": search_items}), _FakeResponse(200, details)]
            )
            with _patch_client(fake):
                videos, error = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=1,
                    )
                )
            self.assertIsNone(error)
            self.assertEqual([v["video_id"] for v in videos], ["v1"])
            # Only max(6, min(16, limit * 5)) ids are carried into the details call.
            self.assertEqual(fake.requests[1][1]["id"], "v1,v2,v3,v4,v5,v6")

    def test_thumbnail_preference_order_is_high_medium_default(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            thumbs = {
                "high": {"url": "https://img/high.jpg"},
                "medium": {"url": "https://img/medium.jpg"},
                "default": {"url": "https://img/default.jpg"},
            }
            with _patch_client(
                [
                    _FakeResponse(200, {"items": [_search_item("v1")]}),
                    _FakeResponse(200, {"items": [_details_item("v1", thumbs=thumbs)]}),
                ]
            ):
                videos, _ = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                    )
                )
            self.assertEqual(videos[0]["thumbnail_url"], "https://img/high.jpg")

    def test_missing_thumbnail_nodes_leave_the_url_empty(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            with _patch_client(
                [
                    _FakeResponse(200, {"items": [_search_item("v1")]}),
                    _FakeResponse(200, {"items": [_details_item("v1", thumbs={"high": "bad"})]}),
                ]
            ):
                videos, _ = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                    )
                )
            self.assertEqual(videos[0]["thumbnail_url"], "")

    def test_details_items_without_a_usable_id_are_ignored(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            details = {
                "items": [
                    "not-a-dict",
                    {"snippet": {"title": "No Id"}, "contentDetails": {"duration": "PT10M"}},
                    {"id": "  ", "snippet": {"title": "Blank Id"}},
                    _details_item("v1", title="HTTP Retry Patterns", duration="PT10M"),
                ]
            }
            with _patch_client(
                [
                    _FakeResponse(200, {"items": [_search_item("v1")]}),
                    _FakeResponse(200, details),
                ]
            ):
                videos, error = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                    )
                )
            self.assertIsNone(error)
            self.assertEqual([v["video_id"] for v in videos], ["v1"])

    def test_search_id_cap_uses_the_inner_floor_for_small_limits(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            # limit=0 is clamped to 1 by the caller, but the helper must still
            # floor the search page at 6 results.
            self.assertEqual(max(6, min(12, 1 * 3)), 6)
            self.assertEqual(max(6, min(16, 1 * 5)), 6)
            self.assertEqual(max(6, min(12, 0 * 3)), 6)

    def test_results_are_truncated_to_the_requested_limit(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            search_items = [_search_item(f"v{i}") for i in range(1, 5)]
            details = {
                "items": [
                    _details_item(f"v{i}", title=f"Talk {i}", duration="PT10M") for i in range(1, 5)
                ]
            }
            with _patch_client(
                [_FakeResponse(200, {"items": search_items}), _FakeResponse(200, details)]
            ):
                videos, _ = asyncio.run(
                    recommender._fetch_videos(
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=2,
                    )
                )
            self.assertEqual([v["video_id"] for v in videos], ["v1", "v2"])


class VideoRecommenderRecommendTests(unittest.TestCase):
    def test_disabled_config_returns_empty_payload_without_fetching(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings(enable_topic_videos=False))
            fake = _fake([])
            with _patch_client(fake):
                out = asyncio.run(
                    recommender.recommend(
                        topic_id="01-http",
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                        force_refresh=False,
                    )
                )

            self.assertFalse(out["enabled"])
            self.assertEqual(out["videos"], [])
            self.assertEqual(out["source"], "")
            self.assertFalse(out["cached"])
            self.assertEqual(out["topic_id"], "01-http")
            self.assertEqual(fake.requests, [])

    def test_second_call_is_served_from_the_cache(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            fake = _fake(
                [
                    _FakeResponse(200, {"items": [_search_item("v1")]}),
                    _FakeResponse(
                        200, {"items": [_details_item("v1", title="HTTP Retry Patterns")]}
                    ),
                ]
            )
            with _patch_client(fake):
                first = asyncio.run(
                    recommender.recommend(
                        topic_id="01-http",
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                        force_refresh=False,
                    )
                )
                second = asyncio.run(
                    recommender.recommend(
                        topic_id="01-http",
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                        force_refresh=False,
                    )
                )

            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(len(fake.requests), 2)
            self.assertEqual(second["videos"], first["videos"])
            self.assertEqual(second["source"], "youtube")

    def test_force_refresh_bypasses_a_warm_cache(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            store.upsert_topic_video_cache(
                cache_key=VideoRecommender._cache_key(
                    topic_id="01-http",
                    section_heading="Basics",
                    preferred_language="",
                    limit=3,
                ),
                topic_id="01-http",
                section_heading="Basics",
                preferred_language="",
                payload={"videos": [{"video_id": "stale"}], "source": "youtube"},
            )
            fake = _fake(
                [
                    _FakeResponse(200, {"items": [_search_item("fresh")]}),
                    _FakeResponse(
                        200, {"items": [_details_item("fresh", title="Fresh Talk", duration="PT10M")]}
                    ),
                ]
            )
            with _patch_client(fake):
                out = asyncio.run(
                    recommender.recommend(
                        topic_id="01-http",
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                        force_refresh=True,
                    )
                )

            self.assertFalse(out["cached"])
            self.assertEqual([v["video_id"] for v in out["videos"]], ["fresh"])

    def test_expired_cache_entry_is_refetched(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings(topic_videos_cache_ttl_hours=1))
            cache_key = VideoRecommender._cache_key(
                topic_id="01-http", section_heading="Basics", preferred_language="", limit=3
            )
            store.upsert_topic_video_cache(
                cache_key=cache_key,
                topic_id="01-http",
                section_heading="Basics",
                preferred_language="",
                payload={"videos": [{"video_id": "stale"}], "source": "youtube"},
            )
            stale = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
            store._conn.execute(
                "UPDATE topic_video_cache SET fetched_at = ? WHERE cache_key = ?", (stale, cache_key)
            )

            fake = _fake(
                [
                    _FakeResponse(200, {"items": [_search_item("fresh")]}),
                    _FakeResponse(
                        200, {"items": [_details_item("fresh", title="Fresh Talk", duration="PT10M")]}
                    ),
                ]
            )
            with _patch_client(fake):
                out = asyncio.run(
                    recommender.recommend(
                        topic_id="01-http",
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                        force_refresh=False,
                    )
                )

            self.assertFalse(out["cached"])
            self.assertEqual([v["video_id"] for v in out["videos"]], ["fresh"])

    def test_corrupt_cache_payload_is_treated_as_a_miss(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            cache_key = VideoRecommender._cache_key(
                topic_id="01-http", section_heading="Basics", preferred_language="", limit=3
            )
            store.upsert_topic_video_cache(
                cache_key=cache_key,
                topic_id="01-http",
                section_heading="Basics",
                preferred_language="",
                payload={"videos": []},
            )
            store._conn.execute(
                "UPDATE topic_video_cache SET payload_json = ? WHERE cache_key = ?",
                ("{not json", cache_key),
            )
            fake = _fake(
                [
                    _FakeResponse(200, {"items": [_search_item("fresh")]}),
                    _FakeResponse(
                        200, {"items": [_details_item("fresh", title="Fresh Talk", duration="PT10M")]}
                    ),
                ]
            )
            with _patch_client(fake):
                out = asyncio.run(
                    recommender.recommend(
                        topic_id="01-http",
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                        force_refresh=False,
                    )
                )
            self.assertFalse(out["cached"])
            self.assertEqual([v["video_id"] for v in out["videos"]], ["fresh"])

    def test_non_quota_fetch_error_still_caches_and_reports_the_reason(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            fake = _fake([_FakeResponse(500, {"error": {"message": "backendError"}})])
            with _patch_client(fake):
                out = asyncio.run(
                    recommender.recommend(
                        topic_id="01-http",
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                        force_refresh=True,
                    )
                )
                cached = asyncio.run(
                    store.run_async(
                        store.get_topic_video_cache,
                        cache_key=VideoRecommender._cache_key(
                            topic_id="01-http",
                            section_heading="Basics",
                            preferred_language="",
                            limit=3,
                        ),
                        max_age_hours=168,
                    )
                )

            self.assertEqual(out["videos"], [])
            self.assertEqual(out["reason"], "search_failed")
            self.assertTrue(out["enabled"])
            # The empty result is cached, so a later panel render is not a new call.
            self.assertIsNotNone(cached)
            self.assertEqual(cached["payload"], {"videos": [], "source": "youtube"})

    def test_quota_error_is_not_fatal_when_disable_on_quota_is_off(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings(topic_videos_disable_on_quota=False))
            fake = _fake([_FakeResponse(429, {})])
            with _patch_client(fake):
                out = asyncio.run(
                    recommender.recommend(
                        topic_id="01-http",
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=3,
                        force_refresh=True,
                    )
                )

            self.assertTrue(out["enabled"])
            self.assertEqual(out["status"], "enabled")
            self.assertEqual(out["reason"], "quota_exhausted")

    def test_limit_is_clamped_into_one_to_five(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings())
            fake = _fake(
                [
                    _FakeResponse(
                        200, {"items": [_search_item(f"v{i}") for i in range(1, 8)]}
                    ),
                    _FakeResponse(
                        200,
                        {
                            "items": [
                                _details_item(f"v{i}", title=f"Talk {i}", duration="PT10M")
                                for i in range(1, 8)
                            ]
                        },
                    ),
                ]
            )
            with _patch_client(fake):
                out = asyncio.run(
                    recommender.recommend(
                        topic_id="01-http",
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=99,
                        force_refresh=True,
                    )
                )

            self.assertEqual(len(out["videos"]), 5)
            self.assertEqual(fake.requests[0][1]["maxResults"], 12)

    def test_zero_limit_falls_back_to_the_configured_default(self):
        with _Store() as store:
            recommender = VideoRecommender(store, _settings(topic_videos_default_limit=2))
            fake = _fake(
                [
                    _FakeResponse(
                        200, {"items": [_search_item(f"v{i}") for i in range(1, 5)]}
                    ),
                    _FakeResponse(
                        200,
                        {
                            "items": [
                                _details_item(f"v{i}", title=f"Talk {i}", duration="PT10M")
                                for i in range(1, 5)
                            ]
                        },
                    ),
                ]
            )
            with _patch_client(fake):
                out = asyncio.run(
                    recommender.recommend(
                        topic_id="01-http",
                        topic_title="HTTP",
                        section_heading="Basics",
                        preferred_language="",
                        limit=0,
                        force_refresh=True,
                    )
                )
            self.assertEqual(len(out["videos"]), 2)


if __name__ == "__main__":
    unittest.main()
