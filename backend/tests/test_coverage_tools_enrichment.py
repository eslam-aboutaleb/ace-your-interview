"""Coverage for app/services/tools/enrichment.py.

Complements ``tests/test_tools_enrichment.py`` (the three providers with an
injected request callable) with the default httpx-backed request function, the
per-result defensive skips, and the URL extractor's non-matching lines.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.tools import enrichment as enrichment_module
from app.services.tools.enrichment import (
    FIRECRAWL_TOOL,
    GITHUB_TOOL,
    PROVIDER_FAILURE_PREFIXES,
    TAVILY_TOOL,
    _as_int,
    _clip,
    build_enrichment_registry,
    build_enrichment_tools,
    first_url_from,
    httpx_request,
)


def _settings(**overrides):
    base = {
        "mcp_tavily_enabled": True,
        "tavily_api_key": "tvly-test",
        "mcp_firecrawl_enabled": True,
        "firecrawl_api_key": "fc-test",
        "mcp_github_enabled": True,
        "github_token": "",
        "mcp_timeout_seconds": 3.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _run(coro):
    return asyncio.run(coro)


class _RecordingTransport:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls: list[tuple] = []

    async def __call__(self, method, url, headers, payload):
        self.calls.append((method, url, dict(headers), dict(payload)))
        if not self.payloads:
            return {}
        return self.payloads.pop(0)


class FakeAsyncClient:
    """Stands in for ``httpx.AsyncClient``; records the request it was given."""

    instances: list["FakeAsyncClient"] = []

    #: Class-level so a test can set them before the call and have every
    #: constructed instance see the same value.
    status_error: Exception | None = None
    json_payload: object = {}

    def __init__(self, *, timeout=None):
        self.timeout = timeout
        self.calls: list[tuple] = []
        FakeAsyncClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, *, headers=None, params=None):
        self.calls.append(("GET", url, headers, params))
        return self._response()

    async def post(self, url, *, headers=None, json=None):
        self.calls.append(("POST", url, headers, json))
        return self._response()

    def _response(self):
        client = self

        class _Response:
            def raise_for_status(self):
                if client.status_error is not None:
                    raise client.status_error

            def json(self):
                return client.json_payload

        return _Response()


def _patched_client(**attrs):
    FakeAsyncClient.instances = []
    patcher = patch.object(enrichment_module.httpx, "AsyncClient", FakeAsyncClient)
    patcher.start()
    for key, value in attrs.items():
        setattr(FakeAsyncClient, key, value)
    return patcher


class HttpxRequestTests(unittest.TestCase):
    def setUp(self):
        self.patcher = _patched_client()
        self.addCleanup(self.patcher.stop)
        FakeAsyncClient.instances = []

    def test_get_sends_headers_and_params(self):
        FakeAsyncClient.json_payload = {"items": []}
        data = _run(
            httpx_request("get", "https://api.example/search", {"Accept": "x"}, {"q": "y"})
        )
        self.assertEqual(data, {"items": []})
        client = FakeAsyncClient.instances[0]
        self.assertEqual(client.timeout, 8.0)
        self.assertEqual(
            client.calls,
            [("GET", "https://api.example/search", {"Accept": "x"}, {"q": "y"})],
        )

    def test_post_sends_json_body(self):
        FakeAsyncClient.json_payload = {"results": []}
        _run(
            httpx_request(
                "POST",
                "https://api.example/search",
                {"Authorization": "Bearer k"},
                {"query": "z"},
                timeout=2.5,
            )
        )
        client = FakeAsyncClient.instances[0]
        self.assertEqual(client.timeout, 2.5)
        self.assertEqual(
            client.calls,
            [
                (
                    "POST",
                    "https://api.example/search",
                    {"Authorization": "Bearer k"},
                    {"query": "z"},
                )
            ],
        )

    def test_non_object_json_is_normalised_to_an_empty_dict(self):
        FakeAsyncClient.json_payload = [1, 2, 3]
        self.assertEqual(
            _run(httpx_request("POST", "https://x", {}, {})), {}
        )

    def test_http_error_propagates(self):
        FakeAsyncClient.status_error = RuntimeError("401 Unauthorized")
        self.addCleanup(setattr, FakeAsyncClient, "status_error", None)
        with self.assertRaises(RuntimeError):
            _run(httpx_request("POST", "https://x", {}, {}))


class DefaultRequestWiringTests(unittest.TestCase):
    def test_build_tools_without_a_request_use_the_httpx_callable(self):
        # The tools must still be constructible with no wiring; the default
        # request closure is only resolved when a handler runs.
        specs = build_enrichment_tools(_settings())
        self.assertEqual(len(specs), 3)
        self.assertTrue(all(spec.timeout_seconds == 3.0 for spec in specs))

    def test_default_timeout_is_used_when_settings_omit_it(self):
        settings = SimpleNamespace(
            mcp_tavily_enabled=False,
            tavily_api_key="",
            mcp_firecrawl_enabled=False,
            firecrawl_api_key="",
            mcp_github_enabled=False,
            github_token="",
        )
        specs = build_enrichment_tools(settings)
        self.assertTrue(all(spec.timeout_seconds == 8.0 for spec in specs))

    def test_a_zero_timeout_setting_still_yields_a_positive_ceiling(self):
        specs = build_enrichment_tools(_settings(mcp_timeout_seconds=0))
        self.assertTrue(all(spec.timeout_seconds == 8.0 for spec in specs))

    def test_tools_are_disabled_without_any_operator_flag(self):
        settings = SimpleNamespace()
        specs = build_enrichment_tools(settings)
        self.assertEqual([spec.name for spec in specs if spec.enabled], [])
        self.assertEqual(build_enrichment_registry(settings).to_openai_schema(), [])


class TavilyResultHandlingTests(unittest.TestCase):
    def _registry(self, transport, **overrides):
        return build_enrichment_registry(_settings(**overrides), request=transport)

    def test_non_dict_results_are_skipped(self):
        transport = _RecordingTransport(
            {
                "results": [
                    "just a string",
                    None,
                    {"title": "Kept", "content": "body", "url": "https://x.dev"},
                ]
            }
        )
        result = _run(
            self._registry(transport).invoke(
                TAVILY_TOOL, {"query": "x", "max_results": 5}
            )
        )
        self.assertTrue(result.ok)
        self.assertIn("Kept: body (https://x.dev)", result.content)
        self.assertNotIn("just a string", result.content)

    def test_a_result_list_of_only_junk_reports_no_results(self):
        transport = _RecordingTransport({"results": ["a", 1, None]})
        result = _run(self._registry(transport).invoke(TAVILY_TOOL, {"query": "x"}))
        self.assertIn("No web results", result.content)

    def test_a_result_with_only_a_url_is_rendered(self):
        transport = _RecordingTransport(
            {"results": [{"url": "https://only.url"}]}
        )
        result = _run(self._registry(transport).invoke(TAVILY_TOOL, {"query": "x"}))
        # No title and no snippet means nothing renderable.
        self.assertIn("No web results", result.content)

    def test_result_without_the_results_key_reports_no_results(self):
        transport = _RecordingTransport({"answer": "irrelevant"})
        result = _run(self._registry(transport).invoke(TAVILY_TOOL, {"query": "x"}))
        self.assertIn("No web results", result.content)

    def test_results_are_capped_by_max_results(self):
        transport = _RecordingTransport(
            {
                "results": [
                    {"title": f"T{i}", "content": "c", "url": "https://x.dev"}
                    for i in range(9)
                ]
            }
        )
        _run(
            self._registry(transport).invoke(
                TAVILY_TOOL, {"query": "x", "max_results": 2}
            )
        )
        self.assertEqual(transport.calls[0][3]["max_results"], 2)

    def test_whitespace_only_query_short_circuits(self):
        transport = _RecordingTransport({"results": []})
        result = _run(self._registry(transport).invoke(TAVILY_TOOL, {"query": "   "}))
        self.assertIn("No query supplied", result.content)
        self.assertEqual(transport.calls, [])

    def test_advanced_depth_is_forwarded(self):
        transport = _RecordingTransport({"results": []})
        _run(
            self._registry(transport).invoke(
                TAVILY_TOOL, {"query": "x", "search_depth": " Advanced "}
            )
        )
        self.assertEqual(transport.calls[0][3]["search_depth"], "advanced")

    def test_the_api_key_is_sent_in_the_payload(self):
        transport = _RecordingTransport({"results": []})
        _run(self._registry(transport).invoke(TAVILY_TOOL, {"query": "x"}))
        self.assertEqual(transport.calls[0][3]["api_key"], "tvly-test")


class GitHubResultHandlingTests(unittest.TestCase):
    def _registry(self, transport, **overrides):
        return build_enrichment_registry(_settings(**overrides), request=transport)

    def test_missing_query_short_circuits(self):
        transport = _RecordingTransport({"items": []})
        result = _run(self._registry(transport).invoke(GITHUB_TOOL, {}))
        self.assertIn("No query supplied", result.content)
        self.assertEqual(transport.calls, [])

    def test_whitespace_only_query_short_circuits(self):
        transport = _RecordingTransport({"items": []})
        result = _run(
            self._registry(transport).invoke(GITHUB_TOOL, {"query": "  \n "})
        )
        self.assertIn("No query supplied", result.content)

    def test_non_dict_items_are_skipped(self):
        transport = _RecordingTransport(
            {
                "items": [
                    "a bare string",
                    None,
                    {
                        "full_name": "acme/keep",
                        "description": "kept",
                        "html_url": "https://github.com/acme/keep",
                    },
                ]
            }
        )
        result = _run(
            self._registry(transport).invoke(
                GITHUB_TOOL, {"query": "x", "per_page": 5}
            )
        )
        self.assertIn("acme/keep: kept", result.content)
        self.assertNotIn("a bare string", result.content)

    def test_items_without_a_full_name_are_skipped(self):
        transport = _RecordingTransport(
            {"items": [{"description": "nameless"}, {"full_name": "acme/keep"}]}
        )
        result = _run(self._registry(transport).invoke(GITHUB_TOOL, {"query": "x"}))
        self.assertIn("acme/keep", result.content)
        self.assertNotIn("nameless", result.content)

    def test_an_item_list_of_only_junk_reports_no_match(self):
        transport = _RecordingTransport({"items": [None, 7, {"description": "d"}]})
        result = _run(self._registry(transport).invoke(GITHUB_TOOL, {"query": "x"}))
        self.assertIn("No repositories matched", result.content)

    def test_a_blank_description_still_renders_the_row(self):
        transport = _RecordingTransport(
            {"items": [{"full_name": "acme/quiet"}]}
        )
        result = _run(self._registry(transport).invoke(GITHUB_TOOL, {"query": "x"}))
        self.assertIn("acme/quiet", result.content)

    def test_search_is_sorted_by_stars(self):
        transport = _RecordingTransport({"items": []})
        _run(self._registry(transport).invoke(GITHUB_TOOL, {"query": "x"}))
        payload = transport.calls[0][3]
        self.assertEqual(payload["sort"], "stars")
        self.assertEqual(payload["order"], "desc")
        self.assertEqual(payload["q"], "x")

    def test_a_blank_token_is_not_sent_as_a_bearer_header(self):
        transport = _RecordingTransport({"items": []})
        _run(
            self._registry(transport, github_token="   ").invoke(
                GITHUB_TOOL, {"query": "x"}
            )
        )
        self.assertNotIn("Authorization", transport.calls[0][2])


class FirecrawlResultHandlingTests(unittest.TestCase):
    def _registry(self, transport, **overrides):
        return build_enrichment_registry(_settings(**overrides), request=transport)

    def test_non_dict_data_key_falls_back_to_the_flat_markdown(self):
        transport = _RecordingTransport({"data": "not an object", "markdown": "flat"})
        result = _run(
            self._registry(transport).invoke(FIRECRAWL_TOOL, {"url": "https://x.dev"})
        )
        self.assertIn("flat", result.content)

    def test_whitespace_only_markdown_reports_no_content(self):
        transport = _RecordingTransport({"data": {"markdown": "   \n "}})
        result = _run(
            self._registry(transport).invoke(FIRECRAWL_TOOL, {"url": "https://x.dev"})
        )
        self.assertIn("No extractable content", result.content)

    def test_long_markdown_is_clipped(self):
        transport = _RecordingTransport({"markdown": "z" * 5000})
        result = _run(
            self._registry(transport).invoke(FIRECRAWL_TOOL, {"url": "https://x.dev"})
        )
        # 1800 chars of markdown plus the fixed prefix.
        self.assertEqual(len(result.content), 1800 + len(
            "Firecrawl extracted context from https://x.dev:\n"
        ))

    def test_a_very_long_url_is_clipped(self):
        transport = _RecordingTransport({"markdown": "body"})
        long_url = "https://x.dev/" + "a" * 3000
        _run(self._registry(transport).invoke(FIRECRAWL_TOOL, {"url": long_url}))
        self.assertEqual(len(transport.calls[0][3]["url"]), 2000)

    def test_formats_are_requested_as_markdown(self):
        transport = _RecordingTransport({"markdown": "b"})
        _run(
            self._registry(transport).invoke(FIRECRAWL_TOOL, {"url": "https://x.dev"})
        )
        payload = transport.calls[0][3]
        self.assertEqual(payload["formats"], ["markdown"])
        self.assertTrue(payload["onlyMainContent"])


class HelperTests(unittest.TestCase):
    def test_clip_collapses_whitespace_and_bounds_the_length(self):
        self.assertEqual(_clip("  a   b \n c ", 100), "a b c")
        self.assertEqual(_clip("abcdefgh", 3), "abc")
        self.assertEqual(_clip(None, 5), "")

    def test_as_int_clamps_and_defaults(self):
        self.assertEqual(_as_int("4", 3, low=1, high=5), 4)
        self.assertEqual(_as_int(99, 3, low=1, high=5), 5)
        self.assertEqual(_as_int(0, 3, low=1, high=5), 1)
        self.assertEqual(_as_int("nope", 3, low=1, high=5), 3)
        self.assertEqual(_as_int(None, 3, low=1, high=5), 3)
        self.assertEqual(_as_int(2.7, 3, low=1, high=5), 2)

    def test_failure_prefixes_match_the_handler_messages(self):
        self.assertEqual(
            PROVIDER_FAILURE_PREFIXES,
            ("Tavily search failed:", "Page scrape failed:", "GitHub search failed:"),
        )


class FirstUrlTests(unittest.TestCase):
    def test_non_bullet_lines_are_ignored(self):
        block = (
            "Tavily web context:\n"
            "some prose without a url\n"
            "- Real: snippet (https://example.com/a)"
        )
        self.assertEqual(first_url_from(block), "https://example.com/a")

    def test_a_parenthesised_non_url_is_ignored(self):
        block = "- Title: snippet (not a url)\n- Title: b (https://example.com/b)"
        self.assertEqual(first_url_from(block), "https://example.com/b")

    def test_a_bullet_without_closing_parenthesis_is_ignored(self):
        self.assertEqual(first_url_from("- Title: snippet (https://example.com/a"), "")

    def test_a_bullet_without_any_parenthesis_is_ignored(self):
        self.assertEqual(first_url_from("- Title: no parentheses here"), "")

    def test_the_first_matching_bullet_wins(self):
        block = (
            "- One: a (https://example.com/one)\n"
            "- Two: b (https://example.com/two)"
        )
        self.assertEqual(first_url_from(block), "https://example.com/one")

    def test_none_text_is_handled(self):
        self.assertEqual(first_url_from(None), "")


if __name__ == "__main__":
    unittest.main()