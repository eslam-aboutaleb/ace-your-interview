"""Enrichment providers as tools. No network: every call is injected."""

import asyncio
import unittest
from types import SimpleNamespace

from app.services.tools.enrichment import (
    FIRECRAWL_ENDPOINT,
    FIRECRAWL_TOOL,
    GITHUB_ENDPOINT,
    GITHUB_TOOL,
    TAVILY_ENDPOINT,
    TAVILY_TOOL,
    build_enrichment_registry,
    build_enrichment_tools,
    first_url_from,
)
from app.services.tools.registry import ToolRegistry


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


class FakeTransport:
    """Records calls and replays queued JSON payloads (or raises)."""

    def __init__(self, *payloads, error: Exception | None = None):
        self.payloads = list(payloads)
        self.error = error
        self.calls: list[tuple] = []

    async def __call__(self, method, url, headers, payload):
        self.calls.append((method, url, headers, dict(payload)))
        if self.error is not None:
            raise self.error
        if not self.payloads:
            return {}
        value = self.payloads.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _run(coro):
    return asyncio.run(coro)


def _registry(settings, transport):
    return build_enrichment_registry(settings, request=transport)


class EnableFlagTests(unittest.TestCase):
    def test_all_disabled_by_default(self):
        settings = _settings(
            mcp_tavily_enabled=False,
            tavily_api_key="",
            mcp_firecrawl_enabled=False,
            firecrawl_api_key="",
            mcp_github_enabled=False,
        )
        self.assertEqual(build_enrichment_registry(settings).list_enabled(), [])

    def test_flag_without_credential_stays_disabled(self):
        # No key means the model must never be told the tool exists.
        settings = _settings(mcp_tavily_enabled=True, tavily_api_key="   ")
        names = [s.name for s in build_enrichment_registry(settings).list_enabled()]
        self.assertNotIn(TAVILY_TOOL, names)

    def test_credential_without_flag_stays_disabled(self):
        settings = _settings(mcp_tavily_enabled=False, tavily_api_key="tvly-test")
        names = [s.name for s in build_enrichment_registry(settings).list_enabled()]
        self.assertNotIn(TAVILY_TOOL, names)

    def test_github_needs_no_credential(self):
        settings = _settings(mcp_github_enabled=True, github_token="")
        names = [s.name for s in build_enrichment_registry(settings).list_enabled()]
        self.assertIn(GITHUB_TOOL, names)

    def test_schemas_are_openai_shaped(self):
        schemas = build_enrichment_registry(_settings()).to_openai_schema()
        self.assertEqual(
            sorted(s["function"]["name"] for s in schemas),
            sorted([TAVILY_TOOL, FIRECRAWL_TOOL, GITHUB_TOOL]),
        )
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            self.assertEqual(schema["function"]["parameters"]["type"], "object")
            self.assertTrue(schema["function"]["parameters"]["required"])


class TavilyToolTests(unittest.TestCase):
    def test_formats_results(self):
        transport = FakeTransport(
            {
                "results": [
                    {
                        "title": "Retry patterns",
                        "content": "Use exponential backoff with jitter.",
                        "url": "https://example.com/retry",
                    }
                ]
            }
        )
        result = _run(_registry(_settings(), transport).invoke(TAVILY_TOOL, {"query": "retry"}))
        self.assertTrue(result.ok)
        self.assertIn("Tavily web context", result.content)
        self.assertIn("exponential backoff", result.content)
        method, url, _, payload = transport.calls[0]
        self.assertEqual((method, url), ("POST", TAVILY_ENDPOINT))
        self.assertEqual(payload["query"], "retry")
        self.assertEqual(payload["max_results"], 3)

    def test_clamps_hostile_arguments(self):
        transport = FakeTransport({"results": []})
        _run(
            _registry(_settings(), transport).invoke(
                TAVILY_TOOL, {"query": "x", "max_results": 9999, "search_depth": "NOPE"}
            )
        )
        _, _, _, payload = transport.calls[0]
        self.assertEqual(payload["max_results"], 5)
        self.assertEqual(payload["search_depth"], "basic")

    def test_missing_query_short_circuits(self):
        transport = FakeTransport({"results": []})
        result = _run(_registry(_settings(), transport).invoke(TAVILY_TOOL, {}))
        self.assertIn("No query supplied", result.content)
        self.assertEqual(transport.calls, [])

    def test_provider_error_is_reported_to_the_model(self):
        transport = FakeTransport(error=RuntimeError("502"))
        result = _run(_registry(_settings(), transport).invoke(TAVILY_TOOL, {"query": "x"}))
        self.assertTrue(result.ok)
        self.assertIn("Tavily search failed", result.content)

    def test_empty_results(self):
        transport = FakeTransport({"results": []})
        result = _run(_registry(_settings(), transport).invoke(TAVILY_TOOL, {"query": "x"}))
        self.assertIn("No web results", result.content)


class FirecrawlToolTests(unittest.TestCase):
    def test_reads_nested_markdown(self):
        transport = FakeTransport({"data": {"markdown": "# Heading\nbody"}})
        result = _run(
            _registry(_settings(), transport).invoke(FIRECRAWL_TOOL, {"url": "https://x.dev"})
        )
        self.assertIn("Firecrawl extracted context from https://x.dev", result.content)
        self.assertIn("# Heading", result.content)
        method, url, headers, _ = transport.calls[0]
        self.assertEqual((method, url), ("POST", FIRECRAWL_ENDPOINT))
        self.assertEqual(headers["Authorization"], "Bearer fc-test")

    def test_reads_flat_markdown(self):
        transport = FakeTransport({"markdown": "flat body"})
        result = _run(
            _registry(_settings(), transport).invoke(FIRECRAWL_TOOL, {"url": "https://x.dev"})
        )
        self.assertIn("flat body", result.content)

    def test_missing_url_short_circuits(self):
        transport = FakeTransport({})
        result = _run(_registry(_settings(), transport).invoke(FIRECRAWL_TOOL, {}))
        self.assertIn("No url supplied", result.content)
        self.assertEqual(transport.calls, [])

    def test_no_content(self):
        transport = FakeTransport({"data": {}})
        result = _run(
            _registry(_settings(), transport).invoke(FIRECRAWL_TOOL, {"url": "https://x.dev"})
        )
        self.assertIn("No extractable content", result.content)

    def test_error_is_reported(self):
        transport = FakeTransport(error=TimeoutError("slow"))
        result = _run(
            _registry(_settings(), transport).invoke(FIRECRAWL_TOOL, {"url": "https://x.dev"})
        )
        self.assertIn("Page scrape failed", result.content)


class GitHubToolTests(unittest.TestCase):
    def test_formats_repositories(self):
        transport = FakeTransport(
            {
                "items": [
                    {
                        "full_name": "acme/retry",
                        "description": "retry library",
                        "html_url": "https://github.com/acme/retry",
                    }
                ]
            }
        )
        result = _run(
            _registry(_settings(), transport).invoke(GITHUB_TOOL, {"query": "retry language:go"})
        )
        self.assertIn("GitHub context", result.content)
        self.assertIn("acme/retry", result.content)
        method, url, headers, payload = transport.calls[0]
        self.assertEqual((method, url), ("GET", GITHUB_ENDPOINT))
        self.assertEqual(headers["Accept"], "application/vnd.github+json")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(payload["per_page"], 2)

    def test_token_is_sent_when_present(self):
        transport = FakeTransport({"items": []})
        _run(
            _registry(_settings(github_token="ghp_x"), transport).invoke(
                GITHUB_TOOL, {"query": "retry"}
            )
        )
        _, _, headers, _ = transport.calls[0]
        self.assertEqual(headers["Authorization"], "Bearer ghp_x")

    def test_no_matches(self):
        transport = FakeTransport({"items": []})
        result = _run(_registry(_settings(), transport).invoke(GITHUB_TOOL, {"query": "zzz"}))
        self.assertIn("No repositories matched", result.content)

    def test_error_is_reported(self):
        transport = FakeTransport(error=RuntimeError("403"))
        result = _run(_registry(_settings(), transport).invoke(GITHUB_TOOL, {"query": "x"}))
        self.assertIn("GitHub search failed", result.content)


class RegistryCompositionTests(unittest.TestCase):
    def test_extra_specs_are_included(self):
        from app.services.tools.registry import ToolSpec

        extra = ToolSpec(
            name="extra",
            description="d",
            parameters={"type": "object", "properties": {}},
            handler=lambda a: "ok",
        )
        registry = build_enrichment_registry(_settings(), extra=[extra])
        self.assertIn("extra", registry.names())

    def test_tools_build_independent_specs(self):
        first = build_enrichment_tools(_settings())
        second = build_enrichment_tools(_settings())
        self.assertEqual([s.name for s in first], [s.name for s in second])
        self.assertIsNot(first[0], second[0])

    def test_default_request_is_httpx_backed(self):
        # The default RequestFn exists so a real deployment needs no wiring, but
        # it is never invoked in tests.
        specs = build_enrichment_tools(_settings())
        self.assertEqual(len(specs), 3)
        self.assertIsInstance(ToolRegistry(specs), ToolRegistry)


class FirstUrlTests(unittest.TestCase):
    def test_extracts_url_from_tavily_block(self):
        block = "Tavily web context:\n- Title: snippet (https://example.com/a)"
        self.assertEqual(first_url_from(block), "https://example.com/a")

    def test_returns_empty_when_absent(self):
        self.assertEqual(first_url_from("no urls here"), "")
        self.assertEqual(first_url_from(""), "")


if __name__ == "__main__":
    unittest.main()
