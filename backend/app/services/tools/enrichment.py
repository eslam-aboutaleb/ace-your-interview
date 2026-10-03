"""External enrichment providers exposed as tools.

Tavily (web search), Firecrawl (page scrape) and GitHub (repository search) were
three private methods on the old gateway. They are now :class:`ToolSpec` entries
in the registry, so the model decides which of them to call, when, and with what
arguments — instead of the server firing all three on every query.

Every HTTP call goes through an injectable ``request`` callable, so the tools are
fully exercised in tests with no network access and no monkeypatching.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.services.tools.registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)

TAVILY_TOOL = "tavily_search"
FIRECRAWL_TOOL = "scrape_page"
GITHUB_TOOL = "github_search_repos"

TAVILY_ENDPOINT = "https://api.tavily.com/search"
FIRECRAWL_ENDPOINT = "https://api.firecrawl.dev/v1/scrape"
GITHUB_ENDPOINT = "https://api.github.com/search/repositories"

#: (method, url, headers, payload) -> decoded JSON object.
RequestFn = Callable[[str, str, dict, dict], Awaitable[dict]]

#: Prefixes a handler uses to report that the provider call did not happen.
#: In the model tool loop that text is useful — it tells the model the tool is
#: not working. In the deterministic single-pass mode it is *not* context, so
#: :mod:`app.services.mcp_gateway` filters these out before injecting anything
#: into a user prompt.
PROVIDER_FAILURE_PREFIXES: tuple[str, ...] = (
    "Tavily search failed:",
    "Page scrape failed:",
    "GitHub search failed:",
)


async def httpx_request(
    method: str,
    url: str,
    headers: dict,
    payload: dict,
    *,
    timeout: float = 8.0,
) -> dict:
    """Default :data:`RequestFn`: one httpx call, no retries."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        if str(method).upper() == "GET":
            res = await client.get(url, headers=headers, params=payload)
        else:
            res = await client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()
    return data if isinstance(data, dict) else {}


def _clip(text: str, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def _as_int(value: Any, default: int, *, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def build_enrichment_tools(
    settings: Any,
    *,
    request: RequestFn | None = None,
) -> list[ToolSpec]:
    """Build the enrichment tool specs for one settings object.

    ``enabled`` is derived from the operator flags *and* from credential
    presence, so a tool with no API key is never advertised to the model: the
    model cannot waste a turn discovering a tool is unusable.
    """
    fetch = request or (lambda m, u, h, p: httpx_request(m, u, h, p, timeout=float(getattr(settings, "mcp_timeout_seconds", 8.0))))
    timeout = float(getattr(settings, "mcp_timeout_seconds", 8.0) or 8.0)

    tavily_ready = bool(
        getattr(settings, "mcp_tavily_enabled", False)
        and str(getattr(settings, "tavily_api_key", "") or "").strip()
    )
    firecrawl_ready = bool(
        getattr(settings, "mcp_firecrawl_enabled", False)
        and str(getattr(settings, "firecrawl_api_key", "") or "").strip()
    )
    github_ready = bool(getattr(settings, "mcp_github_enabled", False))

    async def tavily_search(arguments: dict[str, Any]) -> str:
        query = _clip(arguments.get("query"), 300)
        if not query:
            return "No query supplied."
        depth = str(arguments.get("search_depth") or "basic").strip().lower()
        if depth not in ("basic", "advanced"):
            depth = "basic"
        max_results = _as_int(arguments.get("max_results"), 3, low=1, high=5)
        try:
            data = await fetch(
                "POST",
                TAVILY_ENDPOINT,
                {},
                {
                    "api_key": str(getattr(settings, "tavily_api_key", "")),
                    "query": query,
                    "search_depth": depth,
                    "max_results": max_results,
                },
            )
        except Exception as exc:  # noqa: BLE001 - reported to the model, not raised
            logger.info("tavily_search_failed query=%s err=%s", query[:120], exc)
            return f"Tavily search failed: {type(exc).__name__}."
        rows: list[str] = []
        for item in (data.get("results") or [])[:max_results]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            snippet = _clip(item.get("content"), 320)
            url = str(item.get("url", "")).strip()
            if title or snippet:
                rows.append(f"- {title}: {snippet} ({url})".strip())
        if not rows:
            return "No web results for that query."
        return "Tavily web context:\n" + "\n".join(rows)

    async def scrape_page(arguments: dict[str, Any]) -> str:
        url = str(arguments.get("url") or "").strip()[:2000]
        if not url:
            return "No url supplied."
        try:
            data = await fetch(
                "POST",
                FIRECRAWL_ENDPOINT,
                {"Authorization": f"Bearer {getattr(settings, 'firecrawl_api_key', '')}"},
                {"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("scrape_page_failed url=%s err=%s", url[:200], exc)
            return f"Page scrape failed: {type(exc).__name__}."
        markdown = ""
        if isinstance(data.get("data"), dict):
            markdown = str(data["data"].get("markdown", "")).strip()
        if not markdown:
            markdown = str(data.get("markdown", "")).strip()
        if not markdown:
            return f"No extractable content at {url}."
        return f"Firecrawl extracted context from {url}:\n{markdown[:1800]}"

    async def github_search_repos(arguments: dict[str, Any]) -> str:
        query = _clip(arguments.get("query"), 260)
        if not query:
            return "No query supplied."
        per_page = _as_int(arguments.get("per_page"), 2, low=1, high=5)
        headers = {"Accept": "application/vnd.github+json"}
        token = str(getattr(settings, "github_token", "") or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            data = await fetch(
                "GET",
                GITHUB_ENDPOINT,
                headers,
                {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": per_page,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("github_search_failed query=%s err=%s", query[:160], exc)
            return f"GitHub search failed: {type(exc).__name__}."
        rows: list[str] = []
        for item in (data.get("items") or [])[:per_page]:
            if not isinstance(item, dict):
                continue
            full_name = str(item.get("full_name", "")).strip()
            if not full_name:
                continue
            description = str(item.get("description", "")).strip()[:220]
            rows.append(
                f"- {full_name}: {description} ({str(item.get('html_url', '')).strip()})"
            )
        if not rows:
            return "No repositories matched that query."
        return "GitHub context:\n" + "\n".join(rows)

    return [
        ToolSpec(
            name=TAVILY_TOOL,
            description=(
                "Search the public web and return short result snippets. "
                "Use for current facts, terminology, or practice material."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "basic is cheaper; advanced is more thorough.",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "How many results to return.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=tavily_search,
            enabled=tavily_ready,
            timeout_seconds=timeout,
            metadata={"source": "tavily"},
        ),
        ToolSpec(
            name=FIRECRAWL_TOOL,
            description=(
                "Fetch one URL and return its main content as markdown. "
                "Use after a search returns a promising link."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL."}
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            handler=scrape_page,
            enabled=firecrawl_ready,
            timeout_seconds=timeout,
            metadata={"source": "firecrawl"},
        ),
        ToolSpec(
            name=GITHUB_TOOL,
            description=(
                "Search public GitHub repositories by keyword. "
                "Use to find real implementations, READMEs, or prior art."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "GitHub search query."},
                    "per_page": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "How many repositories to return.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=github_search_repos,
            enabled=github_ready,
            timeout_seconds=timeout,
            metadata={"source": "github"},
        ),
    ]


def build_enrichment_registry(
    settings: Any,
    *,
    request: RequestFn | None = None,
    extra: list[ToolSpec] | None = None,
) -> ToolRegistry:
    """A registry containing the enrichment tools (plus any extras)."""
    registry = ToolRegistry(build_enrichment_tools(settings, request=request))
    if extra:
        registry.register_all(extra, replace_existing=True)
    return registry


def first_url_from(text: str) -> str:
    """Extract the first ``(url)`` from a Tavily-formatted context block."""
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        if line.endswith(")") and "(" in line:
            candidate = line.rsplit("(", 1)[-1].rstrip(")").strip()
            if candidate.startswith("http"):
                return candidate
    return ""
