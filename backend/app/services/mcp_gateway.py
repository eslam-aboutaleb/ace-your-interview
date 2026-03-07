"""MCP-like context gateway for optional external enrichment."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_FLOW_STAGE = {
    "custom_topic": 1,
    "chat": 1,
    "questions": 2,
    "quiz": 2,
    "interview": 3,
}


class MCPGateway:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def _flow_flag_enabled(self, flow: str) -> bool:
        if flow == "custom_topic":
            return bool(self.settings.mcp_enable_custom_topic)
        if flow == "chat":
            return bool(self.settings.mcp_enable_chat)
        if flow == "questions":
            return bool(self.settings.mcp_enable_questions)
        if flow == "quiz":
            return bool(self.settings.mcp_enable_quiz)
        if flow == "interview":
            return bool(self.settings.mcp_enable_interview)
        return False

    def is_enabled_for_flow(self, flow: str) -> bool:
        stage_required = _FLOW_STAGE.get(flow, 99)
        if not self.settings.enable_mcp_gateway:
            return False
        if not self._flow_flag_enabled(flow):
            return False
        return int(self.settings.mcp_rollout_stage) >= stage_required

    async def _tavily_context(self, query: str) -> tuple[str, str]:
        if not (self.settings.mcp_tavily_enabled and self.settings.tavily_api_key.strip()):
            return "", ""
        endpoint = "https://api.tavily.com/search"
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 3,
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.mcp_timeout_seconds) as client:
                res = await client.post(endpoint, json=payload)
                res.raise_for_status()
                data = res.json() if isinstance(res.json(), dict) else {}
        except Exception as exc:
            logger.info("mcp_tavily_failed flow_query=%s err=%s", query[:120], exc)
            return "", ""

        rows: list[str] = []
        first_url = ""
        for item in data.get("results", [])[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()
            url = str(item.get("url", "")).strip()
            if not first_url and url:
                first_url = url
            snippet = " ".join(content.split())[:320]
            if title or snippet:
                rows.append(f"- {title}: {snippet} ({url})".strip())
        if not rows:
            return "", first_url
        return "Tavily web context:\n" + "\n".join(rows), first_url

    async def _firecrawl_context(self, url: str) -> str:
        if not url:
            return ""
        if not (self.settings.mcp_firecrawl_enabled and self.settings.firecrawl_api_key.strip()):
            return ""
        endpoint = "https://api.firecrawl.dev/v1/scrape"
        headers = {"Authorization": f"Bearer {self.settings.firecrawl_api_key}"}
        payload = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.mcp_timeout_seconds) as client:
                res = await client.post(endpoint, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json() if isinstance(res.json(), dict) else {}
        except Exception as exc:
            logger.info("mcp_firecrawl_failed url=%s err=%s", url[:200], exc)
            return ""

        markdown = ""
        if isinstance(data.get("data"), dict):
            markdown = str(data["data"].get("markdown", "")).strip()
        if not markdown:
            markdown = str(data.get("markdown", "")).strip()
        if not markdown:
            return ""
        return f"Firecrawl extracted context from {url}:\n{markdown[:1800]}"

    async def _github_context(self, query: str) -> str:
        if not self.settings.mcp_github_enabled:
            return ""
        headers = {"Accept": "application/vnd.github+json"}
        if self.settings.github_token.strip():
            headers["Authorization"] = f"Bearer {self.settings.github_token.strip()}"

        endpoint = "https://api.github.com/search/repositories"
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 2}
        try:
            async with httpx.AsyncClient(timeout=self.settings.mcp_timeout_seconds) as client:
                res = await client.get(endpoint, headers=headers, params=params)
                res.raise_for_status()
                data = res.json() if isinstance(res.json(), dict) else {}
        except Exception as exc:
            logger.info("mcp_github_failed query=%s err=%s", query[:160], exc)
            return ""

        rows: list[str] = []
        for item in data.get("items", [])[:2]:
            if not isinstance(item, dict):
                continue
            full_name = str(item.get("full_name", "")).strip()
            description = str(item.get("description", "")).strip()
            html_url = str(item.get("html_url", "")).strip()
            if full_name:
                rows.append(f"- {full_name}: {description[:220]} ({html_url})")
        if not rows:
            return ""
        return "GitHub context:\n" + "\n".join(rows)

    async def gather_context(
        self,
        *,
        flow: str,
        query: str,
        topic_id: str = "",
        topic_title: str = "",
    ) -> str:
        if not self.is_enabled_for_flow(flow):
            return ""

        normalized_query = " ".join(str(query or "").split())[:300]
        if not normalized_query:
            return ""

        parts: list[str] = []
        try:
            tavily_text, first_url = await self._tavily_context(normalized_query)
            if tavily_text:
                parts.append(tavily_text)

            firecrawl_text = await self._firecrawl_context(first_url)
            if firecrawl_text:
                parts.append(firecrawl_text)

            github_query = " ".join(
                token for token in [topic_title, topic_id, normalized_query] if token
            )[:260]
            github_text = await self._github_context(github_query)
            if github_text:
                parts.append(github_text)
        except Exception as exc:
            logger.exception("mcp_gather_context_unexpected flow=%s err=%s", flow, exc)
            return ""

        if not parts:
            return ""

        merged = "\n\n".join(parts)
        return merged[: int(self.settings.mcp_max_context_chars)]
