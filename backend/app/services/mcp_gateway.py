"""MCP-like context gateway for optional external enrichment."""

from __future__ import annotations

import logging
import re

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

    @staticmethod
    def _query_terms(text: str) -> list[str]:
        stop = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "what",
            "when",
            "where",
            "which",
            "into",
            "about",
            "your",
        }
        out: list[str] = []
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower()):
            if len(token) < 4 or token in stop:
                continue
            if token not in out:
                out.append(token)
            if len(out) >= 8:
                break
        return out

    def _rewrite_query_candidates(
        self,
        *,
        query: str,
        topic_id: str,
        topic_title: str,
        max_steps: int,
    ) -> list[str]:
        base = " ".join(str(query or "").split())[:300]
        if not base:
            return []
        candidates: list[str] = [base]
        tokens = self._query_terms(base)
        topic_tokens = self._query_terms(f"{topic_title} {topic_id}")
        if tokens:
            candidates.append(f"{' '.join(tokens[:4])} interview best practices")
        if topic_tokens:
            candidates.append(f"{' '.join(topic_tokens[:4])} implementation pitfalls tradeoffs")
        out: list[str] = []
        for cand in candidates:
            normalized = " ".join(cand.split())[:300]
            if normalized and normalized not in out:
                out.append(normalized)
            if len(out) >= max_steps:
                break
        return out

    def _context_is_adequate(self, *, query: str, merged: str, step: int, max_steps: int) -> bool:
        if step >= max_steps:
            return True
        if len(merged) >= 1400:
            return True
        query_terms = self._query_terms(query)
        merged_l = merged.lower()
        overlap = sum(1 for term in query_terms if term in merged_l)
        return overlap >= 3 and len(merged) >= 600

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
        max_steps = max(1, int(getattr(self.settings, "mcp_agentic_max_steps", 2)))
        if not getattr(self.settings, "mcp_agentic_loop_enabled", False):
            max_steps = 1
        queries = self._rewrite_query_candidates(
            query=normalized_query,
            topic_id=topic_id,
            topic_title=topic_title,
            max_steps=max_steps,
        )
        if not queries:
            return ""
        try:
            for step, query_variant in enumerate(queries, start=1):
                tavily_text, first_url = await self._tavily_context(query_variant)
                if tavily_text:
                    parts.append(tavily_text)

                firecrawl_text = await self._firecrawl_context(first_url)
                if firecrawl_text:
                    parts.append(firecrawl_text)

                github_query = " ".join(
                    token for token in [topic_title, topic_id, query_variant] if token
                )[:260]
                github_text = await self._github_context(github_query)
                if github_text:
                    parts.append(github_text)

                merged_preview = "\n\n".join(parts)
                if self._context_is_adequate(
                    query=normalized_query,
                    merged=merged_preview,
                    step=step,
                    max_steps=max_steps,
                ):
                    break
        except Exception as exc:
            logger.exception("mcp_gather_context_unexpected flow=%s err=%s", flow, exc)
            return ""

        if not parts:
            return ""

        merged = "\n\n".join(parts)
        return merged[: int(self.settings.mcp_max_context_chars)]
