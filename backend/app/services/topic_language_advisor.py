"""Infer topic-specific programming language options with LLM + fallback."""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any, Optional

from app.schemas.models import LLMConfigRequest, TopicDetail
from app.services.llm_client import (
    TERMINAL_ERROR_CODES,
    LLMClient,
    parse_json_object,
)
from app.services.llm_policy import raise_if_policy_blocked_result
from app.services.prompt_blocks import (
    UNTRUSTED_CLAUSE,
    render_contract,
    untrusted_block,
)

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_MAX_TOPIC_CONTEXT = 7000
_MAX_TOKENS_CAP = 600
_ADVISOR_SYSTEM_PROMPT = (
    "You are a technical curriculum advisor. You inspect one topic's metadata and section "
    "content and decide whether the topic needs programming-language-specific code examples, "
    "and which languages to offer for them. You always answer with a single JSON object and "
    "never with prose."
)
_CANONICAL_LANGUAGE_MAP = {
    "js": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "ts": "typescript",
    "py": "python",
    "golang": "go",
    "gcp sql": "sql",
    "postgresql": "sql",
    "mysql": "sql",
    "shell": "bash",
    "sh": "bash",
    "c#": "csharp",
    "c++": "cpp",
}


def _normalise_language(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    mapped = _CANONICAL_LANGUAGE_MAP.get(raw, raw)
    mapped = re.sub(r"[^a-z0-9_+\-#]", "", mapped)
    if mapped in _CANONICAL_LANGUAGE_MAP:
        mapped = _CANONICAL_LANGUAGE_MAP[mapped]
    return mapped[:40]


def _normalise_languages(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for item in values:
        lang = _normalise_language(str(item))
        if not lang or lang in out:
            continue
        out.append(lang)
        if len(out) >= 8:
            break
    return out


def _fallback_profile(topic: TopicDetail) -> dict[str, Any]:
    title = (topic.title or "").lower()
    description = (topic.description or "").lower()
    text = f"{topic.id} {title} {description}"

    if topic.track == "frontend":
        return {
            "requires_programming": True,
            "language_options": ["typescript", "javascript", "css", "html"],
            "source": "fallback",
        }
    if topic.track == "backend":
        langs = ["python", "java", "go", "typescript", "sql"]
        if "java" in text:
            langs = ["java", "kotlin", "sql", "bash"]
        elif "python" in text:
            langs = ["python", "sql", "bash", "go"]
        elif "node" in text or "express" in text:
            langs = ["typescript", "javascript", "sql", "bash"]
        return {
            "requires_programming": True,
            "language_options": langs[:8],
            "source": "fallback",
        }
    if topic.track == "ai_stack":
        return {
            "requires_programming": True,
            "language_options": ["python", "typescript", "sql", "bash"],
            "source": "fallback",
        }

    if any(k in text for k in ("algorithm", "coding", "implementation", "api", "database")):
        return {
            "requires_programming": True,
            "language_options": ["python", "java", "go", "sql"],
            "source": "fallback",
        }

    return {
        "requires_programming": False,
        "language_options": [],
        "source": "fallback",
    }


def _supported_options(client: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Drop call options ``client.completion`` does not accept.

    ``LLMClient`` takes the full Stage-2 option set, but the client is injected,
    so a narrower implementation may be in use. Filter against the bound
    signature rather than assuming one shape.
    """
    try:
        params = inspect.signature(client.completion).parameters
    except (TypeError, ValueError):
        return dict(options)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return options
    return {key: value for key, value in options.items() if key in params}



def _build_prompt(topic: TopicDetail, mcp_context: str = "") -> str:
    """Build the user turn. The advisor persona lives in ``system`` (Stage 2.1)."""
    contract = render_contract(
        schema_label="Return ONLY valid JSON",
        schema_block="""{
  "requires_programming": true|false,
  "language_options": ["lowercase canonical language names"]
}""",
        rules=[
            "If requires_programming=false, language_options must be [].",
            "If requires_programming=true, return 3 to 8 language options, ordered by relevance.",
            "Keep options specific to this topic and common in interview prep.",
            "Use canonical lowercase names like: python, javascript, typescript, java, go, csharp, cpp, rust, sql, bash, kotlin, swift, php, ruby.",
            "Do not include frameworks or cloud providers as languages.",
            "Output JSON only.",
            UNTRUSTED_CLAUSE,
        ],
    )
    metadata_block = "\n".join(
        [
            f"Topic ID: {topic.id}",
            f"Track: {topic.track}",
            f"Levels: {', '.join(topic.levels)}",
        ]
    )
    untrusted_blocks = "\n\n".join(
        block
        for block in (
            untrusted_block("topic_title", topic.title),
            untrusted_block("topic_description", topic.description),
            untrusted_block("topic_content", topic.raw_content, _MAX_TOPIC_CONTEXT),
            untrusted_block("external_context", mcp_context, 2500),
        )
        if block
    )
    return f"""{contract}

{metadata_block}

{untrusted_blocks}
"""


class TopicLanguageAdvisor:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def advise_topic(
        self,
        *,
        topic: TopicDetail,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        mcp_context: str = "",
    ) -> dict[str, Any]:
        prompt = _build_prompt(topic, mcp_context=mcp_context)
        for _ in range(_MAX_ATTEMPTS):
            result = await self.llm.completion(
                prompt,
                llm_config,
                user_identity=user_identity,
                **_supported_options(
                    self.llm,
                    {
                        "task": "final",
                        "system": _ADVISOR_SYSTEM_PROMPT,
                        "structured": True,
                        "max_tokens_cap": _MAX_TOKENS_CAP,
                    },
                ),
            )
            raise_if_policy_blocked_result(result)
            if not result.get("success"):
                # A truncated response or an exhausted budget cannot be fixed by
                # re-sending the identical prompt.
                if str(result.get("error_code") or "") in TERMINAL_ERROR_CODES:
                    break
                continue
            payload = parse_json_object(result.get("analysis", ""))
            if not payload:
                continue

            requires_programming = bool(payload.get("requires_programming"))
            options = _normalise_languages(payload.get("language_options"))
            if requires_programming and len(options) < 3:
                continue
            if not requires_programming:
                options = []
            return {
                "requires_programming": requires_programming,
                "language_options": options[:8],
                "source": "llm",
            }

        logger.warning("Falling back to deterministic language profile for topic=%s", topic.id)
        return _fallback_profile(topic)
