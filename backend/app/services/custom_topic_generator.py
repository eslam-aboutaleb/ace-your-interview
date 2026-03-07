"""Generate user-defined custom topic roadmaps via LLM with validation/fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from app.schemas.models import LLMConfigRequest, TopicDetail
from app.services.llm_client import LLMClient
from app.services.llm_policy import raise_if_policy_blocked_result
from app.services.mcp_gateway import MCPGateway

logger = logging.getLogger(__name__)

_VALID_TRACKS = {"backend", "frontend", "system_design", "ai_stack"}
_VALID_LEVELS = {"junior", "mid", "senior"}
_DEFAULT_LEVELS = ["junior", "mid", "senior"]
_MIN_SECTIONS = 100
_MAX_SECTIONS = 150
_DEFAULT_SECTIONS = 120
_MAX_ATTEMPTS = 4
_GENERIC_HEADING_PATTERNS = [
    re.compile(r"^(module|part|chapter|section|topic)\s*[\w\-\.]*$", re.IGNORECASE),
    re.compile(r"^(introduction|overview|basics?|advanced|intermediate|conclusion)$", re.IGNORECASE),
]
_REQUIRED_COVERAGE_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "fundamentals": ("fundamental", "basics", "terminology", "mental model", "core concepts"),
    "workflow": ("setup", "workflow", "tooling", "delivery", "ci/cd", "build"),
    "implementation": ("implementation", "integration", "api", "data flow", "state management"),
    "debugging": ("debug", "troubleshoot", "diagnostic", "root cause", "pitfall"),
    "testing": ("test", "verification", "qa", "unit", "integration test"),
    "performance": ("performance", "latency", "throughput", "optimization", "profiling"),
    "security": ("security", "auth", "hardening", "threat", "compliance"),
    "reliability": ("reliability", "resilience", "availability", "fault", "failure mode"),
    "architecture": ("architecture", "system design", "design pattern", "scalability", "decomposition"),
    "operations": ("operations", "observability", "monitoring", "incident", "runbook"),
}
_DIMENSION_LEVEL = {
    "fundamentals": "junior",
    "workflow": "junior",
    "implementation": "mid",
    "debugging": "mid",
    "testing": "mid",
    "performance": "mid",
    "security": "senior",
    "reliability": "senior",
    "architecture": "senior",
    "operations": "senior",
}
_DIMENSION_LABELS = {
    "fundamentals": "Foundations and Core Terminology",
    "workflow": "Workflow, Tooling, and Delivery",
    "implementation": "Implementation Patterns and Integration",
    "debugging": "Debugging and Root Cause Analysis",
    "testing": "Testing and Quality Verification",
    "performance": "Performance and Scalability",
    "security": "Security and Threat Mitigation",
    "reliability": "Reliability and Failure Recovery",
    "architecture": "Architecture Tradeoffs and System Design",
    "operations": "Operations, Observability, and Runbooks",
}


def _slugify_topic(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug or "topic"


def _clamp_target_sections(value: int) -> int:
    if value < _MIN_SECTIONS:
        return _MIN_SECTIONS
    if value > _MAX_SECTIONS:
        return _MAX_SECTIONS
    return value


def _normalise_track(track: str) -> str:
    t = (track or "").strip().lower().replace("-", "_").replace(" ", "_")
    if t in _VALID_TRACKS:
        return t
    return ""


def _infer_track(topic: str) -> str:
    text = topic.lower()
    if any(k in text for k in ("system design", "distributed", "scaling", "reliability")):
        return "system_design"
    if any(k in text for k in ("frontend", "react", "vue", "css", "javascript", "web")):
        return "frontend"
    if any(k in text for k in ("ai", "llm", "prompt", "rag", "agent", "ml", "machine learning")):
        return "ai_stack"
    return "backend"


def _normalise_levels(levels: Any) -> list[str]:
    if isinstance(levels, list):
        raw = [str(x).strip().lower() for x in levels]
    elif isinstance(levels, str):
        raw = [x.strip().lower() for x in levels.split(",")]
    else:
        raw = []
    out: list[str] = []
    for item in raw:
        if item in _VALID_LEVELS and item not in out:
            out.append(item)
    return out or list(_DEFAULT_LEVELS)


def _level_rank(level: str) -> int:
    if level == "junior":
        return 0
    if level == "mid":
        return 1
    return 2


def _level_for_position(index: int, total: int) -> str:
    if total <= 0:
        return "junior"
    junior_cutoff = max(1, total // 3)
    mid_cutoff = max(junior_cutoff + 1, (2 * total) // 3)
    if index < junior_cutoff:
        return "junior"
    if index < mid_cutoff:
        return "mid"
    return "senior"


def _format_heading(level: str, heading: str) -> str:
    stripped = heading.strip()
    if not stripped:
        return ""
    prefix = f"{level.title()}:"
    if stripped.lower().startswith(("junior:", "mid:", "senior:")):
        return stripped[:200]
    return f"{prefix} {stripped}"[:200]


def _heading_is_generic(heading: str) -> bool:
    text = heading.strip()
    if not text:
        return True
    lowered = text.lower()
    if len(lowered) < 8:
        return True
    if len(lowered.split()) < 2:
        return True
    for pattern in _GENERIC_HEADING_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _autofill_section(topic: str, index: int, total: int) -> dict[str, str]:
    level = _level_for_position(index, total)
    junior_cutoff = max(1, total // 3)
    mid_cutoff = max(junior_cutoff + 1, (2 * total) // 3)
    if level == "junior":
        level_offset = index
    elif level == "mid":
        level_offset = max(0, index - junior_cutoff)
    else:
        level_offset = max(0, index - mid_cutoff)
    track = _infer_track(topic)
    level_templates_by_track = {
        "backend": {
            "junior": [
                "Language Syntax, Types, and Core APIs",
                "Toolchain Setup, Build, and Dependency Management",
                "Control Flow, Error Handling, and Debugging",
                "Object Modeling and Data Structures",
                "Unit Testing and Test Doubles",
                "HTTP Basics and REST Foundations",
                "Persistence Fundamentals with SQL and NoSQL",
                "Code Style, Refactoring, and Readability",
                "Version Control and Collaborative Workflows",
                "Common Beginner Interview Scenarios",
            ],
            "mid": [
                "Concurrency Patterns and Thread Safety",
                "API Design, Validation, and Contract Evolution",
                "Transactions, Isolation, and Data Consistency",
                "Caching Strategies and Invalidation Tradeoffs",
                "Observability with Logs, Metrics, and Traces",
                "Security Hardening and Secret Management",
                "Performance Profiling and Bottleneck Reduction",
                "Integration Testing and Failure Injection",
                "CI/CD Pipelines and Safe Release Tactics",
                "Service Boundaries and Domain-Driven Design",
            ],
            "senior": [
                "Distributed Architecture and Service Decomposition",
                "Scalability Strategy and Capacity Planning",
                "Reliability Engineering, SLOs, and Incident Response",
                "Data Consistency Models and Recovery Design",
                "Multi-Region Architecture and Disaster Readiness",
                "Cost-Performance Optimization at Scale",
                "Security Governance and Compliance Tradeoffs",
                "Legacy Migration and Modernization Strategy",
                "Architecture Review, Decision Records, and Alignment",
                "Technical Leadership, Mentoring, and Hiring Signals",
            ],
        },
        "frontend": {
            "junior": [
                "HTML Semantics, Accessibility Basics, and ARIA",
                "CSS Layout Systems and Responsive Design",
                "JavaScript/TypeScript Fundamentals and Debugging",
                "Component Composition and Reusability",
                "State Basics and Event-Driven UI Patterns",
                "API Fetching and Error States in UI",
                "Form Handling, Validation, and UX Feedback",
                "Testing Basics with Unit and Component Tests",
                "Performance Basics: Rendering and Bundle Size",
                "Common Beginner Frontend Interview Scenarios",
            ],
            "mid": [
                "Advanced State Management and Data Flow",
                "Routing Architecture and Navigation Patterns",
                "Design Systems, Tokens, and UI Consistency",
                "Frontend Security: XSS, CSRF, and Token Storage",
                "Caching, Pagination, and Optimistic Updates",
                "Observability for Frontend Errors and UX Metrics",
                "Performance Optimization with Profiling",
                "Integration and End-to-End Testing Strategy",
                "Microfrontend and Module Boundary Tradeoffs",
                "Accessibility Audits and Remediation Workflow",
            ],
            "senior": [
                "Frontend Platform Architecture and Governance",
                "SSR, SSG, and Edge Rendering Tradeoffs",
                "Scalable Design System Operations",
                "Performance Budgets and Web Vitals Program",
                "Cross-Team API Contract Strategy",
                "Resilience Patterns for Degraded Dependencies",
                "Migration Strategy Across Framework Versions",
                "Security and Privacy by Design in UI Platforms",
                "Architecture Decision-Making Under Constraints",
                "Senior Frontend Leadership and Mentoring Patterns",
            ],
        },
        "system_design": {
            "junior": [
                "Core System Design Terminology and Mental Models",
                "Latency, Throughput, and Basic Capacity Math",
                "Load Balancing and Stateless Service Basics",
                "Database Selection Fundamentals",
                "Caching Fundamentals and Common Patterns",
                "Message Queues and Async Processing Basics",
                "Consistency, Availability, and Partition Tolerance",
                "Failure Handling and Retry Basics",
                "Monitoring Basics and Alert Fundamentals",
                "Beginner System Design Interview Walkthroughs",
            ],
            "mid": [
                "Data Partitioning and Sharding Tradeoffs",
                "Replication Strategies and Read/Write Patterns",
                "Idempotency, Deduplication, and Exactly-Once Myths",
                "Distributed Transactions and Saga Patterns",
                "API Gateway Patterns and Traffic Management",
                "Hotspot Mitigation and Backpressure Strategy",
                "Observability Architecture for Distributed Systems",
                "Security Boundaries and Zero-Trust Considerations",
                "Storage Tiering and Cost-Aware Design",
                "Mid-Level Design Reviews with Tradeoff Analysis",
            ],
            "senior": [
                "Multi-Region Topology and Data Sovereignty",
                "Disaster Recovery and Business Continuity Planning",
                "SLO Design, Error Budgets, and Reliability Culture",
                "Evolutionary Architecture and Migration Planning",
                "High-Scale Event-Driven Architecture Tradeoffs",
                "Operational Readiness and Incident Command",
                "Cost Governance and FinOps for Distributed Systems",
                "Security and Compliance Architecture at Scale",
                "Executive Communication for Architecture Decisions",
                "Senior System Design Interview Deep Dives",
            ],
        },
        "ai_stack": {
            "junior": [
                "LLM Fundamentals, Tokens, and Context Windows",
                "Prompt Design Basics and Evaluation Criteria",
                "Embeddings and Vector Search Fundamentals",
                "RAG Pipeline Basics and Chunking Strategies",
                "Model Selection Basics and Cost Awareness",
                "Tool Calling Fundamentals and API Contracts",
                "Guardrails and Basic Safety Practices",
                "Offline Evaluation Basics and Test Sets",
                "Serving Basics: Latency, Throughput, and Caching",
                "Beginner AI Interview Scenarios and Reasoning",
            ],
            "mid": [
                "Advanced Prompting and Structured Outputs",
                "RAG Retrieval Quality and Re-Ranking Tradeoffs",
                "Agent Workflow Design and Tool Reliability",
                "Hallucination Mitigation and Citation Strategies",
                "Model Routing and Dynamic Fallbacks",
                "Observability for LLM Apps and Failure Analysis",
                "Human-in-the-Loop and Feedback Loops",
                "Cost Optimization Across Models and Pipelines",
                "Security for AI Workloads and Data Handling",
                "Mid-Level AI System Case Studies",
            ],
            "senior": [
                "AI Platform Architecture and Multi-Tenant Design",
                "Evaluation Governance and Model Lifecycle Strategy",
                "Risk Management, Policy Controls, and Compliance",
                "Advanced Agent Orchestration and Reliability",
                "Production Incident Response for AI Systems",
                "Data Flywheel Strategy and Continuous Improvement",
                "Scalable Inference Infrastructure Tradeoffs",
                "Vendor Strategy and Portability Decisions",
                "AI Product Strategy and Technical Leadership",
                "Senior AI Architecture Interviews and Tradeoffs",
            ],
        },
    }
    templates = level_templates_by_track.get(track, level_templates_by_track["backend"])[level]
    template = templates[level_offset % len(templates)]
    depth_lenses = [
        "Concepts",
        "Implementation",
        "Hands-on Lab",
        "Debugging",
        "Testing",
        "Performance",
        "Security",
        "Reliability",
        "Design Tradeoffs",
        "Interview Deep Dive",
        "Production Case Study",
        "Common Pitfalls",
    ]
    cycle = level_offset // len(templates)
    lens = depth_lenses[cycle % len(depth_lenses)]
    heading = _format_heading(level, f"{topic}: {template} ({lens})")
    content = (
        f"Teach {template.lower()} in the context of {topic}, including hands-on implementation details, "
        "typical mistakes, and interview-ready tradeoff reasoning for this level."
    )
    return {"heading": heading, "content": content[:1800]}


def _level_bounds(level: str, total: int) -> tuple[int, int]:
    junior_cutoff = max(1, total // 3)
    mid_cutoff = max(junior_cutoff + 1, (2 * total) // 3)
    if level == "junior":
        return 0, junior_cutoff
    if level == "mid":
        return junior_cutoff, mid_cutoff
    return mid_cutoff, total


def _covered_dimensions(sections: list[dict[str, str]]) -> set[str]:
    covered: set[str] = set()
    for sec in sections:
        corpus = f"{sec.get('heading', '')} {sec.get('content', '')}".lower()
        for dimension, keywords in _REQUIRED_COVERAGE_DIMENSIONS.items():
            if any(keyword in corpus for keyword in keywords):
                covered.add(dimension)
    return covered


def _build_dimension_section(topic: str, *, level: str, dimension: str) -> dict[str, str]:
    label = _DIMENSION_LABELS.get(dimension, dimension.replace("_", " ").title())
    heading = _format_heading(level, f"{topic}: {label}")
    content = (
        f"Cover {label.lower()} for {topic} with practical implementation details, "
        "common failure patterns, and interview-ready tradeoff reasoning. Include how priorities "
        "change across project scale, production constraints, and team ownership boundaries."
    )
    return {"heading": heading, "content": content[:1800]}


def _enforce_coverage_dimensions(
    sections: list[dict[str, str]], topic: str, target_sections: int
) -> list[dict[str, str]]:
    if not sections:
        return sections

    enforced = sections[:target_sections]
    missing = [
        dimension
        for dimension in _REQUIRED_COVERAGE_DIMENSIONS
        if dimension not in _covered_dimensions(enforced)
    ]
    if not missing:
        return enforced

    seen_headings = {str(sec.get("heading", "")).strip().lower() for sec in enforced}
    used_indexes: set[int] = set()
    for dimension in missing:
        level = _DIMENSION_LEVEL.get(dimension, "mid")
        start, end = _level_bounds(level, len(enforced))
        target_index: Optional[int] = None

        for idx in range(end - 1, start - 1, -1):
            if idx not in used_indexes:
                target_index = idx
                break
        if target_index is None:
            for idx in range(len(enforced) - 1, -1, -1):
                if idx not in used_indexes:
                    target_index = idx
                    break
        if target_index is None:
            break

        replacement = _build_dimension_section(topic, level=level, dimension=dimension)
        replacement_key = replacement["heading"].strip().lower()
        suffix = 2
        while replacement_key in seen_headings:
            replacement["heading"] = _format_heading(
                level,
                f"{topic}: {_DIMENSION_LABELS.get(dimension, dimension)} {suffix}",
            )
            replacement_key = replacement["heading"].strip().lower()
            suffix += 1

        current_key = str(enforced[target_index].get("heading", "")).strip().lower()
        seen_headings.discard(current_key)
        seen_headings.add(replacement_key)
        enforced[target_index] = replacement
        used_indexes.add(target_index)

    return enforced


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _normalise_sections(sections: Any, topic: str, target_sections: int) -> list[dict[str, str]]:
    ranked_sections: list[tuple[int, int, dict[str, str]]] = []
    seen: set[str] = set()
    if isinstance(sections, list):
        for idx, item in enumerate(sections):
            if not isinstance(item, dict):
                continue
            heading = str(item.get("heading", "")).strip()
            content = str(item.get("content", "")).strip()
            if len(heading) < 3 or len(content) < 10:
                continue
            if _heading_is_generic(heading):
                continue
            level_raw = str(item.get("level", "")).strip().lower()
            level = level_raw if level_raw in _VALID_LEVELS else _level_for_position(idx, target_sections)
            formatted_heading = _format_heading(level, heading)
            heading_key = formatted_heading.lower()
            if heading_key in seen:
                continue
            seen.add(heading_key)
            ranked_sections.append(
                (
                    _level_rank(level),
                    idx,
                    {
                        "heading": formatted_heading,
                        "content": content[:1800],
                    },
                )
            )
            if len(ranked_sections) >= target_sections:
                break

    ranked_sections.sort(key=lambda item: (item[0], item[1]))
    normalised = [item[2] for item in ranked_sections]

    if len(normalised) < target_sections:
        for idx in range(len(normalised), target_sections):
            fill = _autofill_section(topic, idx, target_sections)
            heading_key = fill["heading"].lower()
            if heading_key in seen:
                continue
            seen.add(heading_key)
            normalised.append(fill)
            if len(normalised) >= target_sections:
                break

    normalised = _enforce_coverage_dimensions(normalised[:target_sections], topic, target_sections)
    return normalised[:target_sections]


def _build_raw_content(title: str, description: str, sections: list[dict[str, str]]) -> str:
    parts = [f"# {title}", "", description.strip()]
    for sec in sections:
        parts.extend(["", f"## {sec['heading']}", "", sec["content"]])
    return "\n".join(parts).strip()


def _build_prompt(topic: str, target_sections: int, mcp_context: str = "") -> str:
    junior_count = target_sections // 3
    mid_count = target_sections // 3
    senior_count = target_sections - junior_count - mid_count
    mcp_block = (
        f"\n\nExternal context (optional, use only if relevant and factual):\n{mcp_context[:2500]}"
        if mcp_context
        else ""
    )
    return f"""You are an expert curriculum architect and technical interview coach.
Design this as if you are preparing someone to become a strong practitioner and teacher in the topic.

Create a deep learning roadmap for the custom topic: "{topic}".

Return ONLY valid JSON object with this exact schema:
{{
  "title": "string",
  "description": "string",
  "track": "backend|frontend|system_design|ai_stack",
  "levels": ["junior", "mid", "senior"],
  "sections": [
    {{
      "heading": "specific subtopic title",
      "content": "2-4 concise sentences with practical learning notes",
      "level": "junior|mid|senior"
    }}
  ]
}}

Hard constraints:
- sections length must be exactly {target_sections}.
- Ensure level progression and ordering:
  - first {junior_count} sections are junior,
  - next {mid_count} sections are mid,
  - final {senior_count} sections are senior.
- Subtopics must be granular, non-duplicative, and prerequisite-aware.
- Each heading must be a real, concrete topic name specific to "{topic}".
- Do NOT use generic headings like "Module 1", "Part A", "Topic X", or placeholders.
- Cover a complete path from fundamentals to advanced architecture and interview tradeoffs.
- Ensure the roadmap includes:
  - foundational concepts and terminology,
  - setup/tooling and workflow,
  - core implementation patterns,
  - debugging and testing strategy,
  - performance, security, and reliability,
  - architecture/system tradeoffs,
  - production operations and leadership decisions.
- At least one section must explicitly cover each dimension above (no coverage gaps).
- Every section heading should stand alone as a teachable lesson title.
- Keep each section content concise and practical.
- Output JSON only with escaped newlines/quotes/backslashes.
{mcp_block}
"""


def _fallback_topic_detail(topic: str, topic_id: str, target_sections: int) -> TopicDetail:
    title = f"{topic.strip().title()} Interview Roadmap"
    description = (
        f"Comprehensive custom roadmap for {topic} covering fundamentals, implementation, "
        "architecture, performance, and interview-oriented tradeoffs."
    )
    track = _infer_track(topic)
    levels = list(_DEFAULT_LEVELS)
    sections = _normalise_sections([], topic, target_sections)
    return TopicDetail(
        id=topic_id,
        title=title,
        description=description,
        track=track,
        levels=levels,
        sections=sections,
        raw_content=_build_raw_content(title, description, sections),
    )


class CustomTopicGenerator:
    """Generate custom roadmap content with deterministic topic IDs."""

    def __init__(self, llm_client: LLMClient, mcp_gateway: MCPGateway | None = None):
        self.llm = llm_client
        self.mcp = mcp_gateway

    async def generate_topic(
        self,
        *,
        topic: str,
        target_sections: int = _DEFAULT_SECTIONS,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> TopicDetail:
        source_topic = (topic or "").strip()
        if not source_topic:
            source_topic = "Custom Topic"
        target = _clamp_target_sections(target_sections)
        topic_id = f"custom-{_slugify_topic(source_topic)}"
        mcp_context = ""
        if self.mcp:
            mcp_context = await self.mcp.gather_context(
                flow="custom_topic",
                query=f"{source_topic} technical learning roadmap and interview preparation",
                topic_id=topic_id,
                topic_title=source_topic,
            )
        prompt = _build_prompt(source_topic, target, mcp_context=mcp_context)

        for _ in range(_MAX_ATTEMPTS):
            result = await self.llm.completion(
                prompt,
                llm_config,
                user_identity=user_identity,
            )
            raise_if_policy_blocked_result(result)
            if not result.get("success"):
                continue
            payload = _parse_json_object(result.get("analysis", ""))
            if not payload:
                continue

            title = str(payload.get("title", "")).strip() or f"{source_topic.title()} Interview Roadmap"
            description = str(payload.get("description", "")).strip()
            if not description:
                description = (
                    f"Comprehensive custom roadmap for {source_topic} with interview-focused "
                    "subtopics and practical depth."
                )
            track = _normalise_track(str(payload.get("track", ""))) or _infer_track(source_topic)
            levels = _normalise_levels(payload.get("levels"))
            sections = _normalise_sections(payload.get("sections"), source_topic, target)
            if len(sections) < _MIN_SECTIONS:
                continue
            return TopicDetail(
                id=topic_id,
                title=title[:200],
                description=description[:600],
                track=track,
                levels=levels,
                sections=sections,
                raw_content=_build_raw_content(title, description, sections),
            )

        logger.warning("Falling back to synthetic custom roadmap for topic=%s", source_topic)
        return _fallback_topic_detail(source_topic, topic_id, target)
