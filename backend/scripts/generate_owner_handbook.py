#!/usr/bin/env python3
"""Generate deep static owner-handbook curriculum from a manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from app.services.llm_client import LLMClient
except Exception:  # pragma: no cover - optional runtime path
    LLMClient = None


TRACK_LABELS = {
    "backend": "backend systems",
    "frontend": "frontend applications",
    "system_design": "distributed systems",
    "ai_stack": "AI products",
}

LEVEL_EXPECTATIONS = [
    "junior: establish fundamentals and vocabulary",
    "mid: explain implementation constraints and tradeoffs",
    "senior: justify architecture and operational risk decisions",
]

SECTION_ANGLES = [
    "Foundations",
    "Mental Model",
    "Architecture Pattern",
    "Implementation Workflow",
    "Design Decisions",
    "Failure Modes",
    "Debugging Strategy",
    "Performance Lens",
    "Security Lens",
    "Reliability Lens",
    "Scalability Lens",
    "Testing Strategy",
    "Observability Signals",
    "Operational Checklist",
    "Interview Scenario",
    "Tradeoff Matrix",
    "Code Review Lens",
    "Migration Approach",
    "Production Runbook",
    "Advanced Considerations",
]

SCENARIO_SNIPPETS = [
    "Start from a concrete product requirement and map it to request, data, and dependency boundaries.",
    "Document expected behavior for success, degraded operation, and hard failure so behavior stays predictable.",
    "Tie the concept to implementation details such as contracts, storage choices, retries, and rollback strategy.",
    "Include instrumentation signals, alert thresholds, and triage steps that keep troubleshooting fast under pressure.",
    "Explicitly describe the failure blast radius and how to reduce it with isolation, throttling, and safe defaults.",
    "Capture the review checklist used in code review to prevent regressions before deployment.",
]

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", flags=re.DOTALL)
TITLE_RE = re.compile(r"^#\s+(.+)$", flags=re.MULTILINE)
SECTION_RE = re.compile(r"^##\s+", flags=re.MULTILINE)

DEFAULT_MAX_ATTEMPTS = 4
MIN_CONTENT_LEN = 180


@dataclass
class TopicFile:
    topic_id: str
    path: Path
    frontmatter_raw: str
    track: str
    levels: list[str]
    title: str
    description: str


@dataclass
class TopicManifest:
    target_sections: int
    coverage_clusters: list[str]


def _normalise_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_frontmatter(raw: str) -> tuple[str, str]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return "", raw
    return match.group(1).strip(), match.group(2)


def _extract_title(body: str, fallback: str) -> str:
    match = TITLE_RE.search(body)
    if match:
        return match.group(1).strip()
    return fallback


def _extract_description(body: str) -> str:
    lines = body.splitlines()
    title_seen = False
    buffer: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            title_seen = True
            continue
        if not title_seen:
            continue
        if stripped.startswith("## "):
            break
        if stripped:
            buffer.append(stripped)
        elif buffer:
            break
    return " ".join(buffer).strip()


def _parse_track(frontmatter: str) -> str:
    match = re.search(r"^track:\s*(.+)$", frontmatter, flags=re.MULTILINE)
    if not match:
        return "backend"
    return match.group(1).strip().lower().replace("-", "_")


def _parse_levels(frontmatter: str) -> list[str]:
    match = re.search(r"^levels:\s*(.+)$", frontmatter, flags=re.MULTILINE)
    if not match:
        return ["junior", "mid", "senior"]
    raw = match.group(1).strip()
    if raw.startswith("[") and raw.endswith("]"):
        return [x.strip() for x in raw[1:-1].split(",") if x.strip()]
    return [x.strip() for x in raw.split(",") if x.strip()]


def _discover_topics(docs_dir: Path) -> list[TopicFile]:
    topics: list[TopicFile] = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        topic_id = path.stem
        raw = path.read_text(encoding="utf-8")
        frontmatter_raw, body = _extract_frontmatter(raw)
        title = _extract_title(body, topic_id.replace("-", " ").title())
        description = _extract_description(body)
        topics.append(
            TopicFile(
                topic_id=topic_id,
                path=path,
                frontmatter_raw=frontmatter_raw,
                track=_parse_track(frontmatter_raw),
                levels=_parse_levels(frontmatter_raw),
                title=title,
                description=description,
            )
        )
    return topics


def _load_manifest(path: Path) -> dict[str, TopicManifest]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_topics = data.get("topics")
    if not isinstance(raw_topics, dict):
        raise ValueError("Manifest must include an object key named 'topics'.")

    out: dict[str, TopicManifest] = {}
    for topic_id, item in raw_topics.items():
        if not isinstance(item, dict):
            raise ValueError(f"Manifest topic '{topic_id}' must be an object.")
        target_sections = int(item.get("target_sections", 0))
        coverage_clusters = item.get("coverage_clusters")
        if target_sections <= 0:
            raise ValueError(f"Manifest topic '{topic_id}' has invalid target_sections.")
        if not isinstance(coverage_clusters, list) or not coverage_clusters:
            raise ValueError(f"Manifest topic '{topic_id}' has invalid coverage_clusters.")
        clusters = [str(c).strip() for c in coverage_clusters if str(c).strip()]
        out[topic_id] = TopicManifest(
            target_sections=target_sections,
            coverage_clusters=clusters,
        )
    return out


def _parse_json_payload(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}

    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _validate_sections_payload(
    *,
    payload: dict[str, Any],
    target_sections: int,
    coverage_clusters: list[str],
) -> tuple[bool, str]:
    title = payload.get("title")
    description = payload.get("description")
    sections = payload.get("sections")
    if not isinstance(title, str) or len(title.strip()) < 3:
        return False, "invalid_title"
    if not isinstance(description, str) or len(description.strip()) < 20:
        return False, "invalid_description"
    if not isinstance(sections, list):
        return False, "invalid_sections_type"
    if len(sections) != target_sections:
        return False, "wrong_sections_count"

    seen_headings: set[str] = set()
    corpus_text_parts: list[str] = [title, description]
    for section in sections:
        if not isinstance(section, dict):
            return False, "section_not_object"
        heading = str(section.get("heading", "")).strip()
        content = str(section.get("content", "")).strip()
        if len(heading) < 6:
            return False, "heading_too_short"
        if len(content) < MIN_CONTENT_LEN:
            return False, "content_too_short"
        heading_key = _normalise_key(heading)
        if heading_key in seen_headings:
            return False, "duplicate_heading"
        seen_headings.add(heading_key)
        corpus_text_parts.extend([heading, content])

    corpus_text = " ".join(corpus_text_parts).lower()
    for cluster in coverage_clusters:
        if cluster.lower() not in corpus_text:
            return False, f"missing_cluster:{cluster}"

    return True, ""


def _build_llm_prompt(topic: TopicFile, target_sections: int, clusters: list[str]) -> str:
    clusters_bullets = "\n".join([f"- {c}" for c in clusters])
    level_expectations = "\n".join([f"- {line}" for line in LEVEL_EXPECTATIONS])
    return f"""You are designing a deep interview-prep curriculum.

Create full course content for this topic:
- topic_id: {topic.topic_id}
- title: {topic.title}
- track: {topic.track}
- levels: {", ".join(topic.levels)}
- required sections: exactly {target_sections}

Coverage clusters that must appear explicitly in headings or content:
{clusters_bullets}

Candidate expectations:
{level_expectations}

Return ONLY valid JSON object:
{{
  "title": "string",
  "description": "2-4 sentences",
  "sections": [
    {{
      "heading": "specific section title",
      "content": "2-4 practical interview-focused sentences"
    }}
  ]
}}

Rules:
- sections length must be exactly {target_sections}
- headings must be unique and non-overlapping
- content must be concrete, practical, and interview-oriented
- include architecture, implementation, debugging, performance, security/reliability, and operations where relevant
"""


def _deterministic_sections(
    *,
    topic: TopicFile,
    target_sections: int,
    clusters: list[str],
) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    seen: set[str] = set()
    track_label = TRACK_LABELS.get(topic.track, "software systems")

    idx = 0
    while len(sections) < target_sections:
        cluster = clusters[idx % len(clusters)]
        angle = SECTION_ANGLES[(idx // len(clusters)) % SECTION_ANGLES.__len__()]
        scenario = SCENARIO_SNIPPETS[(idx + (idx // max(1, len(clusters)))) % len(SCENARIO_SNIPPETS)]
        section_no = len(sections) + 1
        heading = f"{cluster}: {angle} {section_no:02d}"
        heading_key = _normalise_key(heading)
        if heading_key in seen:
            idx += 1
            continue
        seen.add(heading_key)

        level_focus = LEVEL_EXPECTATIONS[(section_no - 1) % len(LEVEL_EXPECTATIONS)]
        content = (
            f"This module expands {cluster.lower()} in {topic.title} so candidates can explain how the concept behaves in real {track_label} work, not only definitions. "
            f"{scenario} "
            f"Discuss concrete tradeoffs across correctness, latency, cost, and maintainability, then show how the approach changes by scope and scale. "
            f"Interview emphasis: {level_focus}."
        )
        sections.append({"heading": heading, "content": content})
        idx += 1

    return sections


def _render_markdown(
    *,
    frontmatter_raw: str,
    title: str,
    description: str,
    sections: list[dict[str, str]],
) -> str:
    parts: list[str] = []
    if frontmatter_raw.strip():
        parts.extend(["---", frontmatter_raw.strip(), "---", ""])
    parts.extend([f"# {title.strip()}", "", description.strip()])
    for section in sections:
        parts.extend(["", f"## {section['heading'].strip()}", "", section["content"].strip()])
    return "\n".join(parts).strip() + "\n"


def _topic_summary(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return len(SECTION_RE.findall(text))


def _check_curriculum(
    *,
    topics: list[TopicFile],
    manifest: dict[str, TopicManifest],
) -> int:
    missing_manifest = [t.topic_id for t in topics if t.topic_id not in manifest]
    if missing_manifest:
        print(f"ERROR: Missing manifest entries for topics: {', '.join(missing_manifest)}")
        return 1

    total = 0
    failures: list[str] = []
    for topic in topics:
        expected = manifest[topic.topic_id].target_sections
        found = _topic_summary(topic.path)
        total += found
        if found != expected:
            failures.append(
                f"{topic.topic_id}: expected {expected} sections, found {found}"
            )
        if found < 40:
            failures.append(f"{topic.topic_id}: must have at least 40 sections (found {found})")

    if len(topics) != 22:
        failures.append(f"expected 22 static topics, found {len(topics)}")
    if total < 1000:
        failures.append(f"total sections must be >= 1000 (found {total})")

    print(f"Checked {len(topics)} topics, total sections={total}")
    if failures:
        for line in failures:
            print(f"ERROR: {line}")
        return 1
    print("Curriculum checks passed.")
    return 0


def _has_runtime_llm_credentials() -> bool:
    env_keys = [
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "GITHUB_TOKEN",
    ]
    return any(os.getenv(key) for key in env_keys)


async def _generate_topic_sections(
    *,
    topic: TopicFile,
    manifest_item: TopicManifest,
    llm_mode: str,
    max_attempts: int,
) -> tuple[str, str, list[dict[str, str]], str]:
    should_try_llm = llm_mode == "always" or (llm_mode == "auto" and _has_runtime_llm_credentials())
    if should_try_llm and LLMClient is None:
        if llm_mode == "always":
            raise RuntimeError("LLM mode is 'always' but backend LLM client import failed.")
        should_try_llm = False

    if should_try_llm:
        client = LLMClient()
        prompt = _build_llm_prompt(topic, manifest_item.target_sections, manifest_item.coverage_clusters)
        for attempt in range(1, max_attempts + 1):
            result = await client.completion(prompt)
            if not result.get("success"):
                continue
            payload = _parse_json_payload(str(result.get("analysis", "")))
            if not payload:
                continue
            is_valid, reason = _validate_sections_payload(
                payload=payload,
                target_sections=manifest_item.target_sections,
                coverage_clusters=manifest_item.coverage_clusters,
            )
            if not is_valid:
                prompt = (
                    f"{prompt}\n\nPrevious attempt failed validation: {reason}. "
                    "Regenerate and return only valid JSON."
                )
                continue
            title = str(payload.get("title", "")).strip() or topic.title
            description = str(payload.get("description", "")).strip() or topic.description
            sections = payload.get("sections") or []
            return title, description, sections, f"llm(attempt={attempt})"

        if llm_mode == "always":
            raise RuntimeError(f"LLM generation failed for topic {topic.topic_id} after {max_attempts} attempts.")

    sections = _deterministic_sections(
        topic=topic,
        target_sections=manifest_item.target_sections,
        clusters=manifest_item.coverage_clusters,
    )
    payload = {"title": topic.title, "description": topic.description or f"Comprehensive roadmap for {topic.title}.", "sections": sections}
    is_valid, reason = _validate_sections_payload(
        payload=payload,
        target_sections=manifest_item.target_sections,
        coverage_clusters=manifest_item.coverage_clusters,
    )
    if not is_valid:
        raise RuntimeError(f"Deterministic generation produced invalid payload for {topic.topic_id}: {reason}")
    return payload["title"], payload["description"], sections, "deterministic"


async def _run_generation(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    topics = _discover_topics(docs_dir)
    manifest = _load_manifest(manifest_path)

    if args.check_only:
        return _check_curriculum(topics=topics, manifest=manifest)

    selected_ids = set(args.topic_id or [])
    selected_topics = [t for t in topics if not selected_ids or t.topic_id in selected_ids]
    if selected_ids:
        missing = sorted(selected_ids - {t.topic_id for t in selected_topics})
        if missing:
            print(f"ERROR: topic ids not found: {', '.join(missing)}")
            return 1

    if not selected_topics:
        print("No topics selected.")
        return 1

    for topic in selected_topics:
        manifest_item = manifest.get(topic.topic_id)
        if not manifest_item:
            print(f"ERROR: missing manifest entry for topic {topic.topic_id}")
            return 1

        title, description, sections, source = await _generate_topic_sections(
            topic=topic,
            manifest_item=manifest_item,
            llm_mode=args.llm_mode,
            max_attempts=args.max_attempts,
        )
        rendered = _render_markdown(
            frontmatter_raw=topic.frontmatter_raw,
            title=title,
            description=description,
            sections=sections,
        )

        if args.dry_run:
            print(
                f"[DRY-RUN] {topic.topic_id}: sections={len(sections)} source={source}"
            )
            continue

        topic.path.write_text(rendered, encoding="utf-8")
        print(f"WROTE {topic.topic_id}: sections={len(sections)} source={source}")

    if not args.dry_run and not selected_ids:
        post_topics = _discover_topics(docs_dir)
        return _check_curriculum(topics=post_topics, manifest=manifest)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    backend_root = Path(__file__).resolve().parents[1]
    default_docs = backend_root / "docs" / "owner-handbook"
    default_manifest = default_docs / "coverage_manifest.json"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        default=str(default_docs),
        help="Path to owner-handbook markdown directory.",
    )
    parser.add_argument(
        "--manifest",
        default=str(default_manifest),
        help="Path to coverage manifest JSON.",
    )
    parser.add_argument(
        "--topic-id",
        action="append",
        help="Limit generation/check to a specific topic id (can be used multiple times).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum LLM attempts per topic.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["auto", "always", "never"],
        default="auto",
        help="LLM usage mode. 'auto' uses LLM when credentials are present, otherwise deterministic.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generation summary without writing files.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate existing handbook against manifest and exit.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_run_generation(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pragma: no cover - CLI hard failure path
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
