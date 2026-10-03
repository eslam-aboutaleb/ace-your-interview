"""JD analysis — requirement extraction + resume gap analysis.

Compares a job description against the user's stored
:class:`ResumeProfile` and produces matched skills, gaps,
weak spots, likely follow-ups, and recommended focus areas.

Safety invariant (enforced by :func:`_filter_to_jd_text`):
every skill the analyzer reports must literally appear in the
JD text, so the analyzer can never invent a skill.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.llm_client import LLMClient
from app.services.llm_policy import raise_if_policy_blocked_result

logger = logging.getLogger(__name__)

#: Maximum skills reported per section.
MAX_MATCHED = 30
MAX_GAPS = 20
MAX_WEAK_SPOTS = 10
MAX_FOLLOWUPS = 5
MAX_FOCUS_AREAS = 12

#: Token-overlap threshold for fuzzy skill matching.
_MATCH_JACCARD_THRESHOLD = 0.5

#: Curated lexicon for the deterministic fallback path. Every
#: entry is matched as a case-insensitive substring of the JD,
#: so fallback extraction can never produce a skill that is not
#: in the JD text.
_COMMON_SKILL_LEXICON: tuple[str, ...] = (
    "python", "javascript", "typescript", "java", "kotlin", "swift",
    "go", "golang", "rust", "c++", "c#", "ruby", "php", "scala",
    "r", "sql", "nosql", "graphql", "grpc", "rest", "html", "css",
    "react", "react native", "angular", "vue", "svelte", "next.js",
    "node.js", "express", "django", "fastapi", "flask", "spring",
    "spring boot", "rails", "laravel", "asp.net", "tensorflow",
    "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "machine learning", "deep learning", "nlp", "computer vision",
    "llm", "genai", "openai", "langchain", "vector database",
    "postgresql", "mysql", "mongodb", "redis", "cassandra",
    "elasticsearch", "kafka", "rabbitmq", "docker", "kubernetes",
    "k8s", "terraform", "aws", "gcp", "azure", "gce", "ec2",
    "s3", "lambda", "cloudflare", "git", "ci/cd", "jenkins",
    "github actions", "linux", "bash", "system design",
    "microservices", "distributed systems", "concurrency",
    "testing", "pytest", "jest", "cypress", "selenium",
    "figma", "tailwind", "sass", "webpack", "vite", "sqlite",
    "sqlite-vec", "prompt engineering", "rag", "embeddings",
    "observability", "grafana", "prometheus", "datadog",
    "agile", "scrum", "leadership", "communication",
    "system architecture", "api design", "caching",
    "load balancing", "rate limiting", "security", "oauth",
    "jwt", "performance optimization", "code review",
    "mentoring", "stakeholder management", "data modeling",
    "event-driven architecture", "cqrs", "websockets",
    "real-time", "accessibility", "responsive design",
    "design patterns", "algorithms", "data structures",
)


def _normalise_skill(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9+#.]+", "", text)
    return text


def _skill_tokens(value: str) -> set[str]:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s+#.]+", " ", text)
    return {token for token in text.split() if len(token) > 1}


def _skills_match(left: str, right: str) -> bool:
    """True when two skill strings refer to the same skill."""
    a = _normalise_skill(left)
    b = _normalise_skill(right)
    if not a or not b:
        return False
    if a == b:
        return True
    # Substring containment ("react" matches "reactjs", "sql"
    # matches "nosql" is avoided by requiring min length 4 for
    # the shorter side to reduce false positives).
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 4 and shorter in longer:
        return True
    tokens_a = _skill_tokens(left)
    tokens_b = _skill_tokens(right)
    if not tokens_a or not tokens_b:
        return False
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) >= _MATCH_JACCARD_THRESHOLD


def _filter_to_jd_text(skills: list[str], jd_text: str) -> list[str]:
    """Keep only skills that literally appear in the JD text."""
    jd_lower = str(jd_text or "").lower()
    out: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        cleaned = str(skill).strip()
        if not cleaned:
            continue
        key = _normalise_skill(cleaned)
        if not key or key in seen:
            continue
        if cleaned.lower() not in jd_lower:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _extract_requirements_deterministic(jd_text: str) -> list[str]:
    """Fallback requirement extraction via the curated lexicon."""
    jd_lower = str(jd_text or "").lower()
    found: list[str] = []
    for skill in _COMMON_SKILL_LEXICON:
        if skill in jd_lower:
            found.append(skill)
    return found


class JDAnalyzer:
    """Extract JD requirements and diff them against a resume profile."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def analyze(
        self,
        *,
        jd_text: str,
        profile: dict[str, Any] | None,
        llm_config: Any = None,
        user_identity: dict | None = None,
    ) -> dict[str, Any]:
        requirements = await self._extract_requirements(
            jd_text,
            llm_config=llm_config,
            user_identity=user_identity,
        )
        return self._build_analysis(jd_text=jd_text, requirements=requirements, profile=profile)

    async def _extract_requirements(
        self,
        jd_text: str,
        *,
        llm_config: Any = None,
        user_identity: dict | None = None,
    ) -> list[str]:
        prompt = self._build_prompt(jd_text)
        result = await self.llm.completion(
            prompt,
            llm_config,
            user_identity=user_identity,
            structured=True,
        )
        # Policy blocks (approval required, personal credential
        # required, study-app not assigned) must propagate to the
        # app-level handlers; only extraction failures fall back.
        raise_if_policy_blocked_result(result)
        payload: dict[str, Any] = {}
        if result.get("success"):
            try:
                payload = self._extract_json(result.get("analysis", ""))
            except Exception as exc:  # noqa: BLE001 - fallback is deterministic
                logger.warning("JD requirement JSON extraction failed: %s", exc)
        raw_skills: list[str] = []
        for key in ("required_skills", "preferred_skills", "skills"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_skills.extend(str(x) for x in value)
        if raw_skills:
            # Safety net: the LLM may hallucinate a skill that is
            # not in the JD; drop anything that does not literally
            # appear in the JD text.
            return _filter_to_jd_text(raw_skills, jd_text)

        return _extract_requirements_deterministic(jd_text)

    @staticmethod
    def _build_prompt(jd_text: str) -> str:
        bounded = str(jd_text or "")[:16000]
        return f"""You are a job description analyst. Extract the requirements from the job description below.

Return ONLY a JSON object with this exact shape:
{{
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1", "skill2"],
  "responsibilities": ["responsibility1", "responsibility2"]
}}

Rules:
- required_skills: technologies, languages, frameworks, and tools the candidate must have.
- preferred_skills: nice-to-have skills.
- Only extract skills that literally appear in the job description text. Do not invent or infer skills.
- Keep every string concise (<= 60 characters).

Job description:
{bounded}
"""

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass
        fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.DOTALL)
        if fenced:
            try:
                data = json.loads(fenced.group(1))
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                pass
        return {}

    def _build_analysis(
        self,
        *,
        jd_text: str,
        requirements: list[str],
        profile: dict[str, Any] | None,
    ) -> dict[str, Any]:
        profile_skills = [
            str(s).strip()
            for s in ((profile or {}).get("skills") or [])
            if str(s).strip()
        ]
        profile_weak_spots = [
            str(s).strip()
            for s in ((profile or {}).get("weak_spots") or [])
            if str(s).strip()
        ]

        matched: list[str] = []
        gaps: list[str] = []
        for requirement in requirements:
            if any(_skills_match(requirement, skill) for skill in profile_skills):
                matched.append(requirement)
            else:
                gaps.append(requirement)

        weak_spots = self._derive_weak_spots(
            profile_weak_spots=profile_weak_spots,
            requirements=requirements,
            gaps=gaps,
            profile_skills=profile_skills,
        )

        likely_followups = self._derive_followups(gaps)
        recommended_focus_areas = self._derive_focus_areas(gaps, weak_spots)

        return {
            "matched": matched[:MAX_MATCHED],
            "gaps": gaps[:MAX_GAPS],
            "weak_spots": weak_spots[:MAX_WEAK_SPOTS],
            "likely_followups": likely_followups[:MAX_FOLLOWUPS],
            "recommended_focus_areas": recommended_focus_areas[:MAX_FOCUS_AREAS],
        }

    @staticmethod
    def _derive_weak_spots(
        *,
        profile_weak_spots: list[str],
        requirements: list[str],
        gaps: list[str],
        profile_skills: list[str],
    ) -> list[str]:
        """Weak spots: profile-declared weak spots relevant to the JD,
        plus gaps adjacent to skills the candidate already has."""
        out: list[str] = []
        seen: set[str] = set()

        for weak in profile_weak_spots:
            if any(_skills_match(weak, req) for req in requirements):
                key = _normalise_skill(weak)
                if key and key not in seen:
                    seen.add(key)
                    out.append(weak)

        if not out:
            # Adjacent gaps: the candidate has a related skill but the
            # JD asks for a sibling they have not evidenced.
            profile_tokens: set[str] = set()
            for skill in profile_skills:
                profile_tokens |= _skill_tokens(skill)
            for gap in gaps:
                gap_tokens = _skill_tokens(gap)
                if gap_tokens & profile_tokens:
                    key = _normalise_skill(gap)
                    if key and key not in seen:
                        seen.add(key)
                        out.append(gap)
        return out

    @staticmethod
    def _derive_followups(gaps: list[str]) -> list[str]:
        templates = (
            "Tell me about your experience with {skill}.",
            "Where have you used {skill} in production?",
            "What tradeoffs did you weigh when adopting {skill}?",
        )
        followups: list[str] = []
        for index, gap in enumerate(gaps):
            if index >= MAX_FOLLOWUPS:
                break
            template = templates[index % len(templates)]
            followups.append(template.format(skill=gap))
        return followups

    @staticmethod
    def _derive_focus_areas(gaps: list[str], weak_spots: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in [*gaps, *weak_spots]:
            key = _normalise_skill(item)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= MAX_FOCUS_AREAS:
                break
        return out
