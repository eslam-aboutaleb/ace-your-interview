"""JD analyzer tests — gap precision on a fixture resume + JD.

Validation target (from the plan): a fixture resume and JD with
planted gaps must detect >= 80% of the gaps and never invent a
skill that is not literally in the JD text.
"""

import asyncio
import unittest

from app.services.jd_analyzer import (
    JDAnalyzer,
    _extract_requirements_deterministic,
    _filter_to_jd_text,
    _skills_match,
)
from app.services.llm_policy import (
    STUDY_APP_NOT_ASSIGNED_CODE,
    StudyAppLLMNotAssignedError,
)


class ScriptedLLM:
    """Return a fixed structured extraction, or a policy block."""

    def __init__(self, analysis: str = "", error_code: str = ""):
        self.analysis = analysis
        self.error_code = error_code
        self.calls = 0

    async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
        self.calls += 1
        if self.error_code:
            return {
                "success": False,
                "analysis": "",
                "metadata": {},
                "error": "blocked",
                "error_code": self.error_code,
                "finish_reason": "",
                "usage": {},
            }
        return {
            "success": True,
            "analysis": self.analysis,
            "metadata": {},
            "error_code": "",
            "finish_reason": "stop",
            "usage": {},
        }


# Fixture JD: every requirement is a lexicon skill so the
# deterministic fallback path extracts them all.
JD_TEXT = (
    "We are hiring a backend engineer. You will work with Python, "
    "FastAPI, Docker, Kubernetes, PostgreSQL, Redis, Kafka, "
    "Terraform, AWS and gRPC. You should have system design and "
    "microservices experience, plus strong testing habits."
)

# Fixture resume profile: covers half the JD requirements.
PROFILE = {
    "user_id": "u1",
    "skills": ["Python", "FastAPI", "Docker", "PostgreSQL", "testing"],
    "projects": ["Resume parser API"],
    "experience_years": 3.5,
    "roles": ["Backend Engineer"],
    "strengths": ["Clear tradeoff reasoning"],
    "weak_spots": ["No metrics on resume"],
    "raw_text_hash": "hash",
    "updated_at": "2026-01-01T00:00:00+00:00",
}

# The planted gaps: JD requirements the profile does not cover.
PLANTED_GAPS = [
    "kubernetes",
    "redis",
    "kafka",
    "terraform",
    "aws",
    "grpc",
    "system design",
    "microservices",
]


class DeterministicExtractionTests(unittest.TestCase):
    def test_lexicon_extraction_finds_all_planted_requirements(self):
        found = _extract_requirements_deterministic(JD_TEXT)
        found_lower = {skill.lower() for skill in found}
        for gap in PLANTED_GAPS:
            self.assertIn(gap, found_lower)
        for matched in ("python", "fastapi", "docker", "postgresql", "testing"):
            self.assertIn(matched, found_lower)

    def test_deterministic_extraction_never_invents_skills(self):
        jd_lower = JD_TEXT.lower()
        for skill in _extract_requirements_deterministic(JD_TEXT):
            self.assertIn(skill.lower(), jd_lower)


class SkillMatchingTests(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(_skills_match("Python", "python"))

    def test_fuzzy_match(self):
        self.assertTrue(_skills_match("reactjs", "React"))
        self.assertTrue(_skills_match("PostgreSQL", "postgres"))

    def test_no_match(self):
        self.assertFalse(_skills_match("python", "kafka"))
        self.assertFalse(_skills_match("", "python"))


class FilterToJdTextTests(unittest.TestCase):
    def test_hallucinated_skills_are_dropped(self):
        skills = ["Python", "QtQuick", "Haskell", "FastAPI"]
        kept = _filter_to_jd_text(skills, JD_TEXT)
        self.assertEqual(kept, ["Python", "FastAPI"])

    def test_deduplicates(self):
        kept = _filter_to_jd_text(["Python", "python", "PYTHON"], JD_TEXT)
        self.assertEqual(kept, ["Python"])


class GapAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = JDAnalyzer(ScriptedLLM())

    def test_gap_analysis_detects_planted_gaps(self):
        analysis = self.analyzer._build_analysis(
            jd_text=JD_TEXT,
            requirements=_extract_requirements_deterministic(JD_TEXT),
            profile=PROFILE,
        )
        gaps_lower = {gap.lower() for gap in analysis["gaps"]}
        detected = [gap for gap in PLANTED_GAPS if gap in gaps_lower]
        # >= 80% of the planted gaps must be detected.
        self.assertGreaterEqual(len(detected) / len(PLANTED_GAPS), 0.8)

    def test_matched_skills_are_the_covered_requirements(self):
        analysis = self.analyzer._build_analysis(
            jd_text=JD_TEXT,
            requirements=_extract_requirements_deterministic(JD_TEXT),
            profile=PROFILE,
        )
        matched_lower = {skill.lower() for skill in analysis["matched"]}
        for covered in ("python", "fastapi", "docker", "postgresql", "testing"):
            self.assertIn(covered, matched_lower)

    def test_no_invented_skills_in_any_section(self):
        analysis = self.analyzer._build_analysis(
            jd_text=JD_TEXT,
            requirements=_extract_requirements_deterministic(JD_TEXT),
            profile=PROFILE,
        )
        jd_lower = JD_TEXT.lower()
        for section in ("matched", "gaps", "weak_spots", "recommended_focus_areas"):
            for skill in analysis[section]:
                self.assertIn(
                    str(skill).lower(),
                    jd_lower,
                    f"{section} invented skill: {skill}",
                )

    def test_followups_and_focus_areas_derive_from_gaps(self):
        analysis = self.analyzer._build_analysis(
            jd_text=JD_TEXT,
            requirements=_extract_requirements_deterministic(JD_TEXT),
            profile=PROFILE,
        )
        self.assertTrue(analysis["likely_followups"])
        self.assertTrue(analysis["recommended_focus_areas"])
        # Every followup asks about a gap skill.
        gaps_lower = {gap.lower() for gap in analysis["gaps"]}
        for followup in analysis["likely_followups"]:
            self.assertTrue(followup)
            mentioned = [
                gap
                for gap in gaps_lower
                if gap in followup.lower()
            ]
            self.assertTrue(mentioned, f"followup mentions no gap: {followup}")
        # Every focus area is a gap or a weak spot, never a new skill.
        focus_lower = {area.lower() for area in analysis["recommended_focus_areas"]}
        gap_lower = {gap.lower() for gap in analysis["gaps"]}
        self.assertTrue(focus_lower.issubset(gap_lower))

    def test_no_profile_reports_everything_as_a_gap(self):
        analysis = self.analyzer._build_analysis(
            jd_text=JD_TEXT,
            requirements=_extract_requirements_deterministic(JD_TEXT),
            profile=None,
        )
        self.assertEqual(analysis["matched"], [])
        self.assertTrue(len(analysis["gaps"]) >= len(PLANTED_GAPS))

    def test_llm_extraction_path(self):
        llm = ScriptedLLM(
            '{"required_skills":["Python","Kubernetes","Redis"],'
            '"preferred_skills":["gRPC"],"responsibilities":["Build APIs"]}'
        )
        analyzer = JDAnalyzer(llm)
        analysis = asyncio.run(
            analyzer.analyze(jd_text=JD_TEXT, profile=PROFILE)
        )
        self.assertEqual(analysis["matched"], ["Python"])
        self.assertEqual(
            sorted(analysis["gaps"]),
            ["Kubernetes", "Redis", "gRPC"],
        )

    def test_llm_hallucinations_are_filtered(self):
        llm = ScriptedLLM(
            '{"required_skills":["Python","QtQuick","Haskell","FastAPI"]}'
        )
        analyzer = JDAnalyzer(llm)
        analysis = asyncio.run(
            analyzer.analyze(jd_text=JD_TEXT, profile=PROFILE)
        )
        self.assertEqual(analysis["matched"], ["Python", "FastAPI"])
        self.assertEqual(analysis["gaps"], [])

    def test_llm_failure_falls_back_to_deterministic(self):
        llm = ScriptedLLM(error_code="some_provider_error")
        analyzer = JDAnalyzer(llm)
        analysis = asyncio.run(
            analyzer.analyze(jd_text=JD_TEXT, profile=PROFILE)
        )
        gaps_lower = {gap.lower() for gap in analysis["gaps"]}
        detected = [gap for gap in PLANTED_GAPS if gap in gaps_lower]
        self.assertGreaterEqual(len(detected) / len(PLANTED_GAPS), 0.8)

    def test_policy_block_propagates(self):
        llm = ScriptedLLM(error_code=STUDY_APP_NOT_ASSIGNED_CODE)
        analyzer = JDAnalyzer(llm)
        with self.assertRaises(StudyAppLLMNotAssignedError):
            asyncio.run(analyzer.analyze(jd_text=JD_TEXT, profile=PROFILE))


if __name__ == "__main__":
    unittest.main()
