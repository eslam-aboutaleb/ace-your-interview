"""Readiness per role/company (review item 5).

Covers the deterministic role weight models and their alias
matching, the built-in company requirements catalog (including the
unknown-company default), the gap derivation that cites topics,
interview competencies and error patterns, the ranked best-fit role
list, the role/company-aware prioritisation inside
``build_recommendations`` (and therefore ``build_study_plan``), and
the ``GET /api/learning/readiness`` endpoint contract.

Everything is deterministic and offline: the tests seed only stored
evidence (FSRS cards, interview reports, error patterns) and never
call an LLM.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import learning
from app.services import learning_planner as lp
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore

ADAPTIVE_ENV = "STUDY_ENABLE_ADAPTIVE_LEARNING"

#: Every role id the built-in catalog ships.
_BUILTIN_ROLE_IDS = {
    "backend_engineer",
    "frontend_engineer",
    "fullstack_engineer",
    "platform_engineer",
    "ai_engineer",
}


class _ReadinessFixture:
    """Learning + planner stores on one temporary database."""

    def __enter__(self) -> "_ReadinessFixture":
        self._td = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._td.name, "learning.db")
        self.learning = LearningStore(self.db_path)
        self.planner = LearningPlannerStore(self.db_path)
        return self

    def __exit__(self, *exc_info) -> bool:
        self._td.cleanup()
        return False

    def set_profile(self, *, user_id: str, **payload) -> dict:
        return self.planner.upsert_profile(user_id=user_id, payload=payload)

    def seed_attempt(
        self,
        *,
        user_id: str,
        question_id: str,
        topic_id: str,
        is_correct: bool = False,
        confidence: int = 3,
        mode: str = "quiz",
    ) -> None:
        self.learning.record_attempt(
            user_id=user_id,
            question_id=question_id,
            topic_id=topic_id,
            user_answer="seeded answer",
            is_correct=is_correct,
            confidence=confidence,
            response_time_ms=900,
            mode=mode,
        )

    def decay_card(self, *, user_id: str, card_id: str, days: int = 120) -> None:
        """Backdate a card so its FSRS retention reads as weak and due."""
        now = datetime.now(UTC)
        self.learning._conn.execute(
            "UPDATE fsrs_cards SET last_review_at = ?, due_at = ? "
            "WHERE user_id = ? AND card_id = ?",
            (
                (now - timedelta(days=days)).isoformat(),
                (now - timedelta(days=1)).isoformat(),
                user_id,
                card_id,
            ),
        )
        self.learning._conn.commit()

    def decay_retention(
        self, *, user_id: str, card_id: str, days: int = 120
    ) -> None:
        """Backdate a card's retention while keeping it off the due queue.

        Leaves ``due_at`` in the future, so the diagnostic's due-count
        term stays zero and the priority arithmetic under test is
        driven by mastery, the role weights and the company weights
        instead of flooding the 100-point cap.
        """
        now = datetime.now(UTC)
        self.learning._conn.execute(
            "UPDATE fsrs_cards SET last_review_at = ?, due_at = ? "
            "WHERE user_id = ? AND card_id = ?",
            (
                (now - timedelta(days=days)).isoformat(),
                (now + timedelta(days=1)).isoformat(),
                user_id,
                card_id,
            ),
        )
        self.learning._conn.commit()

    def seed_weak_topic(
        self, *, user_id: str, question_id: str, topic_id: str, due: bool = True
    ) -> None:
        self.seed_attempt(
            user_id=user_id,
            question_id=question_id,
            topic_id=topic_id,
            is_correct=False,
            confidence=2,
        )
        if due:
            self.decay_card(user_id=user_id, card_id=question_id)
        else:
            self.decay_retention(user_id=user_id, card_id=question_id)

    def seed_interview_report(
        self,
        *,
        user_id: str,
        rubric: dict[str, float],
        overall: float | None = 60,
    ) -> None:
        payload: dict[str, object] = {"rubric_averages": rubric}
        if overall is not None:
            payload["overall_score"] = overall
        self.learning._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interview_reports (
                user_id TEXT NOT NULL,
                report_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.learning._conn.execute(
            "INSERT INTO interview_reports(user_id, report_json, updated_at) "
            "VALUES (?, ?, ?)",
            (
                user_id,
                lp.json.dumps(payload),
                datetime.now(UTC).isoformat(),
            ),
        )
        self.learning._conn.commit()

    def seed_interview_competency(
        self, *, user_id: str, dimension: str, is_correct: bool = False
    ) -> None:
        """One scored interview turn on a single competency dimension."""
        self.seed_attempt(
            user_id=user_id,
            question_id=f"interview-competency:{dimension}",
            topic_id="interview-competency",
            is_correct=is_correct,
            confidence=2 if not is_correct else 5,
            mode="interview",
        )

    def seed_error_pattern(
        self,
        *,
        user_id: str,
        topic_id: str,
        error_type: str,
        objective: str = "an objective",
        count: int = 1,
    ) -> None:
        for _ in range(max(1, count)):
            self.learning.record_error_pattern(
                user_id=user_id,
                topic_id=topic_id,
                objective=objective,
                error_type=error_type,
                explanation="seeded explanation",
                evidence="seeded evidence",
            )


class RoleModelTests(unittest.TestCase):
    def test_target_role_matches_the_builtin_catalog(self):
        for target_role, role_id in (
            ("Senior Backend Engineer", "backend_engineer"),
            ("Frontend Engineer", "frontend_engineer"),
            ("Full Stack Developer", "fullstack_engineer"),
            ("Platform Engineer", "platform_engineer"),
            ("Machine Learning Engineer", "ai_engineer"),
        ):
            with self.subTest(target_role=target_role):
                profile = {"target_role": target_role, "primary_track": "backend"}
                model = lp.resolve_role_model(profile)
                self.assertEqual(model["role_id"], role_id)
                self.assertEqual(model["match_source"], "catalog")

    def test_hyphenated_role_titles_match_their_word_aliases(self):
        for target_role, role_id in (
            ("Back-End Dev", "backend_engineer"),
            ("Front End Engineer", "frontend_engineer"),
            ("AI Engineer", "ai_engineer"),
            ("ML Engineer", "ai_engineer"),
            ("Systems Engineer", "platform_engineer"),
            ("Distributed Systems Engineer", "platform_engineer"),
        ):
            with self.subTest(target_role=target_role):
                model = lp.resolve_role_model(
                    {"target_role": target_role, "primary_track": "backend"}
                )
                self.assertEqual(model["role_id"], role_id)

    def test_short_alias_never_substring_matches(self):
        # "ai" must not fire on "Email Engineer" the way a substring
        # match would; the fallback keeps the primary track instead.
        model = lp.resolve_role_model(
            {"target_role": "Email Engineer", "primary_track": "backend"}
        )
        self.assertEqual(model["match_source"], "primary_track_default")
        self.assertEqual(model["role_id"], "backend_focus")

    def test_unmatched_role_falls_back_to_the_primary_track(self):
        for primary_track, role_id in (
            ("backend", "backend_focus"),
            ("frontend", "frontend_focus"),
            ("system_design", "system_design_focus"),
            ("ai_stack", "ai_stack_focus"),
        ):
            with self.subTest(primary_track=primary_track):
                model = lp.resolve_role_model(
                    {"target_role": "", "primary_track": primary_track}
                )
                self.assertEqual(model["role_id"], role_id)
                self.assertEqual(model["match_source"], "primary_track_default")
                self.assertEqual(
                    model["track_weights"].get(primary_track),
                    max(model["track_weights"].values()),
                )

    def test_unknown_primary_track_falls_back_to_backend(self):
        model = lp.resolve_role_model(
            {"target_role": "", "primary_track": "not-a-track"}
        )
        self.assertEqual(model["role_id"], "backend_focus")

    def test_every_builtin_weight_model_sums_to_one(self):
        for role in lp._ROLE_CATALOG:
            with self.subTest(role=role["role_id"]):
                self.assertAlmostEqual(
                    sum(lp._normalise_weights(role["track_weights"]).values()),
                    1.0,
                    places=6,
                )
                self.assertAlmostEqual(
                    sum(lp._normalise_weights(role["competency_weights"]).values()),
                    1.0,
                    places=6,
                )

    def test_a_zero_weight_map_degrades_to_an_even_split(self):
        weights = lp._normalise_weights({"backend": 0, "frontend": 0})
        self.assertEqual(len(weights), len(lp._TRACKS))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)


class CompanyModelTests(unittest.TestCase):
    def test_catalog_company_matches_its_tracks_archetype(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
            )
            model = lp.resolve_company_model(
                profile=fx.planner.get_profile(user_id="alice"),
                role_model=lp.resolve_role_model(
                    {"target_role": "Backend Engineer", "primary_track": "backend"}
                ),
                company="Google LLC",
            )
            self.assertEqual(model["match_source"], "catalog")
            self.assertEqual(model["company_key"], "google")
            self.assertEqual(model["bar"], 78)
            self.assertAlmostEqual(
                sum(model["track_weights"].values()), 1.0, places=6
            )

    def test_each_primary_track_has_a_handful_of_archetypes(self):
        for track in lp._TRACKS:
            with self.subTest(track=track):
                catalog = lp._COMPANY_CATALOG[track]
                self.assertGreaterEqual(len(catalog), 3)
                keys = {entry["company_key"] for entry in catalog}
                self.assertEqual(len(keys), len(catalog))
                for entry in catalog:
                    self.assertTrue(entry["company_label"])
                    self.assertAlmostEqual(
                        sum(lp._normalise_weights(entry["track_weights"]).values()),
                        1.0,
                        places=6,
                    )
                    self.assertAlmostEqual(
                        sum(
                            lp._normalise_weights(
                                entry["competency_weights"]
                            ).values()
                        ),
                        1.0,
                        places=6,
                    )
                    self.assertGreaterEqual(entry["bar"], 0.0)
                    self.assertLessEqual(entry["bar"], 100.0)

    def test_unknown_company_inherits_the_role_default_profile(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="frontend",
                target_role="Frontend Engineer",
            )
            profile = fx.planner.get_profile(user_id="alice")
            role_model = lp.resolve_role_model(profile)
            model = lp.resolve_company_model(
                profile=profile, role_model=role_model, company="Acme Corp"
            )
            self.assertEqual(model["match_source"], "default")
            self.assertEqual(model["company_key"], "acmecorp")
            self.assertEqual(model["bar"], lp._DEFAULT_COMPANY_BAR)
            self.assertEqual(model["track_weights"], role_model["track_weights"])
            self.assertEqual(
                model["competency_weights"], role_model["competency_weights"]
            )

    def test_company_catalog_is_scoped_to_the_primary_track(self):
        # A Google fit is computed against the backend catalog; a
        # learner on the AI stack meets the ai_stack archetypes.
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice", primary_track="ai_stack", target_role="AI Engineer"
            )
            profile = fx.planner.get_profile(user_id="alice")
            role_model = lp.resolve_role_model(profile)
            model = lp.resolve_company_model(
                profile=profile, role_model=role_model, company="OpenAI"
            )
            self.assertEqual(model["match_source"], "catalog")
            self.assertEqual(model["company_key"], "openai")
            missing = lp.resolve_company_model(
                profile=profile, role_model=role_model, company="Shopify"
            )
            self.assertEqual(missing["match_source"], "default")


class ReadinessComputationTests(unittest.TestCase):
    def test_no_evidence_reads_not_started_on_declared_confidence(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(user_id="cold", primary_track="backend")
            data = fx.planner.build_readiness(user_id="cold")
            role = data["role"]
            # No attempts, no interview evidence: the declared 1-5
            # confidence is all there is, so the label says so.
            self.assertEqual(role["match_source"], "primary_track_default")
            self.assertEqual(role["readiness"], 60.0)
            self.assertEqual(role["readiness_label"], "Not Started")
            self.assertEqual(role["evidence"]["topic_reviews"], 0)

    def test_role_weights_move_the_score_for_the_same_evidence(self):
        with _ReadinessFixture() as fx:
            # One strong frontend review, every other track untouched
            # at the declared-confidence default.
            fx.seed_attempt(
                user_id="alice",
                question_id="q-fe",
                topic_id="06-frontend-core-architecture",
                is_correct=True,
                confidence=5,
            )
            fx.set_profile(
                user_id="alice",
                primary_track="frontend",
                target_role="Frontend Engineer",
            )
            fe = fx.planner.build_readiness(user_id="alice")["role"]
            fx.set_profile(
                user_id="alice",
                primary_track="frontend",
                target_role="Backend Engineer",
            )
            be = fx.planner.build_readiness(user_id="alice")["role"]
            # The same strong frontend evidence is worth far more to a
            # frontend role than to a backend role.
            self.assertEqual(fe["role_id"], "frontend_engineer")
            self.assertEqual(be["role_id"], "backend_engineer")
            self.assertGreater(fe["readiness"], be["readiness"])
            fe_frontend = next(
                row for row in fe["track_breakdown"] if row["id"] == "frontend"
            )
            be_frontend = next(
                row for row in be["track_breakdown"] if row["id"] == "frontend"
            )
            self.assertGreater(fe_frontend["weight"], be_frontend["weight"])
            self.assertGreater(fe_frontend["contribution"], be_frontend["contribution"])

    def test_breakdown_contributions_sum_to_the_readiness_score(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
            )
            fx.seed_weak_topic(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend-fundamentals-and-http",
            )
            fx.seed_interview_competency(user_id="alice", dimension="reasoning_depth")
            fx.seed_interview_report(
                user_id="alice",
                rubric={"technical_accuracy": 4, "reasoning_depth": 2},
            )
            role = fx.planner.build_readiness(user_id="alice")["role"]
            total = sum(row["contribution"] for row in role["track_breakdown"])
            total += sum(
                row["contribution"] for row in role["competency_breakdown"]
            )
            self.assertAlmostEqual(total, role["readiness"], delta=0.05)
            gap_total = sum(row["gap_points"] for row in role["track_breakdown"])
            gap_total += sum(
                row["gap_points"] for row in role["competency_breakdown"]
            )
            self.assertAlmostEqual(gap_total, 100.0 - role["readiness"], delta=0.05)

    def test_interview_evidence_carries_a_quarter_of_the_blend(self):
        with _ReadinessFixture() as fx:
            # Track scores sit on the default confidence (60) with no
            # topic evidence; every interview dimension is a 5/5 (100).
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
            )
            fx.seed_interview_report(
                user_id="alice",
                rubric={
                    "technical_accuracy": 5,
                    "reasoning_depth": 5,
                    "communication_clarity": 5,
                    "completeness": 5,
                    "confidence_signal": 5,
                },
            )
            role = fx.planner.build_readiness(user_id="alice")["role"]
            # 60 * (1 - 0.25) + 100 * 0.25
            self.assertEqual(role["readiness"], 70.0)
            self.assertEqual(role["readiness_label"], "On Track")

    def test_a_perfect_interview_does_not_invent_track_evidence(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
            )
            for dimension in (
                "technical_accuracy",
                "reasoning_depth",
                "communication_clarity",
                "completeness",
                "confidence_signal",
            ):
                fx.seed_interview_competency(
                    user_id="alice", dimension=dimension, is_correct=True
                )
            fx.seed_interview_report(
                user_id="alice", rubric={"technical_accuracy": 5}
            )
            role = fx.planner.build_readiness(user_id="alice")["role"]
            self.assertEqual(role["readiness_label"], "Developing")
            self.assertEqual(role["evidence"]["interview_competency_reviews"], 5)
            self.assertEqual(role["evidence"]["topic_reviews"], 0)

    def test_the_diagnostic_blended_readiness_is_unchanged(self):
        # The role-weighted readiness is additive: run_diagnostic keeps
        # its single blended score so existing clients never move.
        with _ReadinessFixture() as fx:
            diagnostic = fx.planner.run_diagnostic(user_id="cold")
            self.assertEqual(diagnostic["readiness_score"], 60.0)
            fx.seed_interview_report(user_id="alice", rubric={}, overall=3)
            diagnostic = fx.planner.run_diagnostic(user_id="alice")
            self.assertGreater(diagnostic["readiness_score"], 0.0)

    def test_readiness_is_deterministic_for_the_same_state(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
                target_companies=["Google", "Acme"],
            )
            fx.seed_weak_topic(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend-fundamentals-and-http",
            )
            fx.seed_error_pattern(
                user_id="alice",
                topic_id="01-backend-fundamentals-and-http",
                error_type="recall_gap",
            )
            fx.seed_interview_report(
                user_id="alice", rubric={"reasoning_depth": 2}
            )
            first = fx.planner.build_readiness(user_id="alice")
            second = fx.planner.build_readiness(user_id="alice")
            self.assertEqual(first["role"]["readiness"], second["role"]["readiness"])
            self.assertEqual(
                [item["fit"] for item in first["companies"]],
                [item["fit"] for item in second["companies"]],
            )
            self.assertEqual(
                [item["role_id"] for item in first["best_fit_roles"]],
                [item["role_id"] for item in second["best_fit_roles"]],
            )


class GapDerivationTests(unittest.TestCase):
    def test_gaps_rank_by_weighted_gap_and_cite_their_evidence(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
            )
            fx.seed_weak_topic(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend-fundamentals-and-http",
            )
            fx.seed_error_pattern(
                user_id="alice",
                topic_id="01-backend-fundamentals-and-http",
                error_type="recall_gap",
                objective="http status codes",
            )
            role = fx.planner.build_readiness(user_id="alice")["role"]
            gaps = role["gap_dimensions"]
            self.assertTrue(gaps)
            ordered = [gap["gap_points"] for gap in gaps]
            self.assertEqual(ordered, sorted(ordered, reverse=True))
            # The backend track is the learner's heaviest weight, so a
            # weak backend topic is the top-weighted gap.
            self.assertEqual(gaps[0]["id"], "backend")
            evidence_kinds = {item["kind"] for item in gaps[0]["evidence"]}
            self.assertIn("topic", evidence_kinds)
            self.assertIn("error_pattern", evidence_kinds)
            error_refs = [
                item["ref"] for item in gaps[0]["evidence"]
                if item["kind"] == "error_pattern"
            ]
            self.assertEqual(error_refs, ["recall_gap"])

    def test_a_weak_interview_dimension_gap_cites_its_competency(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
            )
            fx.seed_interview_competency(
                user_id="alice", dimension="reasoning_depth"
            )
            fx.seed_error_pattern(
                user_id="alice",
                topic_id="interview-competency",
                error_type="shallow_answer",
            )
            fx.seed_interview_report(
                user_id="alice",
                rubric={
                    "technical_accuracy": 5,
                    "reasoning_depth": 2,
                    "communication_clarity": 5,
                    "completeness": 5,
                    "confidence_signal": 5,
                },
            )
            role = fx.planner.build_readiness(user_id="alice")["role"]
            competency_gaps = [
                gap
                for gap in role["gap_dimensions"]
                if gap["kind"] == "interview_competency"
            ]
            self.assertTrue(competency_gaps)
            reasoning = next(
                gap for gap in competency_gaps if gap["id"] == "reasoning_depth"
            )
            evidence_kinds = {item["kind"] for item in reasoning["evidence"]}
            # The competency card and the interview-sourced error
            # pattern both back the dimension gap.
            self.assertIn("interview_competency", evidence_kinds)
            self.assertIn("error_pattern", evidence_kinds)

    def test_a_maxed_out_dimension_is_not_a_gap(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
            )
            fx.seed_interview_report(
                user_id="alice",
                rubric={
                    "technical_accuracy": 5,
                    "reasoning_depth": 5,
                    "communication_clarity": 5,
                    "completeness": 5,
                    "confidence_signal": 5,
                },
            )
            role = fx.planner.build_readiness(user_id="alice")["role"]
            ids = {gap["id"] for gap in role["gap_dimensions"]}
            for dimension in (
                "technical_accuracy",
                "reasoning_depth",
                "communication_clarity",
                "completeness",
                "confidence_signal",
            ):
                self.assertNotIn(dimension, ids)


class BestFitRoleTests(unittest.TestCase):
    def test_best_fit_ranks_every_role_and_flags_the_target(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
            )
            fx.seed_attempt(
                user_id="alice",
                question_id="q-sd",
                topic_id="10-system-design-foundations",
                is_correct=True,
                confidence=5,
            )
            fx.seed_weak_topic(
                user_id="alice",
                question_id="q-fe",
                topic_id="06-frontend-core-architecture",
            )
            data = fx.planner.build_readiness(user_id="alice")
            best_fit = data["best_fit_roles"]
            self.assertEqual(
                {item["role_id"] for item in best_fit}, _BUILTIN_ROLE_IDS
            )
            scores = [item["readiness"] for item in best_fit]
            self.assertEqual(scores, sorted(scores, reverse=True))
            flagged = [item for item in best_fit if item["is_target"]]
            self.assertEqual(len(flagged), 1)
            self.assertEqual(flagged[0]["role_id"], "backend_engineer")
            # System-design-heavy evidence ranks the platform role
            # above the learner's stated target.
            self.assertEqual(best_fit[0]["role_id"], "platform_engineer")

    def test_a_default_role_target_has_no_best_fit_match(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Email Engineer",
            )
            data = fx.planner.build_readiness(user_id="alice")
            self.assertTrue(
                all(not item["is_target"] for item in data["best_fit_roles"])
            )
            self.assertEqual(data["role"]["match_source"], "primary_track_default")


class CompanyFitTests(unittest.TestCase):
    def test_companies_are_ranked_by_fit_with_a_bar_verdict(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
                target_companies=["Google", "Amazon", "Acme Corp"],
            )
            fx.seed_attempt(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend-fundamentals-and-http",
                is_correct=True,
                confidence=5,
            )
            fx.seed_attempt(
                user_id="alice",
                question_id="q2",
                topic_id="10-system-design-foundations",
                is_correct=True,
                confidence=5,
            )
            data = fx.planner.build_readiness(user_id="alice")
            companies = data["companies"]
            self.assertEqual(len(companies), 3)
            fits = [item["fit"] for item in companies]
            self.assertEqual(fits, sorted(fits, reverse=True))
            by_source = {
                item["company"]: item for item in companies
            }
            self.assertEqual(by_source["Google"]["match_source"], "catalog")
            self.assertEqual(by_source["Google"]["bar"], 78)
            self.assertEqual(by_source["Acme Corp"]["match_source"], "default")
            self.assertEqual(by_source["Acme Corp"]["bar"], 70)
            for entry in companies:
                self.assertEqual(
                    entry["clears_bar"], entry["fit"] >= entry["bar"]
                )

    def test_a_weak_learner_does_not_clear_a_catalog_bar(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
                target_companies=["Google"],
            )
            fx.seed_weak_topic(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend-fundamentals-and-http",
            )
            entry = fx.planner.build_readiness(user_id="alice")["companies"][0]
            self.assertFalse(entry["clears_bar"])
            self.assertLess(entry["fit"], entry["bar"])
            self.assertTrue(entry["gap_dimensions"])

    def test_a_company_fit_gap_names_the_weak_required_track(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
                target_companies=["Google"],
            )
            fx.seed_weak_topic(
                user_id="alice",
                question_id="q-sd",
                topic_id="10-system-design-foundations",
            )
            entry = fx.planner.build_readiness(user_id="alice")["companies"][0]
            gap_ids = [gap["id"] for gap in entry["gap_dimensions"]]
            self.assertIn("system_design", gap_ids)
            self.assertIn("backend", gap_ids)


class RoleAwarePrioritisationTests(unittest.TestCase):
    def test_a_role_weighted_topic_outranks_an_equal_topic(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
                preferred_modalities=["study"],
            )
            # Equally weak, neither due: only the role weights differ.
            fx.seed_weak_topic(
                user_id="alice",
                question_id="q-be",
                topic_id="01-backend-fundamentals-and-http",
                due=False,
            )
            fx.seed_weak_topic(
                user_id="alice",
                question_id="q-fe",
                topic_id="06-frontend-core-architecture",
                due=False,
            )
            recs = fx.planner.build_recommendations(user_id="alice", limit=20)
            backend = next(
                item
                for item in recs["items"]
                if item["topic_id"] == "01-backend-fundamentals-and-http"
            )
            frontend = next(
                item
                for item in recs["items"]
                if item["topic_id"] == "06-frontend-core-architecture"
            )
            # Both topics are equally weak; only the role weights differ.
            self.assertLess(backend["priority"], 100.0)
            self.assertGreater(backend["priority"], frontend["priority"])
            # The bonus is proportional to the weight the role gives
            # the track, so both carry the code and the backend track
            # simply earns more of it.
            self.assertIn("role_priority", backend["reason_codes"])
            self.assertIn("role_priority", frontend["reason_codes"])

    def test_the_top_target_company_adds_a_company_fit_code(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
                target_companies=["Google"],
                preferred_modalities=["study"],
            )
            fx.seed_weak_topic(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend-fundamentals-and-http",
                due=False,
            )
            recs = fx.planner.build_recommendations(user_id="alice", limit=20)
            backend = next(
                item
                for item in recs["items"]
                if item["topic_id"] == "01-backend-fundamentals-and-http"
            )
            self.assertIn("company_fit", backend["reason_codes"])
            self.assertIn("role_priority", backend["reason_codes"])

    def test_without_target_companies_there_is_no_company_fit_code(self):
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
                preferred_modalities=["study"],
            )
            fx.seed_weak_topic(
                user_id="alice",
                question_id="q1",
                topic_id="01-backend-fundamentals-and-http",
                due=False,
            )
            recs = fx.planner.build_recommendations(user_id="alice", limit=20)
            backend = next(
                item
                for item in recs["items"]
                if item["topic_id"] == "01-backend-fundamentals-and-http"
            )
            self.assertNotIn("company_fit", backend["reason_codes"])

    def test_the_study_plan_stays_a_commitment_device(self):
        # The role/company-aware prioritisation must not disturb the
        # plan-commitment contract: task keys stay stable across
        # regeneration and completions survive it.
        with _ReadinessFixture() as fx:
            fx.set_profile(
                user_id="alice",
                primary_track="backend",
                target_role="Backend Engineer",
                target_companies=["Google"],
                preferred_modalities=["study", "quiz", "interview"],
            )
            for idx, topic_id in enumerate(
                (
                    "01-backend-fundamentals-and-http",
                    "02-backend-api-design-and-contracts",
                    "03-backend-data-modeling-and-persistence",
                )
            ):
                fx.seed_weak_topic(
                    user_id="alice",
                    question_id=f"q{idx}",
                    topic_id=topic_id,
                )
            first = fx.planner.build_study_plan(user_id="alice", days=2, daily_items=3)
            today = first["days_plan"][0]["date"]
            stored = fx.planner.list_plan_tasks(user_id="alice", task_date=today)
            self.assertEqual(len(stored), len(first["days_plan"][0]["tasks"]))
            self.assertEqual(first["total_tasks"], 6)
            self.assertTrue(all(task["task_key"] for task in stored))
            task = first["days_plan"][0]["tasks"][0]
            completed = fx.planner.complete_plan_task(
                user_id="alice", task_key=task["task_key"], task_date=today
            )
            self.assertTrue(completed["completed"])
            regenerated = fx.planner.build_study_plan(
                user_id="alice", days=2, daily_items=3
            )
            flags = [item["completed"] for item in regenerated["days_plan"][0]["tasks"]]
            self.assertEqual(flags.count(True), 1)
            self.assertEqual(
                [item["task_key"] for item in regenerated["days_plan"][0]["tasks"]],
                [item["task_key"] for item in first["days_plan"][0]["tasks"]],
            )
            self.assertEqual(
                regenerated["commitment"]["completed_tasks"], 1
            )


class RoleReadinessRouterTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._td.name, "learning.db")
        self.store = LearningStore(self.db_path)
        self.planner = LearningPlannerStore(self.db_path)
        learning.init(self.store, self.planner)
        self._prev_globals = (learning._store, learning._planner)
        self.prev_adaptive = os.environ.get(ADAPTIVE_ENV)
        os.environ[ADAPTIVE_ENV] = "true"
        get_settings.cache_clear()

        self.app = FastAPI()
        self.app.include_router(learning.router)
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "learner",
            "provider": "local",
        }
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        learning._store, learning._planner = self._prev_globals
        self._td.cleanup()
        if self.prev_adaptive is None:
            os.environ.pop(ADAPTIVE_ENV, None)
        else:
            os.environ[ADAPTIVE_ENV] = self.prev_adaptive
        get_settings.cache_clear()

    def test_readiness_endpoint_returns_the_role_and_company_payload(self):
        self.planner.upsert_profile(
            user_id="learner",
            payload={
                "primary_track": "backend",
                "target_role": "Backend Engineer",
                "target_companies": ["Google", "Acme Corp"],
            },
        )
        response = self.client.get("/api/learning/readiness")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["role"]["role_id"], "backend_engineer")
        self.assertIn("readiness", payload["role"])
        self.assertIn("track_breakdown", payload["role"])
        self.assertIn("gap_dimensions", payload["role"])
        self.assertEqual(len(payload["best_fit_roles"]), len(_BUILTIN_ROLE_IDS))
        by_company = {item["company"]: item for item in payload["companies"]}
        self.assertEqual(set(by_company), {"Google", "Acme Corp"})
        self.assertEqual(by_company["Google"]["match_source"], "catalog")
        self.assertEqual(by_company["Acme Corp"]["match_source"], "default")
        fits = [item["fit"] for item in payload["companies"]]
        self.assertEqual(fits, sorted(fits, reverse=True))
        for entry in payload["companies"]:
            self.assertIn("fit", entry)
            self.assertIn("fit_label", entry)
            self.assertIn("bar", entry)
            self.assertIn("clears_bar", entry)
            self.assertIn("gap_dimensions", entry)
        self.assertIn("profile", payload)
        self.assertIn("generated_at", payload)

    def test_readiness_endpoint_respects_the_adaptive_learning_flag(self):
        os.environ[ADAPTIVE_ENV] = "false"
        get_settings.cache_clear()
        response = self.client.get("/api/learning/readiness")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Adaptive learning disabled")

    def test_readiness_endpoint_reports_503_without_a_planner(self):
        learning._planner = None
        response = self.client.get("/api/learning/readiness")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"], "Learning planner not initialised"
        )

    def test_readiness_endpoint_requires_authentication(self):
        plain_app = FastAPI()
        plain_app.include_router(learning.router)
        client = TestClient(plain_app)
        self.assertEqual(client.get("/api/learning/readiness").status_code, 401)


if __name__ == "__main__":
    unittest.main()
