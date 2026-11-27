"""Unit tests for plan preview shaping and edited-plan validation (#6)."""

import unittest

from app.services.plan_preview import (
    PlanValidationError,
    mark_review_obligations,
    plan_stats,
    shape_plan_for_depth,
    validate_edited_plan,
)


def _plan(n_modules: int, objectives_per_module: int = 6) -> dict:
    return {
        "phases": [
            {
                "title": "Phase 1",
                "description": "",
                "modules": [
                    {
                        "title": f"Module {i}",
                        "description": "",
                        "objectives": [
                            {"title": f"Objective {i}.{j}", "kind": "concept", "source_section": f"Module {i}"}
                            for j in range(objectives_per_module)
                        ],
                    }
                    for i in range(n_modules)
                ],
            }
        ]
    }


class ShapeDepthTests(unittest.TestCase):
    def test_survey_caps_modules_and_objectives(self):
        shaped = shape_plan_for_depth(_plan(10, 6), "survey")
        modules = shaped["phases"][0]["modules"]
        self.assertEqual(len(modules), 4)
        for module in modules:
            self.assertEqual(len(module["objectives"]), 3)

    def test_standard_caps_modules_only(self):
        shaped = shape_plan_for_depth(_plan(20, 6), "standard")
        modules = shaped["phases"][0]["modules"]
        self.assertEqual(len(modules), 12)
        self.assertEqual(len(modules[0]["objectives"]), 6)

    def test_deep_keeps_everything(self):
        shaped = shape_plan_for_depth(_plan(20, 8), "deep")
        modules = shaped["phases"][0]["modules"]
        self.assertEqual(len(modules), 20)
        self.assertEqual(len(modules[0]["objectives"]), 8)

    def test_invalid_depth_rejected(self):
        with self.assertRaises(PlanValidationError):
            shape_plan_for_depth(_plan(1), "exhaustive")

    def test_shaping_drops_empty_phases(self):
        plan = {"phases": [_plan(2)["phases"][0], {"title": "Empty", "description": "", "modules": []}]}
        shaped = shape_plan_for_depth(plan, "standard")
        self.assertEqual(len(shaped["phases"]), 1)


class ReviewObligationTests(unittest.TestCase):
    def test_marks_all_objectives_when_topic_has_active_cards(self):
        plan, count = mark_review_obligations(_plan(2, 2), has_active_cards=True)
        self.assertEqual(count, 4)
        for module in plan["phases"][0]["modules"]:
            for objective in module["objectives"]:
                self.assertEqual(objective["coverage"], "review_suggested")

    def test_uncovered_without_cards(self):
        plan, count = mark_review_obligations(_plan(2, 2), has_active_cards=False)
        self.assertEqual(count, 0)
        self.assertEqual(
            plan["phases"][0]["modules"][0]["objectives"][0]["coverage"], "uncovered"
        )


class StatsTests(unittest.TestCase):
    def test_estimated_weeks_uses_hours_and_objectives(self):
        plan = _plan(2, 5)  # 10 objectives * 30min = 5 hours
        stats = plan_stats(plan, available_hours_per_week=5)
        self.assertEqual(stats["objective_count"], 10)
        self.assertEqual(stats["module_count"], 2)
        self.assertEqual(stats["estimated_weeks"], 1)
        stats = plan_stats(plan, available_hours_per_week=1)
        self.assertEqual(stats["estimated_weeks"], 5)

    def test_estimated_weeks_never_zero(self):
        stats = plan_stats(_plan(1, 1), available_hours_per_week=80)
        self.assertEqual(stats["estimated_weeks"], 1)


class ValidateEditedPlanTests(unittest.TestCase):
    def test_valid_plan_cleaned(self):
        cleaned = validate_edited_plan(_plan(1, 2))
        objective = cleaned["phases"][0]["modules"][0]["objectives"][0]
        self.assertEqual(set(objective), {"title", "kind", "source_section"})

    def test_rejects_non_dict_and_empty(self):
        with self.assertRaises(PlanValidationError):
            validate_edited_plan(["not", "a", "dict"])
        with self.assertRaises(PlanValidationError):
            validate_edited_plan({"phases": []})

    def test_rejects_empty_module_objectives(self):
        plan = {"phases": [{"title": "P", "modules": [{"title": "M", "objectives": []}]}]}
        with self.assertRaises(PlanValidationError):
            validate_edited_plan(plan)

    def test_rejects_bad_kind_and_titles(self):
        plan = _plan(1, 1)
        plan["phases"][0]["modules"][0]["objectives"][0]["kind"] = "homework"
        with self.assertRaises(PlanValidationError):
            validate_edited_plan(plan)
        plan = _plan(1, 1)
        plan["phases"][0]["modules"][0]["objectives"][0]["title"] = "x" * 121
        with self.assertRaises(PlanValidationError):
            validate_edited_plan(plan)
        plan = _plan(1, 1)
        plan["phases"][0]["title"] = ""
        with self.assertRaises(PlanValidationError):
            validate_edited_plan(plan)

    def test_rejects_oversized_module(self):
        plan = _plan(1, 17)
        with self.assertRaises(PlanValidationError):
            validate_edited_plan(plan)

    def test_rejects_too_many_phases(self):
        plan = {"phases": [_plan(1, 1)["phases"][0] for _ in range(13)]}
        with self.assertRaises(PlanValidationError):
            validate_edited_plan(plan)


if __name__ == "__main__":
    unittest.main()
