"""``submit_rubric``: the forced output tool for the interview evaluator.

The evaluator integration is not wired yet (``interview_generator`` is owned by
another change set), so these tests are the contract the integration must
satisfy: a well-formed JSON Schema the provider can validate against, and a
handler that rejects exactly the payloads the evaluator's own
``_validate_eval_payload`` would reject.
"""

import asyncio
import unittest

from app.services.tools.registry import ToolRegistry
from app.services.tools.submit_rubric import (
    ALLOWED_TOP_LEVEL_KEYS,
    LIST_MAX_ITEMS,
    RUBRIC_KEYS,
    RUBRIC_RANGES,
    SUBMIT_RUBRIC_TOOL_CHOICE,
    SUBMIT_RUBRIC_TOOL_NAME,
    RubricSubmissionError,
    submit_rubric_parameters,
    submit_rubric_tool,
    validate_rubric_submission,
)


def _valid_payload(**overrides):
    payload = {
        "rubric": {key: 3 for key in RUBRIC_KEYS},
        "strengths": ["Clear structure"],
        "improvements": ["Name the tradeoff"],
        "follow_up_note": (
            "### What strong interviewers wanted to hear\n"
            "Tradeoffs.\n\n"
            "### What to improve next\n"
            "- Add metrics.\n\n"
            "### Stronger sample answer\n"
            "A tighter answer."
        ),
    }
    payload["rubric"]["overall"] = 72
    payload.update(overrides)
    return payload


def _run(coro):
    return asyncio.run(coro)


class SchemaShapeTests(unittest.TestCase):
    def setUp(self):
        self.spec = submit_rubric_tool()

    def test_tool_name_and_description(self):
        self.assertEqual(self.spec.name, SUBMIT_RUBRIC_TOOL_NAME)
        self.assertIn("exactly once", self.spec.description)

    def test_openai_schema_is_well_formed(self):
        schema = self.spec.to_openai_schema()
        self.assertEqual(schema["type"], "function")
        function = schema["function"]
        self.assertEqual(function["name"], SUBMIT_RUBRIC_TOOL_NAME)
        params = function["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIs(params["additionalProperties"], False)
        self.assertEqual(
            sorted(params["required"]),
            ["follow_up_note", "improvements", "rubric", "strengths"],
        )
        rubric = params["properties"]["rubric"]
        self.assertEqual(sorted(rubric["required"]), sorted(RUBRIC_KEYS))
        self.assertIs(rubric["additionalProperties"], False)

    def test_rubric_bounds_match_the_evaluator_prompt(self):
        rubric = self.spec.parameters["properties"]["rubric"]["properties"]
        self.assertEqual((rubric["overall"]["minimum"], rubric["overall"]["maximum"]), (0, 100))
        for key in RUBRIC_KEYS:
            if key == "overall":
                continue
            self.assertEqual((rubric[key]["minimum"], rubric[key]["maximum"]), (0, 5))

    def test_list_bounds(self):
        for key in ("strengths", "improvements"):
            prop = self.spec.parameters["properties"][key]
            self.assertEqual(prop["type"], "array")
            self.assertEqual(prop["items"]["type"], "string")
            self.assertEqual(prop["minItems"], 1)
            self.assertEqual(prop["maxItems"], LIST_MAX_ITEMS)

    def test_schema_is_a_valid_json_schema(self):
        # jsonschema is a dev dependency; skip cleanly if it is absent.
        try:
            import jsonschema
        except ImportError:  # pragma: no cover
            self.skipTest("jsonschema not installed")
        jsonschema.Draft7Validator.check_schema(self.spec.parameters)
        validator = jsonschema.Draft7Validator(self.spec.parameters)
        self.assertEqual(list(validator.iter_errors(_valid_payload())), [])

    def test_schema_rejects_a_payload_missing_a_rubric_key(self):
        try:
            import jsonschema
        except ImportError:  # pragma: no cover
            self.skipTest("jsonschema not installed")
        payload = _valid_payload()
        payload["rubric"].pop("completeness")
        validator = jsonschema.Draft7Validator(self.spec.parameters)
        self.assertTrue(list(validator.iter_errors(payload)))

    def test_schema_rejects_out_of_range_scores(self):
        try:
            import jsonschema
        except ImportError:  # pragma: no cover
            self.skipTest("jsonschema not installed")
        payload = _valid_payload()
        payload["rubric"]["reasoning_depth"] = 9
        validator = jsonschema.Draft7Validator(self.spec.parameters)
        self.assertTrue(list(validator.iter_errors(payload)))

    def test_parameters_are_rebuilt_per_call(self):
        first = submit_rubric_parameters()
        first["properties"].clear()
        self.assertIn("rubric", submit_rubric_parameters()["properties"])

    def test_tool_choice_forces_the_single_tool(self):
        self.assertEqual(SUBMIT_RUBRIC_TOOL_CHOICE["type"], "function")
        self.assertEqual(
            SUBMIT_RUBRIC_TOOL_CHOICE["function"]["name"], SUBMIT_RUBRIC_TOOL_NAME
        )


class HandlerValidationTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry([submit_rubric_tool()])

    def _submit(self, payload):
        return _run(self.registry.invoke(SUBMIT_RUBRIC_TOOL_NAME, payload, tool_call_id="c1"))

    def test_valid_payload_is_returned_as_canonical_json(self):
        result = self._submit(_valid_payload())
        self.assertTrue(result.ok, result.error)
        import json

        decoded = json.loads(result.content)
        self.assertEqual(sorted(decoded["rubric"]), sorted(RUBRIC_KEYS))
        self.assertEqual(decoded["rubric"]["overall"], 72)

    def test_boundary_values_are_accepted(self):
        payload = _valid_payload()
        payload["rubric"] = {key: RUBRIC_RANGES[key][0] for key in RUBRIC_KEYS}
        self.assertTrue(self._submit(payload).ok)
        payload["rubric"] = {key: RUBRIC_RANGES[key][1] for key in RUBRIC_KEYS}
        self.assertTrue(self._submit(payload).ok)

    def test_missing_rubric_is_rejected(self):
        payload = _valid_payload()
        payload["rubric"] = "high"
        self.assertIn("missing_rubric", self._submit(payload).error)

    def test_missing_rubric_keys_are_named(self):
        payload = _valid_payload()
        payload["rubric"].pop("confidence_signal")
        payload["rubric"].pop("overall")
        error = self._submit(payload).error
        self.assertIn("confidence_signal", error)
        self.assertIn("overall", error)

    def test_out_of_range_score_is_rejected(self):
        payload = _valid_payload()
        payload["rubric"]["technical_accuracy"] = 6
        self.assertIn("technical_accuracy=6", self._submit(payload).error)

    def test_negative_score_is_rejected(self):
        payload = _valid_payload()
        payload["rubric"]["overall"] = -1
        self.assertIn("overall=-1", self._submit(payload).error)

    def test_non_numeric_score_is_rejected(self):
        payload = _valid_payload()
        payload["rubric"]["completeness"] = "high"
        self.assertIn("completeness must be a number", self._submit(payload).error)

    def test_boolean_score_is_rejected(self):
        payload = _valid_payload()
        payload["rubric"]["completeness"] = True
        self.assertIn("completeness must be a number", self._submit(payload).error)

    def test_fractional_score_is_rejected(self):
        payload = _valid_payload()
        payload["rubric"]["completeness"] = 3.5
        self.assertIn("whole number", self._submit(payload).error)

    def test_empty_strengths_is_rejected(self):
        self.assertIn("strengths needs at least", self._submit(_valid_payload(strengths=[])).error)

    def test_too_many_improvements_is_rejected(self):
        payload = _valid_payload(improvements=["a", "b", "c", "d", "e"])
        self.assertIn("at most 4", self._submit(payload).error)

    def test_non_list_strengths_is_rejected(self):
        self.assertIn("must be an array", self._submit(_valid_payload(strengths="ok")).error)

    def test_missing_note_is_rejected(self):
        self.assertIn("missing follow_up_note", self._submit(_valid_payload(follow_up_note="")).error)

    def test_unknown_top_level_key_is_rejected(self):
        self.assertIn("unexpected keys", self._submit(_valid_payload(extra="x")).error)

    def test_unknown_rubric_key_is_rejected(self):
        payload = _valid_payload()
        payload["rubric"]["vibes"] = 5
        self.assertIn("unexpected rubric keys", self._submit(payload).error)

    def test_malformed_provider_arguments_are_contained(self):
        result = _run(
            self.registry.invoke(SUBMIT_RUBRIC_TOOL_NAME, "{not json", tool_call_id="c1")
        )
        self.assertFalse(result.ok)
        self.assertIn("invalid_arguments", result.error)

    def test_validator_raises_directly_for_callers(self):
        with self.assertRaises(RubricSubmissionError):
            validate_rubric_submission({"rubric": {}})
        with self.assertRaises(RubricSubmissionError):
            validate_rubric_submission("nope")

    def test_allowed_keys_match_the_contract(self):
        self.assertEqual(
            ALLOWED_TOP_LEVEL_KEYS,
            frozenset({"rubric", "strengths", "improvements", "follow_up_note"}),
        )

    def test_disabled_spec_is_not_advertised(self):
        registry = ToolRegistry([submit_rubric_tool(enabled=False)])
        self.assertEqual(registry.list_enabled(), [])
        self.assertEqual(registry.to_openai_schema(), [])


if __name__ == "__main__":
    unittest.main()
