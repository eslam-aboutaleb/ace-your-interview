"""Coverage for ``app.services.llm_policy``.

The policy module is the single place that decides whether a failed LLM call is
a *policy* refusal (raise) or a *terminal/transport* failure (no raise), so the
tests pin both the raising codes and the codes that must stay non-raising.
"""

import unittest

from app.services.llm_policy import (
    APPROVAL_REQUIRED_CODE,
    APPROVAL_REQUIRED_MESSAGE,
    LLMServiceApprovalRequiredError,
    PERSONAL_CREDENTIAL_REQUIRED_CODE,
    PERSONAL_CREDENTIAL_REQUIRED_MESSAGE,
    PersonalCredentialRequiredError,
    STUDY_APP_NOT_ASSIGNED_CODE,
    STUDY_APP_NOT_ASSIGNED_MESSAGE,
    StudyAppLLMNotAssignedError,
    policy_error_detail,
    raise_if_policy_blocked_result,
)


class PolicyErrorDetailTests(unittest.TestCase):
    def test_detail_uses_exception_class_attributes(self):
        detail = policy_error_detail(LLMServiceApprovalRequiredError())
        self.assertEqual(detail["code"], APPROVAL_REQUIRED_CODE)
        self.assertEqual(detail["message"], APPROVAL_REQUIRED_MESSAGE)

    def test_detail_lowercases_and_strips_the_code(self):
        class _Weird(Exception):
            code = "  MixedCase_Code  "
            message = "  spaced message  "

        detail = policy_error_detail(_Weird())
        self.assertEqual(detail["code"], "mixedcase_code")
        self.assertEqual(detail["message"], "spaced message")

    def test_detail_of_a_plain_exception_has_empty_fields(self):
        self.assertEqual(policy_error_detail(ValueError("boom")), {"code": "", "message": ""})

    def test_detail_accepts_a_cause_without_code_or_message(self):
        exc = LLMServiceApprovalRequiredError(APPROVAL_REQUIRED_MESSAGE)
        self.assertEqual(policy_error_detail(exc)["code"], APPROVAL_REQUIRED_CODE)


class RaiseIfPolicyBlockedResultTests(unittest.TestCase):
    def test_approval_required_raises(self):
        with self.assertRaises(LLMServiceApprovalRequiredError) as ctx:
            raise_if_policy_blocked_result({"error_code": APPROVAL_REQUIRED_CODE})
        self.assertEqual(ctx.exception.code, APPROVAL_REQUIRED_CODE)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_study_app_not_assigned_raises(self):
        with self.assertRaises(StudyAppLLMNotAssignedError) as ctx:
            raise_if_policy_blocked_result({"error_code": STUDY_APP_NOT_ASSIGNED_CODE})
        self.assertEqual(ctx.exception.message, STUDY_APP_NOT_ASSIGNED_MESSAGE)

    def test_personal_credential_required_raises(self):
        with self.assertRaises(PersonalCredentialRequiredError) as ctx:
            raise_if_policy_blocked_result({"error_code": PERSONAL_CREDENTIAL_REQUIRED_CODE})
        self.assertEqual(ctx.exception.message, PERSONAL_CREDENTIAL_REQUIRED_MESSAGE)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_code_matching_is_case_and_whitespace_insensitive(self):
        with self.assertRaises(LLMServiceApprovalRequiredError):
            raise_if_policy_blocked_result({"error_code": f"  {APPROVAL_REQUIRED_CODE.upper()} "})

    def test_terminal_and_transport_codes_do_not_raise(self):
        for code in ("", "llm_budget_exceeded", "llm_truncated", "llm_call_failed", "some_other"):
            with self.subTest(code=code):
                self.assertIsNone(raise_if_policy_blocked_result({"error_code": code}))

    def test_missing_or_non_string_error_code_does_not_raise(self):
        self.assertIsNone(raise_if_policy_blocked_result({}))
        self.assertIsNone(raise_if_policy_blocked_result({"error_code": None}))
        self.assertIsNone(raise_if_policy_blocked_result({"error_code": None}))


class PolicyErrorClassTests(unittest.TestCase):
    def test_each_error_carries_its_own_code_and_status(self):
        cases = [
            (LLMServiceApprovalRequiredError, APPROVAL_REQUIRED_CODE, 403),
            (StudyAppLLMNotAssignedError, STUDY_APP_NOT_ASSIGNED_CODE, 403),
            (PersonalCredentialRequiredError, PERSONAL_CREDENTIAL_REQUIRED_CODE, 400),
        ]
        for cls, code, status in cases:
            with self.subTest(cls=cls.__name__):
                exc = cls("whatever")
                self.assertEqual(exc.code, code)
                self.assertEqual(exc.status_code, status)
                self.assertIsInstance(exc, Exception)


if __name__ == "__main__":
    unittest.main()