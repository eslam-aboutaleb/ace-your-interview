"""Shared LLM access policy constants and helpers."""

from __future__ import annotations

from typing import Any

APPROVAL_REQUIRED_CODE = "llm_service_approval_required"
APPROVAL_REQUIRED_MESSAGE = (
    "Admin approval required for backend LLM service. "
    "Add your own LLM credential in User Settings or ask admin approval."
)
STUDY_APP_NOT_ASSIGNED_CODE = "study_app_llm_not_assigned"
STUDY_APP_NOT_ASSIGNED_MESSAGE = (
    "Study App LLM is selected but no admin assignment exists yet. "
    "Ask admin to assign your Study App provider/model."
)
PERSONAL_CREDENTIAL_REQUIRED_CODE = "personal_credential_required"
PERSONAL_CREDENTIAL_REQUIRED_MESSAGE = (
    "Personal LLM mode requires your own credential. "
    "Add an API key/account in User Settings or switch to Study App LLM."
)


class LLMServiceApprovalRequiredError(Exception):
    """Raised when backend-funded fallback is blocked for a user."""

    code = APPROVAL_REQUIRED_CODE
    message = APPROVAL_REQUIRED_MESSAGE
    status_code = 403


class StudyAppLLMNotAssignedError(Exception):
    """Raised when Study App LLM is selected but no assignment exists."""

    code = STUDY_APP_NOT_ASSIGNED_CODE
    message = STUDY_APP_NOT_ASSIGNED_MESSAGE
    status_code = 403


class PersonalCredentialRequiredError(Exception):
    """Raised when personal mode is selected without personal credential."""

    code = PERSONAL_CREDENTIAL_REQUIRED_CODE
    message = PERSONAL_CREDENTIAL_REQUIRED_MESSAGE
    status_code = 400


def policy_error_detail(exc: Exception) -> dict[str, str]:
    return {
        "code": str(getattr(exc, "code", "")).strip().lower(),
        "message": str(getattr(exc, "message", "")).strip(),
    }


def raise_if_policy_blocked_result(result: dict[str, Any]) -> None:
    error_code = str(result.get("error_code", "")).strip().lower()
    if error_code == APPROVAL_REQUIRED_CODE:
        raise LLMServiceApprovalRequiredError(APPROVAL_REQUIRED_MESSAGE)
    if error_code == STUDY_APP_NOT_ASSIGNED_CODE:
        raise StudyAppLLMNotAssignedError(STUDY_APP_NOT_ASSIGNED_MESSAGE)
    if error_code == PERSONAL_CREDENTIAL_REQUIRED_CODE:
        raise PersonalCredentialRequiredError(PERSONAL_CREDENTIAL_REQUIRED_MESSAGE)
