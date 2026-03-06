"""Shared LLM access policy constants and helpers."""

from __future__ import annotations

from typing import Any

APPROVAL_REQUIRED_CODE = "llm_service_approval_required"
APPROVAL_REQUIRED_MESSAGE = (
    "Admin approval required for backend LLM service. "
    "Add your own LLM credential in User Settings or ask admin approval."
)


class LLMServiceApprovalRequiredError(Exception):
    """Raised when backend-funded fallback is blocked for a user."""

    code = APPROVAL_REQUIRED_CODE
    message = APPROVAL_REQUIRED_MESSAGE


def raise_if_policy_blocked_result(result: dict[str, Any]) -> None:
    error_code = str(result.get("error_code", "")).strip().lower()
    if error_code == APPROVAL_REQUIRED_CODE:
        raise LLMServiceApprovalRequiredError(APPROVAL_REQUIRED_MESSAGE)
