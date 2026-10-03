"""Shared question-text normalisation used by every question de-duplicator.

Kept in its own module so persistence layers (e.g. ``progress_store``) can reuse
the generator's exact normaliser without importing ``question_generator`` (and
therefore LiteLLM) at module import time.
"""

from __future__ import annotations

import re

_WHITESPACE_RUN = re.compile(r"\s+")


def normalise_question(text: str) -> str:
    """Collapse whitespace and lowercase so equal questions compare equal."""
    return _WHITESPACE_RUN.sub(" ", (text or "").strip().lower())
