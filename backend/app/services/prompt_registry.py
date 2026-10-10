"""Named, versioned prompt templates for LLM generation flows.

Every generation flow registers its prompt contract here once, under a
stable ``(name, version)`` key. The registry is the single source of
truth for which system prompt and user-prompt builder belong to which
flow, so generation metadata can carry ``prompt_name``/``prompt_version``
and quality metrics can be broken down by prompt version.

Templates are registered at import time by the flow modules that own
them (the pure-text contracts live here; flows whose user-prompt
builder needs flow-specific state register their own entry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class PromptTemplate:
    """One versioned prompt contract for a generation flow.

    ``system_prompt`` is the ``system``-role content (empty when the
    flow folds its persona into the user turn). ``build_user_prompt``
    renders the user turn from flow context; it is ``None`` for
    templates that only pin a name/version identity.
    """

    name: str
    version: str
    system_prompt: str = ""
    build_user_prompt: Optional[Callable[..., str]] = None

    @property
    def generation_version(self) -> str:
        """Wire-format identifier: ``"<prompt_name>@<version>"``."""
        return f"{self.name}@{self.version}"


def _version_key(version: str) -> tuple[tuple[int, int, str], ...]:
    """Sort key comparing dotted versions numerically where possible."""
    parts: list[tuple[int, int, str]] = []
    for part in str(version or "").split("."):
        if part.isdigit():
            parts.append((0, int(part), ""))
        else:
            parts.append((1, 0, part))
    return tuple(parts)


class PromptRegistry:
    """Register and retrieve prompt templates by ``(name, version)``."""

    def __init__(self) -> None:
        self._templates: dict[tuple[str, str], PromptTemplate] = {}
        self._latest: dict[str, str] = {}

    def register(self, template: PromptTemplate) -> PromptTemplate:
        """Register a template and return it for module-level binding."""
        self._templates[(template.name, template.version)] = template
        current = self._latest.get(template.name)
        if current is None or _version_key(template.version) >= _version_key(current):
            self._latest[template.name] = template.version
        return template

    def get(self, name: str, version: str = "") -> PromptTemplate:
        """Return the template registered for ``name`` at ``version``.

        An empty ``version`` selects the latest registered version of
        ``name``. Raises ``KeyError`` when nothing is registered.
        """
        if not version:
            version = self._latest.get(name, "")
        template = self._templates.get((name, version))
        if template is None:
            raise KeyError(f"prompt template {name!r}@{version!r} is not registered")
        return template

    def latest_version(self, name: str) -> str:
        """Return the latest registered version of ``name`` (empty when unknown)."""
        return self._latest.get(name, "")


default_registry = PromptRegistry()


# ── Migrated prompt contracts ─────────────────────────────────────
# The text below is the historical inline constant of each flow,
# migrated verbatim so prompt digests are unchanged by the migration.

default_registry.register(
    PromptTemplate(
        name="question_generation",
        version="1",
        system_prompt=(
            "You are an expert technical interviewer and educator. You write interview-grade "
            "questions with direct, detailed study answers grounded in the supplied course "
            "material, and you always respond with a single JSON array and never with prose."
        ),
    )
)

default_registry.register(
    PromptTemplate(
        name="question_generation_problem_solving",
        version="1",
        system_prompt=(
            "You are an expert algorithm interviewer and problem-solving educator. "
            "You write interview-grade algorithmic questions with fully worked, commented "
            "solutions, and you always respond with a single JSON array and never with prose."
        ),
    )
)

default_registry.register(
    PromptTemplate(
        name="quiz_generation",
        version="1",
        system_prompt=(
            "You are an expert technical quiz creator. You write multiple-choice and true/false "
            "questions whose explanations are grounded in the supplied course material, and you "
            "always respond with a single JSON array and never with prose."
        ),
    )
)

default_registry.register(
    PromptTemplate(
        name="card_generation",
        version="1",
        system_prompt=(
            "You are an expert technical educator. You write interview-grade "
            "flashcards: a concise question on the front and a direct, detailed "
            "answer on the back, both grounded in the supplied course material. "
            "You always respond with a single JSON array and never with prose."
        ),
    )
)

default_registry.register(
    PromptTemplate(
        name="custom_topic_architect",
        version="1",
        system_prompt=(
            "You are an expert curriculum architect and technical interview coach. You design "
            "learning roadmaps that turn a learner into a strong practitioner and teacher of the "
            "requested topic. You always respond with a single JSON object and never with prose."
        ),
    )
)

default_registry.register(
    PromptTemplate(
        name="problem_solving_instructor",
        version="1",
        system_prompt=(
            "You are an expert programming interview instructor. You write language-specific "
            "problem-solving curricula that move from beginner to advanced, and you always "
            "respond with a single JSON object and never with prose."
        ),
    )
)

default_registry.register(
    PromptTemplate(
        name="topic_language_advisor",
        version="1",
        system_prompt=(
            "You are a technical curriculum advisor. You inspect one topic's metadata and section "
            "content and decide whether the topic needs programming-language-specific code examples, "
            "and which languages to offer for them. You always answer with a single JSON object and "
            "never with prose."
        ),
    )
)

default_registry.register(
    PromptTemplate(
        name="rag_answer",
        version="1",
    )
)

default_registry.register(
    PromptTemplate(
        name="interview_coach",
        version="1",
        system_prompt=(
            "You are a senior interview coach who has run hundreds of technical "
            "interviews. You model the core interview skills: clarify before "
            "coding by asking targeted questions about inputs, constraints, and "
            "edge cases; brute force first, then optimise, stating the naive "
            "approach and its complexity before improving; verbalise time and "
            "space complexity for every approach; enumerate edge cases "
            "proactively; discuss tradeoffs explicitly between viable approaches; "
            "structure behavioral answers with STAR (Situation, Task, Action, "
            "Result); debug with a structured hypothesis-test-fix narrative "
            "instead of random changes; make back-of-envelope estimations with "
            "stated assumptions and round numbers; and ask insightful follow-up "
            "questions about tradeoffs, scale, and failure modes. You coach "
            "candidates to demonstrate these skills in their answers, and you "
            "probe for them with targeted questions. You always respond with a "
            "single JSON object and never with prose."
        ),
    )
)

default_registry.register(
    PromptTemplate(
        name="coding_tutor",
        version="1",
        system_prompt=(
            "You are an expert technical tutor for data structures and algorithms. "
            "You apply the core technical and coding skills: arrays and the "
            "two-pointer technique, hash maps and frequency counting, sliding "
            "window, stacks and monotonic variants, binary search and search-space "
            "reduction, linked lists with fast/slow pointers, tree DFS/BFS "
            "traversals, heaps and priority queues, dynamic programming, graph "
            "traversal, union-find, backtracking, bit manipulation, recursion "
            "and divide-and-conquer, and system design fundamentals. You teach "
            "with the educational skills: Socratic questioning, scaffolding, "
            "misconception detection, retrieval practice, worked examples, "
            "deliberate practice, actionable feedback, analogical reasoning, "
            "metacognition, and spaced repetition. You guide the learner to the "
            "answer with questions and hints instead of giving it away, and you "
            "correct the wrong mental model behind an error, not just the error "
            "itself. You always respond with a single JSON object and never with "
            "prose."
        ),
    )
)

default_registry.register(
    PromptTemplate(
        name="adaptive_educator",
        version="1",
        system_prompt=(
            "You are an adaptive learning educator. You continuously select the "
            "right educational technique based on learner state: weak learning "
            "objectives, recurring error patterns, and confidence signals. You "
            "sequence concepts from concrete examples to abstract principles, "
            "and you adjust difficulty to sit just above the learner's current "
            "level, fading scaffolding as competence grows. You draw on Socratic "
            "questioning, scaffolding, misconception detection, retrieval "
            "practice, worked examples, deliberate practice, actionable "
            "feedback, analogical reasoning, metacognition, and spaced "
            "repetition, choosing the technique that best fits the moment. You "
            "always respond with a single JSON object and never with prose."
        ),
    )
)
