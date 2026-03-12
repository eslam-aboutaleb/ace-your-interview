"""Generate interview questions & answers via LLM with strict validation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, AsyncIterator, Callable, Optional

from app.schemas.models import (
    GenerateQuestionsResponse,
    GenerateQuestionsV2Response,
    GenerateQuizResponse,
    GenerateQuizV2Response,
    LLMConfigRequest,
    QuestionAnswer,
    QuestionAnswerV2,
    QuizChoice,
    QuizQuestion,
    QuizQuestionType,
    QuizQuestionV2,
)
from app.services.llm_client import LLMClient
from app.services.llm_policy import (
    APPROVAL_REQUIRED_CODE,
    APPROVAL_REQUIRED_MESSAGE,
    PERSONAL_CREDENTIAL_REQUIRED_CODE,
    PERSONAL_CREDENTIAL_REQUIRED_MESSAGE,
    STUDY_APP_NOT_ASSIGNED_CODE,
    STUDY_APP_NOT_ASSIGNED_MESSAGE,
    LLMServiceApprovalRequiredError,
    PersonalCredentialRequiredError,
    StudyAppLLMNotAssignedError,
    raise_if_policy_blocked_result,
)
from app.services.markdown_formatter import format_markdown_readable
from app.services.mcp_gateway import MCPGateway
from app.services.prompt_blocks import optional_context_block, render_contract
from app.services.topic_catalog import (
    PROBLEM_SOLVING_DEFAULT_LANGUAGE,
    is_problem_solving_topic,
)

logger = logging.getLogger(__name__)

_MAX_DOC_CONTEXT = 45000
_MAX_TOPIC_CONTEXT = 9000
_BASE_MAX_ATTEMPTS = 5
_MAX_TOTAL_ATTEMPTS = 18
_MAX_RECOVERY_ATTEMPTS = 24
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_LEVELS = {"junior", "mid", "senior"}
_PROBLEM_SOLVING_SECTION_HEADINGS = (
    "Problem",
    "Solution Walkthrough",
    "Complexity",
    "Code",
)
_PROBLEM_SOLVING_CODE_FENCE_RE = re.compile(
    r"```(?P<language>[a-zA-Z0-9_#+-]+)\s*\n(?P<code>[\s\S]*?)```",
    re.MULTILINE,
)
_PROBLEM_SOLVING_H3_RE = re.compile(r"^\s*###\s+(.+?)\s*$", re.MULTILINE)
_QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "why",
    "with",
    "would",
    "you",
}


def _question_id(topic_id: str, question: str) -> str:
    digest = hashlib.sha1(f"{topic_id}:{question.strip().lower()}".encode("utf-8")).hexdigest()
    return f"{topic_id}:{digest[:14]}"


def _normalise_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _question_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", _normalise_question(text))
    normalized: set[str] = set()
    for word in words:
        if len(word) <= 2 or word in _QUESTION_STOPWORDS:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = f"{word[:-3]}y"
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        normalized.add(word)
    return normalized


def _is_near_duplicate_question(candidate: str, seen_questions: list[str]) -> bool:
    cand_norm = _normalise_question(candidate)
    if not cand_norm:
        return True
    cand_tokens = _question_tokens(candidate)
    for seen in seen_questions:
        seen_norm = _normalise_question(seen)
        if not seen_norm:
            continue
        if cand_norm == seen_norm:
            return True
        if (cand_norm in seen_norm or seen_norm in cand_norm) and min(len(cand_norm), len(seen_norm)) >= 24:
            return True
        if SequenceMatcher(None, cand_norm, seen_norm).ratio() >= 0.87:
            return True
        seen_tokens = _question_tokens(seen)
        if cand_tokens and seen_tokens:
            overlap = len(cand_tokens & seen_tokens) / max(1, len(cand_tokens | seen_tokens))
            if overlap >= 0.65:
                return True
    return False


def _clamp_content(doc_content: str) -> str:
    return doc_content[:_MAX_DOC_CONTEXT]


def _normalise_level(level: Optional[str]) -> str:
    lv = (level or "mid").strip().lower()
    return lv if lv in _VALID_LEVELS else "mid"


def _attempt_budget(target_count: int) -> int:
    target = max(1, int(target_count or 1))
    return min(_MAX_TOTAL_ATTEMPTS, max(_BASE_MAX_ATTEMPTS, target + 3))


def _recovery_budget(missing_count: int) -> int:
    missing = max(1, int(missing_count or 1))
    return min(_MAX_RECOVERY_ATTEMPTS, max(3, missing * 3))


def _normalise_existing_questions(existing_questions: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in existing_questions or []:
        text = str(raw or "").strip()
        if not text:
            continue
        norm = _normalise_question(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
        if len(out) >= 80:
            break
    return out


def _problem_solving_language(preferred_language: str) -> str:
    language = (preferred_language or "").strip().lower()
    return language or PROBLEM_SOLVING_DEFAULT_LANGUAGE


_CONCEPTUAL_COACH_LABELS = (
    "**Answer:**",
    "**Why it's right:**",
    "**Interviewer-ready phrasing:**",
    "**Common mistake:**",
    "**Self-check:**",
)
_PROBLEM_SOLVING_WALKTHROUGH_LABELS = (
    "**Short answer:**",
    "**How to think about it:**",
    "**Why it works:**",
    "**Interviewer-ready phrasing:**",
    "**Common mistake:**",
)


def _question_tutoring_arc_rules(problem_solving_mode: bool, language: str = "") -> list[str]:
    if problem_solving_mode:
        walkthrough_labels = ", ".join([f"`{label}`" for label in _PROBLEM_SOLVING_WALKTHROUGH_LABELS])
        return [
            "Write each answer as coach-style study help for the learner, not as internal answer-writing guidance.",
            *_problem_solving_required_markdown(language),
            "In `### Problem`, begin with `**What the interviewer is really testing:**`.",
            f"In `### Solution Walkthrough`, include these bold labels in this order: {walkthrough_labels}.",
            (
                "In `### Complexity`, include `**Time and space:**` followed by "
                "`**Tradeoff / scaling caveat:**`."
            ),
            "Keep the answer generic and topic-grounded; do not inject named canned techniques unless the docs naturally justify them.",
        ]

    coach_labels = ", ".join([f"`{label}`" for label in _CONCEPTUAL_COACH_LABELS])
    return [
        "Write each answer as coach-style study help for the learner, not as internal answer-writing guidance.",
        f"For non-problem-solving answers, use these bold lead-ins in this exact order: {coach_labels}.",
        "Do not add `###` headings to non-problem-solving answers.",
        "The first sentence after `**Answer:**` must directly answer the question.",
        "Speak to the learner, not to the model.",
        (
            "Do not use meta-guideline phrasing such as `the candidate should`, `you should answer by`, "
            "or `when discussing this in an interview` outside `**Interviewer-ready phrasing:**`."
        ),
        "Keep all five coach segments even when the answer is concise; `response_detail` only changes depth inside them.",
    ]


def _question_auxiliary_field_rules(problem_solving_mode: bool) -> list[str]:
    reasoning_rule = (
        "Set `reasoning_summary` to a short mental model or invariant that explains how to choose the approach before coding."
        if problem_solving_mode
        else "Set `reasoning_summary` to a short mental-model takeaway, not answer-writing advice."
    )
    return [
        "Set `learning_objective` to a sentence that starts with `After this question, the learner should be able to...`.",
        "Set `misconception_trap` to a concrete wrong instinct or weak interview answer.",
        reasoning_rule,
    ]


def _question_retry_guidance(problem_solving_mode: bool) -> str:
    if problem_solving_mode:
        return "\n".join(
            [
                "- Regenerate each item as a fresh coach-style study answer for the learner.",
                "- Keep exactly these H3 headings in order: `### Problem`, `### Solution Walkthrough`, `### Complexity`, `### Code`.",
                "- In `### Problem`, start with `**What the interviewer is really testing:**`.",
                "- In `### Solution Walkthrough`, include final answer first, the mental model or invariant, interviewer-ready wording, one mistake to avoid, and one practice twist without adding new `###` headings.",
                "- In `### Complexity`, include `**Time and space:**` and `**Tradeoff / scaling caveat:**`.",
                "- In `### Code`, include exactly one commented fenced code block in the selected language, then `**Invariant note:**`.",
            ]
        )
    return "\n".join(
        [
            "- Regenerate each item as a fresh coach-style study answer for the learner.",
            "- Use only bold lead-ins in this exact order: `**Answer:**`, `**Why it's right:**`, `**Interviewer-ready phrasing:**`, `**Common mistake:**`, `**Self-check:**`.",
            "- The first sentence after `**Answer:**` must directly answer the question.",
            "- Speak to the learner, not to the model, and do not use meta-guideline phrasing outside `**Interviewer-ready phrasing:**`.",
        ]
    )


def _default_learning_objective(problem_solving_mode: bool) -> str:
    if problem_solving_mode:
        return (
            "After this question, the learner should be able to explain the core invariant and map it to a working implementation."
        )
    return (
        "After this question, the learner should be able to explain the current checkpoint and justify its practical tradeoffs."
    )


def _default_misconception_trap(problem_solving_mode: bool) -> str:
    if problem_solving_mode:
        return (
            "A weak interview answer starts coding before naming the invariant, the constraints, and the edge cases that drive the approach."
        )
    return (
        "A weak answer jumps to a preferred pattern before checking the actual requirements, tradeoffs, and failure modes."
    )


def _default_reasoning_summary(problem_solving_mode: bool) -> str:
    if problem_solving_mode:
        return "State the invariant first, then choose the structure that preserves it with the least extra work."
    return "Start from the core constraint, then connect it to the tradeoff that makes the answer correct."


def _problem_solving_required_markdown(language: str) -> list[str]:
    headings = ", ".join([f"`### {heading}`" for heading in _PROBLEM_SOLVING_SECTION_HEADINGS])
    return [
        f"Use exactly these markdown headings in this exact order: {headings}.",
        "Do not add extra `###` headings before, between, or after those sections.",
        f'Under `### Code`, include exactly one fenced `{language}` block.',
        "The code block must contain practical code, not pseudocode.",
        "Add comments that explain each key step or block in the code.",
        "After the code block, add `**Invariant note:**` describing what part of the code preserves the core invariant.",
    ]


_PROBLEM_SOLVING_DEFAULT_SCENARIO_ID = "two_sum_hash_map"
_PROBLEM_SOLVING_FALLBACK_SCENARIOS: tuple[dict[str, str], ...] = (
    {
        "id": "two_sum_hash_map",
        "question": (
            "In {source_scope}, consider an array-based input tied to {focus}. Return the two indices "
            "whose values satisfy a target-sum constraint. Explain your approach, complexity, and "
            "edge-case handling as you would in a live interview."
        ),
        "problem": (
            "Translate the prompt into inputs, outputs, and constraints for {focus}. You need a one-pass "
            "strategy that quickly checks whether the complement of the current value has already appeared."
        ),
        "walkthrough": (
            "Reject O(n^2) brute force once the complement relationship is clear. Use a hash map from "
            "value to index; for each number, look up its complement in O(1), then store the current "
            "value after the check so the same element is never reused."
        ),
        "complexity": (
            "The array is scanned once, so time is O(n). The hash map can store up to n values, so extra "
            "space is O(n). The tradeoff is additional memory for linear-time performance."
        ),
        "reasoning_summary": (
            "Identify the complement invariant first, then choose a hash map for one-pass lookup speed."
        ),
    },
    {
        "id": "longest_substring_sliding_window",
        "question": (
            "In {source_scope}, analyze a string scenario around {focus} and return the length of the "
            "longest substring without repeating characters. Explain how you reason about pointer movement."
        ),
        "problem": (
            "You need the best window length with all unique characters for {focus}. Rechecking every "
            "substring is too slow, so you need an incremental way to maintain validity while scanning."
        ),
        "walkthrough": (
            "Use a sliding window with a left pointer and a map of last-seen indices. When a repeated "
            "character appears inside the active window, move the left pointer past the previous index. "
            "Update window length on each step and keep the maximum."
        ),
        "complexity": (
            "Each character is processed at most twice (by right/left movement), so time is O(n). The "
            "last-seen map stores up to unique characters, so space is O(min(n, alphabet_size))."
        ),
        "reasoning_summary": (
            "Maintain a valid window invariant and move pointers only when the invariant is violated."
        ),
    },
    {
        "id": "merge_intervals_sorting",
        "question": (
            "Using the context of {focus} in {source_scope}, merge all overlapping intervals `[start, end]` "
            "and return the result. Explain your sorting and merge decisions clearly."
        ),
        "problem": (
            "Overlaps can appear in many positions, so local pair checks are unreliable unless intervals "
            "are ordered. The task is to collapse intersecting ranges into canonical disjoint intervals."
        ),
        "walkthrough": (
            "Sort intervals by start. Iterate once while building an output list: if the current interval "
            "does not overlap the last merged interval, append it; otherwise extend the previous end with "
            "the maximum endpoint."
        ),
        "complexity": (
            "Sorting dominates at O(n log n). The merge pass is O(n). Extra output storage is O(n) in the "
            "worst case when no intervals overlap."
        ),
        "reasoning_summary": (
            "Sort to linearize overlap checks, then maintain a single merged frontier while scanning."
        ),
    },
    {
        "id": "top_k_frequent_heap",
        "question": (
            "Given data from {source_scope} related to {focus}, return the `k` most frequent elements. "
            "Explain why your data structure choices fit interview constraints."
        ),
        "problem": (
            "You must rank elements by frequency without sorting every element-value pair when only top `k` "
            "results are needed. The challenge is balancing counting and extraction cost."
        ),
        "walkthrough": (
            "Count frequencies with a hash map, then use a heap to extract the top `k` entries by frequency. "
            "A max-heap gives direct top retrieval; alternatively, a size-limited min-heap can improve memory "
            "when `k` is much smaller than the number of unique values."
        ),
        "complexity": (
            "Frequency counting is O(n). Building and popping from a heap is O(m + k log m) for max-heap "
            "(or O(m log k) with a size-k min-heap), where m is number of unique values."
        ),
        "reasoning_summary": (
            "Separate counting from ranking, then use heap operations to avoid full-frequency sorting."
        ),
    },
    {
        "id": "search_rotated_binary_search",
        "question": (
            "In {source_scope}, you receive a rotated sorted array tied to {focus} and a target value. "
            "Return the index or -1 if not found, and explain how you adapt binary search under rotation."
        ),
        "problem": (
            "Standard binary search assumes full ordering, but rotation breaks global monotonicity. You "
            "must still use O(log n) decisions by identifying which half is sorted at each step."
        ),
        "walkthrough": (
            "At each midpoint, detect whether left or right half is sorted. Check whether the target lies "
            "within the sorted half's bounds; if yes, keep that half, otherwise search the other half. "
            "Repeat until found or bounds cross."
        ),
        "complexity": (
            "Each step halves the search interval, so time is O(log n). Space is O(1) with iterative bounds."
        ),
        "reasoning_summary": (
            "Recover binary-search pruning by exploiting the sorted half that still exists after rotation."
        ),
    },
    {
        "id": "number_of_islands_dfs",
        "question": (
            "Using {focus} as context in {source_scope}, given a 2D grid of `'1'` (land) and `'0'` (water), "
            "return the number of islands. Explain traversal strategy and how you avoid double-counting."
        ),
        "problem": (
            "Connected land cells form components. You need to count components once each, while ensuring "
            "visited land is not counted again from adjacent cells."
        ),
        "walkthrough": (
            "Scan each cell; when unvisited land is found, increment island count and run DFS/BFS to mark "
            "all connected land cells visited. Continue scanning until all cells are processed."
        ),
        "complexity": (
            "Every cell is visited at most once, so time is O(rows * cols). Visited tracking and recursion/"
            "queue storage are O(rows * cols) in the worst case."
        ),
        "reasoning_summary": (
            "Model the grid as connected components and mark each component exactly once."
        ),
    },
)
_PROBLEM_SOLVING_SCENARIOS_BY_ID = {
    item["id"]: item for item in _PROBLEM_SOLVING_FALLBACK_SCENARIOS
}
_PROBLEM_SOLVING_PYTHON_CODE_TEMPLATES: dict[str, str] = {
    "two_sum_hash_map": """def two_sum(nums, target):
    # Store each seen value with its index so complement lookups stay O(1).
    seen = {}

    # Scan once and check complement before storing the current value.
    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            # Return indices as soon as a valid pair is found.
            return [seen[complement], index]
        seen[value] = index

    # Return an empty result when no pair exists.
    return []
""",
    "longest_substring_sliding_window": """def length_of_longest_substring(s):
    # Track the latest index of each character for fast window fixes.
    last_seen = {}
    left = 0
    best = 0

    # Expand the window with `right` and shrink from `left` on duplicates.
    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        best = max(best, right - left + 1)

    # The best window length is the answer.
    return best
""",
    "merge_intervals_sorting": """def merge_intervals(intervals):
    # Sort by interval start so overlap checks become local.
    intervals.sort(key=lambda pair: pair[0])
    merged = []

    # Merge into the latest interval when ranges overlap.
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        merged[-1][1] = max(merged[-1][1], end)

    # Return merged disjoint intervals.
    return merged
""",
    "top_k_frequent_heap": """from collections import Counter
import heapq


def top_k_frequent(nums, k):
    # Count frequency of each value in O(n).
    freq = Counter(nums)

    # Use a max-heap via negative counts for top-k extraction.
    heap = [(-count, value) for value, count in freq.items()]
    heapq.heapify(heap)

    result = []
    for _ in range(min(k, len(heap))):
        _, value = heapq.heappop(heap)
        result.append(value)
    return result
""",
    "search_rotated_binary_search": """def search_rotated(nums, target):
    # Keep classic binary-search boundaries.
    left, right = 0, len(nums) - 1

    # One half is always sorted even after rotation.
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    # Return -1 when target is absent.
    return -1
""",
    "number_of_islands_dfs": """def num_islands(grid):
    # Guard empty input to avoid index errors.
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    visited = set()

    def dfs(r, c):
        # Stop at water, bounds, or previously visited cells.
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if grid[r][c] != "1" or (r, c) in visited:
            return
        visited.add((r, c))

        # Explore all four directions for the current island.
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            dfs(r + dr, c + dc)

    islands = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visited:
                islands += 1
                dfs(r, c)
    return islands
""",
}


def _problem_solving_scenario_ids_for_language(language: str) -> list[str]:
    if language == "python":
        return [item["id"] for item in _PROBLEM_SOLVING_FALLBACK_SCENARIOS]
    return [_PROBLEM_SOLVING_DEFAULT_SCENARIO_ID]


def _problem_solving_question_seed(
    focus: str,
    source_scope: str,
    *,
    scenario_id: str = _PROBLEM_SOLVING_DEFAULT_SCENARIO_ID,
) -> dict[str, str]:
    scenario = _PROBLEM_SOLVING_SCENARIOS_BY_ID.get(scenario_id) or _PROBLEM_SOLVING_SCENARIOS_BY_ID[
        _PROBLEM_SOLVING_DEFAULT_SCENARIO_ID
    ]
    focus_hint = (focus or source_scope or "the current checkpoint").strip()
    scope_hint = (source_scope or "this topic").strip() or "this topic"
    return {
        "question": scenario["question"].format(focus=focus_hint, source_scope=scope_hint),
        "problem": scenario["problem"].format(focus=focus_hint, source_scope=scope_hint),
        "walkthrough": scenario["walkthrough"].format(focus=focus_hint, source_scope=scope_hint),
        "complexity": scenario["complexity"].format(focus=focus_hint, source_scope=scope_hint),
        "reasoning_summary": scenario["reasoning_summary"].format(focus=focus_hint, source_scope=scope_hint),
    }


def _split_first_sentence(text: str) -> tuple[str, str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "", ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    first = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    return first, rest


def _split_last_sentence(text: str) -> tuple[str, str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "", ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(parts) == 1:
        return cleaned, cleaned
    return " ".join(parts[:-1]).strip(), parts[-1].strip()


def _fallback_learning_objective(*, focus: str, source_scope: str, problem_solving_mode: bool) -> str:
    if problem_solving_mode:
        return (
            f"After this question, the learner should be able to explain the invariant behind {focus} "
            f"and turn it into a working solution in {source_scope}."
        )
    return (
        f"After this question, the learner should be able to explain {focus} in {source_scope} "
        "and justify the practical tradeoff behind the answer."
    )


def _fallback_misconception_trap(*, focus: str, problem_solving_mode: bool) -> str:
    if problem_solving_mode:
        return (
            f"A weak interview answer for {focus} jumps into code before stating the invariant, "
            "constraints, and failure cases."
        )
    return (
        f"A weak answer for {focus} names a pattern but never connects it to the actual constraints, "
        "tradeoffs, or failure modes."
    )


def _build_conceptual_fallback_answer(
    *,
    focus: str,
    source_scope: str,
    angle: str,
) -> tuple[str, str]:
    reasoning_summary = (
        f"Name the main constraint around {focus}, then justify the choice that best protects {angle}."
    )
    answer = "\n\n".join(
        [
            (
                f"**Answer:** The right way to reason about {focus} in {source_scope} is to start from the "
                f"requirements, choose the approach that best protects {angle}, and then verify the edge cases."
            ),
            (
                f"**Why it's right:** In plain language, {focus} matters because real systems need a choice that "
                f"holds up under normal use, failure paths, and future growth. If you anchor the answer in the "
                "requirements first, the tradeoffs become easier to explain."
            ),
            (
                f"**Interviewer-ready phrasing:** \"I would frame {focus} around the actual constraints first, "
                f"then explain why this design gives the best balance for {angle} and what I would validate next.\""
            ),
            (
                f"**Common mistake:** A weak answer for {focus} jumps straight to a favorite pattern and never "
                "shows why it fits the real constraints or what could go wrong."
            ),
            (
                f"**Self-check:** If {angle} suddenly became the hardest constraint in {source_scope}, what part "
                "of your reasoning would you revisit first and why?"
            ),
        ]
    ).strip()
    return answer, reasoning_summary


def _problem_solving_code_template(
    language: str,
    *,
    scenario_id: str = _PROBLEM_SOLVING_DEFAULT_SCENARIO_ID,
) -> str:
    if language == "python":
        return _PROBLEM_SOLVING_PYTHON_CODE_TEMPLATES.get(
            scenario_id,
            _PROBLEM_SOLVING_PYTHON_CODE_TEMPLATES[_PROBLEM_SOLVING_DEFAULT_SCENARIO_ID],
        ).strip()

    templates = {
        "python": _PROBLEM_SOLVING_PYTHON_CODE_TEMPLATES[_PROBLEM_SOLVING_DEFAULT_SCENARIO_ID],
        "java": """import java.util.HashMap;
import java.util.Map;

class Solution {
    public int[] twoSum(int[] nums, int target) {
        // Store each seen value with its index so complement lookups stay O(1).
        Map<Integer, Integer> seen = new HashMap<>();

        // Scan the array once and look for the complement before storing the current value.
        for (int index = 0; index < nums.length; index++) {
            int value = nums[index];
            int complement = target - value;
            if (seen.containsKey(complement)) {
                // Return the earlier index and the current index as soon as the pair is found.
                return new int[] {seen.get(complement), index};
            }
            seen.put(value, index);
        }

        // Return an empty result when the input does not contain a valid pair.
        return new int[0];
    }
}
""",
        "cpp": """#include <unordered_map>
#include <vector>

using namespace std;

vector<int> twoSum(const vector<int>& nums, int target) {
    // Store each seen value with its index so complement lookups stay O(1).
    unordered_map<int, int> seen;

    // Scan the array once and look for the complement before storing the current value.
    for (int index = 0; index < static_cast<int>(nums.size()); index++) {
        int value = nums[index];
        int complement = target - value;
        auto found = seen.find(complement);
        if (found != seen.end()) {
            // Return the earlier index and the current index as soon as the pair is found.
            return {found->second, index};
        }
        seen[value] = index;
    }

    // Return an empty result when the input does not contain a valid pair.
    return {};
}
""",
        "javascript": """function twoSum(nums, target) {
  // Store each seen value with its index so complement lookups stay O(1).
  const seen = new Map();

  // Scan the array once and look for the complement before storing the current value.
  for (let index = 0; index < nums.length; index += 1) {
    const value = nums[index];
    const complement = target - value;
    if (seen.has(complement)) {
      // Return the earlier index and the current index as soon as the pair is found.
      return [seen.get(complement), index];
    }
    seen.set(value, index);
  }

  // Return an empty result when the input does not contain a valid pair.
  return [];
}
""",
        "csharp": """using System.Collections.Generic;

public class Solution
{
    public int[] TwoSum(int[] nums, int target)
    {
        // Store each seen value with its index so complement lookups stay O(1).
        var seen = new Dictionary<int, int>();

        // Scan the array once and look for the complement before storing the current value.
        for (var index = 0; index < nums.Length; index++)
        {
            var value = nums[index];
            var complement = target - value;
            if (seen.TryGetValue(complement, out var previousIndex))
            {
                // Return the earlier index and the current index as soon as the pair is found.
                return new[] { previousIndex, index };
            }
            seen[value] = index;
        }

        // Return an empty result when the input does not contain a valid pair.
        return new int[0];
    }
}
""",
        "go": """package main

func twoSum(nums []int, target int) []int {
    // Store each seen value with its index so complement lookups stay O(1).
    seen := make(map[int]int)

    // Scan the array once and look for the complement before storing the current value.
    for index, value := range nums {
        complement := target - value
        if previousIndex, ok := seen[complement]; ok {
            // Return the earlier index and the current index as soon as the pair is found.
            return []int{previousIndex, index}
        }
        seen[value] = index
    }

    // Return an empty result when the input does not contain a valid pair.
    return []int{}
}
""",
    }
    return templates.get(language, templates[PROBLEM_SOLVING_DEFAULT_LANGUAGE]).strip()


def _build_problem_solving_fallback_answer(
    *,
    focus: str,
    source_scope: str,
    language: str,
    scenario_id: str = _PROBLEM_SOLVING_DEFAULT_SCENARIO_ID,
) -> tuple[str, str]:
    seed = _problem_solving_question_seed(
        focus,
        source_scope,
        scenario_id=scenario_id,
    )
    code = _problem_solving_code_template(
        language,
        scenario_id=scenario_id,
    )
    short_answer, _remaining_walkthrough = _split_first_sentence(seed["walkthrough"])
    time_and_space, tradeoff = _split_last_sentence(seed["complexity"])
    answer = "\n\n".join(
        [
            "### Problem",
            (
                f"**What the interviewer is really testing:** Can you translate {focus} into inputs, "
                "constraints, and the invariant before you write code? "
                f"{seed['problem']}"
            ),
            "### Solution Walkthrough",
            f"**Short answer:** {short_answer}",
            f"**How to think about it:** {seed['problem']}",
            f"**Why it works:** {seed['walkthrough']}",
            (
                f"**Interviewer-ready phrasing:** \"The key idea is {seed['reasoning_summary'].rstrip('.')}. "
                "Once that is clear, I can justify the data structure, walk through the update rule, and then code it cleanly.\""
            ),
            (
                f"**Common mistake:** A weak answer for {focus} starts coding the brute-force idea before "
                "stating the invariant and why the chosen structure fits the constraint."
            ),
            "### Complexity",
            f"**Time and space:** {time_and_space}",
            f"**Tradeoff / scaling caveat:** {tradeoff}",
            "### Code",
            f"```{language}\n{code}\n```",
            (
                "**Invariant note:** The state update inside the main scan is what preserves the core invariant, "
                "so each step keeps the partial solution valid before the next iteration."
            ),
        ]
    ).strip()
    return answer, seed["reasoning_summary"]


def _validate_problem_solving_answer(answer: str, preferred_language: str) -> tuple[bool, str]:
    language = _problem_solving_language(preferred_language)
    answer_text = (answer or "").strip()
    headings = [
        match.group(1).strip()
        for match in _PROBLEM_SOLVING_H3_RE.finditer(
            _PROBLEM_SOLVING_CODE_FENCE_RE.sub("", answer_text)
        )
    ]
    if headings != list(_PROBLEM_SOLVING_SECTION_HEADINGS):
        return False, "invalid_problem_solving_heading_sequence"

    code_matches = list(_PROBLEM_SOLVING_CODE_FENCE_RE.finditer(answer_text))
    if len(code_matches) != 1:
        return False, "invalid_problem_solving_code_block_count"

    code_match = code_matches[0]
    if code_match.start() <= answer_text.find("### Code"):
        return False, "problem_solving_code_block_before_code_heading"

    fence_language = code_match.group("language").strip().lower()
    if fence_language != language:
        return False, "problem_solving_code_language_mismatch"

    code = code_match.group("code").strip()
    if not code:
        return False, "empty_problem_solving_code_block"

    comment_count = 0
    for raw_line in code.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if language == "python":
            if stripped.startswith("#"):
                comment_count += 1
        elif (
            stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith("*")
            or stripped.endswith("*/")
        ):
            comment_count += 1
    if comment_count < 2:
        return False, "problem_solving_code_missing_comments"

    return True, ""


def _problem_solving_pattern_signature(question: str, answer: str) -> str:
    text = f"{question}\n{answer}".strip().lower()
    if not text:
        return ""
    if (
        "two sum" in text
        or "two_sum" in text
        or "complement" in text
        or "def two_sum" in text
        or "twosum(" in text
        or ("target" in text and "indices" in text and "hash map" in text)
    ):
        return "two_sum_hash_map"
    if (
        "substring without repeating" in text
        or ("substring" in text and "repeating" in text and "window" in text)
        or "def length_of_longest_substring" in text
    ):
        return "longest_substring_sliding_window"
    if (
        "merge intervals" in text
        or ("interval" in text and "overlap" in text and "sort" in text)
        or "def merge_intervals" in text
    ):
        return "merge_intervals_sorting"
    if (
        "top k frequent" in text
        or "most frequent elements" in text
        or ("heap" in text and "frequency" in text)
        or "def top_k_frequent" in text
    ):
        return "top_k_frequent_heap"
    if (
        "rotated sorted array" in text
        or ("rotated" in text and "binary search" in text)
        or "def search_rotated" in text
    ):
        return "search_rotated_binary_search"
    if (
        "number of islands" in text
        or ("grid" in text and "island" in text and ("dfs" in text or "bfs" in text))
        or "def num_islands" in text
    ):
        return "number_of_islands_dfs"
    return ""


def _problem_solving_unique_pattern_target(target_count: int) -> int:
    return min(max(0, int(target_count or 0)), len(_PROBLEM_SOLVING_SCENARIOS_BY_ID))


def _accept_problem_pattern_signature(
    *,
    signature: str,
    seen_signatures: set[str],
    unique_target: int,
) -> bool:
    if not signature:
        return True
    if signature in seen_signatures and len(seen_signatures) < unique_target:
        return False
    seen_signatures.add(signature)
    return True


def _problem_solving_fallback_count_with_headroom(missing_count: int, problem_solving_mode: bool) -> int:
    missing = max(0, int(missing_count or 0))
    if missing <= 0:
        return 0
    if not problem_solving_mode:
        return missing
    scenario_count = max(1, len(_PROBLEM_SOLVING_SCENARIOS_BY_ID))
    return min(24, max(missing, missing * 3, missing + scenario_count))


def _build_prompt(
    topic_id: str,
    topic_title: str,
    doc_content: str,
    count: int = 5,
    difficulty: Optional[str] = None,
    level: Optional[str] = None,
    section_title: Optional[str] = None,
    section_content: Optional[str] = None,
    response_detail: str = "concise",
    preferred_language: str = "",
    requires_programming: bool = False,
    requested_total_count: Optional[int] = None,
    existing_questions: Optional[list[str]] = None,
    mcp_context: str = "",
) -> str:
    problem_solving_mode = is_problem_solving_topic(topic_id)
    target_level = _normalise_level(level)
    requested_total = max(1, int(requested_total_count or count or 1))
    existing_seed = _normalise_existing_questions(existing_questions)
    diff_clause = ""
    if difficulty:
        diff_clause = f' All questions should be "{difficulty}" difficulty.'

    if section_title and section_content:
        scope = f'the "{section_title}" section of "{topic_title}"'
        content = _clamp_content(section_content)
    else:
        scope = f'"{topic_title}"'
        content = _clamp_content(doc_content)

    detail_clause = (
        "Be concise and high-signal."
        if response_detail != "very_detailed"
        else "Be very detailed with layered explanation depth and concrete examples."
    )
    selected_language = (preferred_language or "").strip().lower()
    problem_solving_language = _problem_solving_language(selected_language)
    code_clause = ""
    if problem_solving_mode:
        code_clause = (
            f'Under `### Code`, include exactly one fenced "{problem_solving_language}" example with comments '
            "that explain each key step or block."
        )
    elif requires_programming and selected_language:
        code_clause = (
            f'Include one practical fenced code example in "{selected_language}" when code clarifies the answer.'
        )
    elif not requires_programming:
        code_clause = "Avoid code blocks unless code is explicitly required by the question."
    mcp_block = optional_context_block(
        "External context (optional, use only if relevant and factual)",
        mcp_context,
        2500,
    )

    role_clause = (
        "You are an expert algorithm interview coach and problem-solving educator."
        if problem_solving_mode
        else "You are an expert technical interviewer and educator."
    )
    problem_scope_rules = (
        [
            "Questions must be true problem-solving prompts (algorithmic/coding style), not generic theory prompts.",
            "Use a template-driven question structure: concrete input/output, explicit constraints, and interview-style reasoning expectations.",
            "Do not copy canonical LeetCode wording; adapt each question to the current topic scope and source content.",
            "Across generated items, vary the kinds of constraints, data-structure choices, and solution shapes.",
            "Every answer must include commented code in the selected language.",
            *_question_tutoring_arc_rules(True, problem_solving_language),
        ]
        if problem_solving_mode
        else [
            "Questions must cover conceptual + practical angles.",
            *_question_tutoring_arc_rules(False),
        ]
    )
    answer_shape = (
        "Markdown with exactly ### Problem, ### Solution Walkthrough, ### Complexity, and ### Code; include the required bold coaching labels inside those sections and exactly one fenced code block"
        if problem_solving_mode
        else "Markdown with bold sections **Answer:**, **Why it's right:**, **Interviewer-ready phrasing:**, **Common mistake:**, and **Self-check:**"
    )
    uniqueness_block = ""
    if existing_seed:
        existing_blob = "\n".join([f"- {q}" for q in existing_seed[:60]])
        uniqueness_block = (
            "\nAlready generated questions for this checkpoint. Do NOT repeat or rephrase these:\n"
            f"{existing_blob}\n"
        )
    prompt_contract = render_contract(
        schema_label="Return ONLY valid JSON in this shape",
        schema_block=f"""[
  {{
    "question": "Question text",
    "answer": "{answer_shape}",
    "difficulty": "easy|medium|hard"
  }}
]""",
        rules=[
            f"Return exactly {count} items; never return fewer.",
            "Questions must be standalone and non-duplicative.",
            "Questions must be NEW relative to already generated checkpoint questions listed above.",
            *problem_scope_rules,
            *_question_auxiliary_field_rules(problem_solving_mode),
            "Answers must be grounded in the provided documentation.",
            detail_clause,
            "Format answers as markdown, but keep structure adaptive.",
            "long answers must be split into short readable paragraphs with blank lines.",
            "use bullets only when listing steps/checklists/categories.",
            (
                "Use only the required headings for problem-solving answers."
                if problem_solving_mode
                else "Do not add extra headings to conceptual answers."
            ),
            "use tables only for direct comparisons/category matrices.",
            "If you use a table, output valid GFM table syntax.",
            "one row per line.",
            "include a separator row (e.g. `| --- | --- |`).",
            (
                "Problem Solving answers must always include the required fenced code block."
                if problem_solving_mode
                else "You may include fenced code blocks when code clarifies an implementation detail."
            ),
            code_clause if code_clause else "Use code examples only when they materially improve clarity.",
            "You may include fenced Mermaid diagrams when architecture or flows are better shown visually.",
            "If you include fences, always use explicit language tags (for example: ```python, ```mermaid).",
            "Because output must be valid JSON, escape newlines, quotes, and backslashes correctly inside string values.",
            "source_quote must be factual text from provided docs.",
            f'target_level must match "{target_level}" exactly.',
            "Do not wrap JSON with prose; return raw JSON only.",
            "Complexity must match target_level.",
            "junior: fundamentals, definitions, and straightforward tradeoffs.",
            "mid: implementation details, constraints, and moderate tradeoffs.",
            "senior: architecture, scaling, risk, and deep tradeoff decisions.",
            "Keep markdown compact and practical.",
        ],
    )
    return f"""{role_clause}

Given documentation about {scope}, generate exactly {count} interview-style questions with coach-style educational answers.{diff_clause}
Target candidate level: "{target_level}".
User requested total questions for this checkpoint: {requested_total}.
This call is generating {count} new questions to fill remaining slots.
{uniqueness_block}

{prompt_contract}

Optional fields (recommended when available): learning_objective, source_section, source_quote,
misconception_trap, reasoning_summary, target_level.

Documentation:
{content}{mcp_block}"""


def _build_quiz_prompt(
    topics_content: list[dict],
    count: int = 10,
    question_types: list[str] | None = None,
    difficulty: Optional[str] = None,
    level: Optional[str] = None,
    response_detail: str = "concise",
    preferred_language: str = "",
    mcp_context: str = "",
) -> str:
    target_level = _normalise_level(level)
    types = question_types or ["mcq", "true_false"]
    diff_clause = ""
    if difficulty:
        diff_clause = f' All questions should be "{difficulty}" difficulty.'

    if "mcq" in types and "true_false" in types:
        type_instructions = "Mix multiple-choice (exactly 4 options A/B/C/D) and true_false (exactly A=True, B=False)."
    elif "mcq" in types:
        type_instructions = "All questions must be MCQ with exactly 4 options A/B/C/D."
    else:
        type_instructions = "All questions must be true_false with exactly 2 options: A=True, B=False."

    detail_clause = (
        "Keep explanations concise and high-signal."
        if response_detail != "very_detailed"
        else "Provide very detailed explanations with practical depth."
    )
    docs_text = ""
    for tc in topics_content:
        topic_requires_programming = bool(tc.get("requires_programming"))
        topic_language = str(tc.get("preferred_language", "")).strip().lower()
        topic_detail = str(tc.get("response_detail", response_detail)).strip().lower()
        topic_notes = [f"detail={topic_detail or response_detail}"]
        if topic_requires_programming and topic_language:
            topic_notes.append(f"preferred_language={topic_language}")
        docs_text += (
            f"\n\n--- Topic: {tc['title']} (id: {tc['id']}) [{', '.join(topic_notes)}] ---\n"
            f"{tc['content'][:_MAX_TOPIC_CONTEXT]}"
        )
    mcp_block = optional_context_block(
        "External context (optional, use only if relevant and factual)",
        mcp_context,
        2500,
    )
    prompt_contract = render_contract(
        schema_block="""[
  {
    "question": "Question text",
    "type": "mcq|true_false",
    "choices": [{"label":"A","text":"..."}, {"label":"B","text":"..."}, ...],
    "correct_answer": "A|B|C|D",
    "explanation": "2-4 sentence explanation",
    "difficulty": "easy|medium|hard",
    "topic_id": "one of provided topic ids",
    "source_quote": "short direct quote from docs",
    "reasoning_summary": "1-2 sentence why answer is correct",
    "target_level": "junior|mid|senior"
  }
]""",
        rules=[
            f"Return exactly {count} items; never return fewer.",
            "topic_id must be one of the provided ids.",
            "correct_answer must match one choice label.",
            "Avoid trick ambiguity; one clearly correct answer.",
            "Distribute questions across topics as evenly as possible.",
            detail_clause,
            "Format explanations as markdown, but keep structure adaptive.",
            "long explanations must be split into short readable paragraphs with blank lines.",
            "use bullets only when listing steps/checklists/categories.",
            "use headings only when the explanation naturally has sections.",
            "use tables only for direct comparisons/category matrices.",
            "If you use a table, output valid GFM table syntax.",
            "one row per line.",
            "include a separator row (e.g. `| --- | --- |`).",
            "You may include fenced code blocks when code clarifies an implementation detail.",
            "Avoid unnecessary code blocks for non-programming topics.",
            (
                f'If code materially clarifies an explanation, prefer fenced "{preferred_language}" snippets.'
                if preferred_language
                else ""
            ),
            "You may include fenced Mermaid diagrams when architecture or flows are better shown visually.",
            "If you include fences, always use explicit language tags (for example: ```yaml, ```mermaid).",
            "Because output must be valid JSON, escape newlines, quotes, and backslashes correctly inside string values.",
            f'target_level must match "{target_level}" exactly.',
            "Complexity must match target_level.",
            "junior: direct recall and simple application.",
            "mid: applied reasoning with concrete constraints.",
            "senior: architecture-level reasoning and tradeoff depth.",
            "Keep markdown compact and practical.",
        ],
    )
    return f"""You are an expert technical quiz creator.

Create exactly {count} quiz questions from the documentation below.{diff_clause}
Target candidate level: "{target_level}".
{type_instructions}

{prompt_contract}

Documentation:
{docs_text[:_MAX_DOC_CONTEXT]}{mcp_block}"""


def _build_retry_prompt(
    base_prompt: str,
    missing_count: int,
    issues: str,
    existing_questions: list[str],
    hard_requirements: Optional[list[str]] = None,
    recovery_guidance: str = "",
) -> str:
    existing_blob = "\n".join([f"- {q}" for q in existing_questions[:50]])
    hard_requirements_block = ""
    if hard_requirements:
        hard_requirements_block = "Hard format requirements:\n" + "\n".join(
            [f"- {item}" for item in hard_requirements]
        )
    recovery_guidance_block = ""
    if recovery_guidance.strip():
        recovery_guidance_block = f"Recovery guidance:\n{recovery_guidance.strip()}\n"
    return f"""Your previous output did not satisfy the schema or quality constraints.

Missing items needed: {missing_count}
Validation issues:
{issues}

{hard_requirements_block}
{recovery_guidance_block}

Do not repeat any of these existing questions:
{existing_blob if existing_blob else "- (none)"}

Return ONLY a JSON array with exactly {missing_count} NEW valid items.

{base_prompt}
"""


def _parse_questions_json(raw: str) -> list[dict]:
    """Robustly extract JSON array from LLM response."""
    text = (raw or "").strip()
    if not text:
        return []

    def _extract_items(payload: Any) -> list[dict]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            nested = payload.get("questions") or payload.get("items")
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
            if payload.get("question") and payload.get("answer"):
                return [payload]
        return []

    def _try_load_json(candidate: str) -> list[dict]:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return []
        return _extract_items(parsed)

    def _strip_outer_code_fence(candidate: str) -> str:
        stripped = candidate.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            return stripped
        return "\n".join(lines[1:-1]).strip()

    def _parse_text_qa_pairs(candidate: str) -> list[dict]:
        cleaned = candidate.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = _strip_outer_code_fence(cleaned)

        q_pattern = re.compile(
            r"(?im)^\s*(?:\d+\s*[\).:\-]\s*)?(?:\*\*)?\s*(?:question|q)\s*\d*\s*[:\-]\s*"
        )
        a_pattern = re.compile(r"(?im)^\s*(?:\*\*)?\s*(?:answer|a)\s*\d*\s*[:\-]\s*")
        q_matches = list(q_pattern.finditer(cleaned))
        items: list[dict] = []

        if q_matches:
            for idx, q_match in enumerate(q_matches):
                block_end = q_matches[idx + 1].start() if idx + 1 < len(q_matches) else len(cleaned)
                block = cleaned[q_match.end() : block_end].strip()
                if not block:
                    continue
                answer_match = a_pattern.search(block)
                if answer_match:
                    question = block[: answer_match.start()].strip(" -*\t\n")
                    answer = block[answer_match.end() :].strip(" \t\n")
                else:
                    parts = [p.strip() for p in block.split("\n", 1)]
                    question = parts[0].strip(" -*\t")
                    answer = parts[1].strip() if len(parts) > 1 else ""
                if question and answer:
                    items.append({"question": question, "answer": answer})
            if items:
                return items

        numbered_pattern = re.compile(
            r"(?ims)^\s*(\d+)[\).:\-]\s*(.+?)(?:\n\s*(?:answer|a)\s*[:\-]\s*(.+?))(?=^\s*\d+[\).:\-]|\Z)"
        )
        for match in numbered_pattern.finditer(cleaned):
            question = match.group(2).strip(" -*\t\n")
            answer = match.group(3).strip()
            if question and answer:
                items.append({"question": question, "answer": answer})
        return items

    parsed = _try_load_json(text)
    if parsed:
        return parsed

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        parsed = _try_load_json(m.group(1))
        if parsed:
            return parsed

    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        start_match = re.search(r"[\[{]", text[cursor:])
        if not start_match:
            break
        start = cursor + start_match.start()
        try:
            decoded, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        parsed = _extract_items(decoded)
        if parsed:
            return parsed
        cursor = start + consumed

    parsed = _parse_text_qa_pairs(text)
    if parsed:
        return parsed

    return []


def _extract_content_fragments(content: str, *, limit: int = 48) -> list[str]:
    text = (content or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return []

    fragments: list[str] = []
    seen: set[str] = set()
    lines = text.splitlines()
    for line in lines:
        chunk = line.strip()
        if not chunk:
            continue
        if chunk.startswith("```"):
            continue
        chunk = re.sub(r"^#{1,6}\s*", "", chunk)
        chunk = re.sub(r"^[\-\*\d\.\)\s]+", "", chunk).strip()
        chunk = re.sub(r"\s+", " ", chunk)
        if len(chunk) < 6:
            continue
        if len(chunk) > 140:
            chunk = chunk[:140].rsplit(" ", 1)[0].strip() or chunk[:140]
        norm = _normalise_question(chunk)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        fragments.append(chunk.rstrip(":;,. "))
        if len(fragments) >= limit:
            break

    if not fragments:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in paragraphs[:limit]:
            cleaned = re.sub(r"\s+", " ", p)
            if len(cleaned) < 6:
                continue
            fragments.append(cleaned[:140].rsplit(" ", 1)[0].strip() or cleaned[:140])
            if len(fragments) >= limit:
                break
    return fragments


def _build_fallback_question_items(
    *,
    topic_id: str,
    topic_title: str,
    doc_content: str,
    count: int,
    difficulty: Optional[str],
    level: Optional[str],
    section_title: Optional[str] = None,
    section_content: Optional[str] = None,
    existing_questions: Optional[list[str]] = None,
    preferred_language: str = "",
    requires_programming: bool = False,
) -> list[dict[str, Any]]:
    remaining = max(0, int(count or 0))
    if remaining <= 0:
        return []

    problem_solving_mode = is_problem_solving_topic(topic_id)
    source_scope = section_title or topic_title or "this topic"
    source_content = section_content if (section_title and section_content) else doc_content
    fragments = _extract_content_fragments(source_content, limit=64)
    if not fragments:
        fragments = [source_scope, topic_title or "core concepts", "practical implementation"]

    existing_seed = _normalise_existing_questions(existing_questions)
    dedup: set[str] = {_normalise_question(q) for q in existing_seed}
    seen_questions: list[str] = [*existing_seed]
    difficulty_value = (difficulty or "medium").strip().lower()
    if difficulty_value not in _VALID_DIFFICULTIES:
        difficulty_value = "medium"
    selected_language = _problem_solving_language(preferred_language) if problem_solving_mode else ""
    scenario_ids = (
        _problem_solving_scenario_ids_for_language(selected_language)
        if problem_solving_mode
        else []
    )

    templates = [
        "What are the key design considerations for {focus} in {topic} from a {angle} perspective?",
        "How would you apply {focus} in a real implementation for {topic} while prioritizing {angle}?",
        "What tradeoffs should be evaluated when working with {focus} in {topic} regarding {angle}?",
        "How would you validate that {focus} is implemented correctly in {topic} with emphasis on {angle}?",
        "A production issue appears around {focus} in {topic}. How would you diagnose and fix it with focus on {angle}?",
        "How would you test and monitor {focus} in {topic} to maintain strong {angle} guarantees?",
        "If {focus} must scale in {topic}, what architecture changes would you make for better {angle}?",
        "When refactoring {focus} in {topic}, how would you reduce risk and preserve {angle} behavior?",
    ]
    angles = [
        "correctness",
        "reliability",
        "performance",
        "security",
        "maintainability",
        "scalability",
        "observability",
        "cost efficiency",
        "testing",
    ]

    items: list[dict[str, Any]] = []
    idx = 0
    max_rounds = max(remaining * 12, 24)
    while len(items) < remaining and idx < max_rounds:
        focus = fragments[idx % len(fragments)]
        angle = angles[(idx // max(1, len(fragments))) % len(angles)]
        scenario_id = _PROBLEM_SOLVING_DEFAULT_SCENARIO_ID
        if problem_solving_mode:
            scenario_id = scenario_ids[idx % len(scenario_ids)]
            question = _problem_solving_question_seed(
                focus,
                source_scope,
                scenario_id=scenario_id,
            )["question"]
        else:
            template = templates[idx % len(templates)]
            question = template.format(
                focus=focus,
                topic=topic_title or "this topic",
                angle=angle,
            ).strip()
        q_norm = _normalise_question(question)
        idx += 1
        if not q_norm or q_norm in dedup or _is_near_duplicate_question(question, seen_questions):
            continue

        if problem_solving_mode:
            answer, reasoning_summary = _build_problem_solving_fallback_answer(
                focus=focus,
                source_scope=source_scope,
                language=selected_language,
                scenario_id=scenario_id,
            )
        else:
            answer, reasoning_summary = _build_conceptual_fallback_answer(
                focus=focus,
                source_scope=source_scope,
                angle=angle,
            )
        item: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "difficulty": difficulty_value,
            "learning_objective": _fallback_learning_objective(
                focus=focus,
                source_scope=source_scope,
                problem_solving_mode=problem_solving_mode,
            ),
            "source_section": source_scope,
            "source_quote": focus,
            "misconception_trap": _fallback_misconception_trap(
                focus=focus,
                problem_solving_mode=problem_solving_mode,
            ),
            "reasoning_summary": reasoning_summary,
            "target_level": _normalise_level(level),
            "topic_id": topic_id,
        }
        valid, _issue = _validate_question_item(
            item,
            topic_id,
            difficulty,
            level,
            preferred_language=selected_language,
            requires_programming=requires_programming,
        )
        if not valid:
            continue
        dedup.add(q_norm)
        seen_questions.append(question)
        items.append(item)

    return items[:remaining]


def _validate_question_item(
    item: dict[str, Any],
    topic_id: str,
    difficulty: Optional[str],
    level: Optional[str],
    preferred_language: str = "",
    requires_programming: bool = False,
) -> tuple[bool, str]:
    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()
    if not question:
        return False, "missing_or_empty_question"
    if not answer:
        return False, "missing_or_empty_answer"

    diff = str(item.get("difficulty", "")).strip().lower()
    if difficulty:
        diff = str(difficulty).strip().lower()
    if diff not in _VALID_DIFFICULTIES:
        diff = "medium"

    target_level = _normalise_level(level)
    problem_solving_mode = is_problem_solving_topic(topic_id)
    item["question"] = question
    item["answer"] = answer
    item["learning_objective"] = (
        str(item.get("learning_objective", "")).strip()
        or _default_learning_objective(problem_solving_mode)
    )
    item["source_section"] = str(item.get("source_section", "")).strip() or "Topic section"
    item["source_quote"] = (
        str(item.get("source_quote", "")).strip()
        or "Generated from the current topic content."
    )
    item["misconception_trap"] = (
        str(item.get("misconception_trap", "")).strip()
        or _default_misconception_trap(problem_solving_mode)
    )
    item["reasoning_summary"] = (
        str(item.get("reasoning_summary", "")).strip()
        or _default_reasoning_summary(problem_solving_mode)
    )
    item["target_level"] = target_level
    item["topic_id"] = topic_id
    item["difficulty"] = diff
    if problem_solving_mode:
        valid_answer, issue = _validate_problem_solving_answer(
            answer,
            preferred_language,
        )
        if not valid_answer:
            return False, issue
    return True, ""


def _validate_quiz_item(
    item: dict[str, Any],
    allowed_topics: set[str],
    allowed_types: set[str],
    difficulty: Optional[str],
    level: Optional[str],
) -> tuple[bool, str]:
    required = [
        "question",
        "type",
        "choices",
        "correct_answer",
        "explanation",
        "topic_id",
        "source_quote",
        "reasoning_summary",
        "target_level",
    ]
    for key in required:
        if key not in item:
            return False, f"missing_{key}"

    q_text = item.get("question")
    q_type = str(item.get("type", "")).lower()
    choices = item.get("choices")
    correct = str(item.get("correct_answer", "")).strip()
    explanation = item.get("explanation")
    topic_id = str(item.get("topic_id", "")).strip()
    source_quote = item.get("source_quote")
    reasoning_summary = item.get("reasoning_summary")
    target_level = str(item.get("target_level", "")).strip().lower()

    if not isinstance(q_text, str) or len(q_text.strip()) < 8:
        return False, "invalid_question"
    if q_type not in {"mcq", "true_false"}:
        return False, "invalid_type"
    if q_type not in allowed_types:
        return False, "disallowed_type"
    if topic_id not in allowed_topics:
        return False, "invalid_topic_id"
    if not isinstance(explanation, str) or len(explanation.strip()) < 15:
        return False, "invalid_explanation"
    if not isinstance(source_quote, str) or len(source_quote.strip()) < 8:
        return False, "invalid_source_quote"
    if not isinstance(reasoning_summary, str) or len(reasoning_summary.strip()) < 8:
        return False, "invalid_reasoning_summary"
    if target_level not in _VALID_LEVELS:
        return False, "invalid_target_level"
    if _normalise_level(level) != target_level:
        return False, "level_mismatch"

    if not isinstance(choices, list):
        return False, "invalid_choices_type"
    labels = []
    for c in choices:
        if not isinstance(c, dict):
            return False, "invalid_choice_item"
        label = str(c.get("label", "")).strip()
        text = c.get("text")
        if label not in {"A", "B", "C", "D"}:
            return False, "invalid_choice_label"
        if not isinstance(text, str) or not text.strip():
            return False, "invalid_choice_text"
        labels.append(label)

    if q_type == "mcq":
        if labels != ["A", "B", "C", "D"]:
            return False, "mcq_labels_must_be_abcd"
        if len(choices) != 4:
            return False, "mcq_must_have_4_choices"
        if correct not in {"A", "B", "C", "D"}:
            return False, "invalid_correct_answer_mcq"
    else:
        if len(choices) != 2:
            return False, "true_false_must_have_2_choices"
        if labels != ["A", "B"]:
            return False, "true_false_labels_must_be_ab"
        true_false_texts = [str(choices[0].get("text", "")).strip().lower(), str(choices[1].get("text", "")).strip().lower()]
        if true_false_texts != ["true", "false"]:
            return False, "true_false_text_must_be_true_false"
        if correct not in {"A", "B"}:
            return False, "invalid_correct_answer_true_false"

    diff = str(item.get("difficulty", "medium")).lower()
    if diff not in _VALID_DIFFICULTIES:
        return False, "invalid_difficulty"
    if difficulty and diff != difficulty:
        return False, "difficulty_mismatch"
    item["difficulty"] = diff
    item["type"] = q_type
    return True, ""


def _build_fallback_quiz_items(
    *,
    topics_content: list[dict],
    count: int,
    question_types: list[str] | None,
    difficulty: Optional[str],
    level: Optional[str],
    existing_questions: list[str],
) -> list[dict[str, Any]]:
    if count <= 0 or not topics_content:
        return []

    allowed_topics = {str(tc.get("id", "")).strip() for tc in topics_content if str(tc.get("id", "")).strip()}
    allowed_types = set(question_types or ["mcq", "true_false"])
    if not allowed_types:
        allowed_types = {"mcq", "true_false"}
    target_level = _normalise_level(level)
    diff = str(difficulty or "medium").strip().lower()
    if diff not in _VALID_DIFFICULTIES:
        diff = "medium"

    out: list[dict[str, Any]] = []
    seen_norm = {_normalise_question(q) for q in _normalise_existing_questions(existing_questions)}
    guard = 0
    while len(out) < count and guard < (count * 12):
        idx = len(out) + guard
        topic = topics_content[idx % len(topics_content)]
        topic_id = str(topic.get("id", "")).strip()
        if not topic_id:
            guard += 1
            continue
        topic_title = str(topic.get("title", topic_id)).strip() or topic_id
        source_scope = str(topic.get("content", "")).strip()
        source_quote = re.sub(r"\s+", " ", source_scope)[:180].strip() or (
            f"{topic_title} interview fundamentals and tradeoffs."
        )
        if len(source_quote) < 8:
            source_quote = f"{topic_title} interview fundamentals and tradeoffs."

        use_mcq = "mcq" in allowed_types and ("true_false" not in allowed_types or idx % 2 == 0)
        if use_mcq:
            stems = [
                "For {topic_title}, which approach best balances correctness, maintainability, and interview communication under constraints?",
                "In a {topic_title} interview scenario, which response structure shows the strongest engineering judgment?",
                "When discussing {topic_title}, which strategy most clearly demonstrates tradeoff-driven thinking?",
                "For {topic_title}, which interviewing approach is most likely to produce a robust and explainable solution?",
            ]
            question = stems[idx % len(stems)].format(topic_title=topic_title)
            choices = [
                {"label": "A", "text": "Clarify constraints, choose a justified approach, and explain tradeoffs."},
                {"label": "B", "text": "Start coding immediately and defer reasoning until the end."},
                {"label": "C", "text": "Optimize micro-details before validating core requirements."},
                {"label": "D", "text": "Assume defaults and skip edge-case discussion to save time."},
            ]
            correct = "A"
            q_type = "mcq"
        else:
            stems = [
                "True or false for {topic_title}: strong interview answers should explicitly state assumptions and tradeoffs before implementation details.",
                "For {topic_title}, true or false: candidates should justify constraints first, then explain implementation choices.",
                "True or false in a {topic_title} interview: skipping assumptions weakens the quality of technical reasoning.",
                "For {topic_title}, true or false: discussing tradeoffs early usually improves answer clarity and credibility.",
            ]
            question = stems[idx % len(stems)].format(topic_title=topic_title)
            choices = [
                {"label": "A", "text": "True"},
                {"label": "B", "text": "False"},
            ]
            correct = "A"
            q_type = "true_false"

        q_norm = _normalise_question(question)
        if not q_norm or q_norm in seen_norm:
            guard += 1
            continue
        seen_norm.add(q_norm)
        item = {
            "question": question,
            "type": q_type,
            "choices": choices,
            "correct_answer": correct,
            "explanation": (
                "Strong responses should make reasoning explicit, connect choices to constraints, "
                "and communicate tradeoffs with clear structure."
            ),
            "difficulty": diff,
            "topic_id": topic_id,
            "source_quote": source_quote,
            "reasoning_summary": "Anchor decisions in constraints, then justify tradeoffs clearly.",
            "target_level": target_level,
        }
        valid, _issue = _validate_quiz_item(
            item,
            allowed_topics=allowed_topics,
            allowed_types=allowed_types,
            difficulty=difficulty,
            level=level,
        )
        if valid:
            out.append(item)
        guard += 1

    return out[:count]


async def _collect_with_retries(
    *,
    llm: LLMClient,
    base_prompt: str,
    llm_config: Optional[LLMConfigRequest],
    user_identity: Optional[dict] = None,
    target_count: int,
    validator: Callable[[dict[str, Any]], tuple[bool, str]],
    existing_questions: Optional[list[str]] = None,
    hard_requirements: Optional[list[str]] = None,
    retry_guidance: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing_seed = _normalise_existing_questions(existing_questions)
    dedup: set[str] = {_normalise_question(q) for q in existing_seed}
    seen_questions: list[str] = [*existing_seed]
    valid_items: list[dict[str, Any]] = []
    malformed_items = 0
    retries_used = 0
    issue_counter: Counter[str] = Counter()
    metadata: dict[str, Any] = {}
    prompt = base_prompt
    last_error = ""

    for _ in range(_attempt_budget(target_count)):
        result = await llm.completion(
            prompt,
            llm_config,
            user_identity=user_identity,
        )
        raise_if_policy_blocked_result(result)
        metadata = result.get("metadata", {})
        if not result.get("success"):
            last_error = result.get("error", "unknown_error")
            retries_used += 1
            prompt = _build_retry_prompt(
                base_prompt=base_prompt,
                missing_count=max(1, target_count - len(valid_items)),
                issues=f"transport_or_provider_error: {last_error}",
                existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
                hard_requirements=hard_requirements,
                recovery_guidance=retry_guidance,
            )
            continue

        parsed = _parse_questions_json(result.get("analysis", ""))
        if not parsed:
            retries_used += 1
            issue_counter["json_parse_failed"] += 1
            prompt = _build_retry_prompt(
                base_prompt=base_prompt,
                missing_count=max(1, target_count - len(valid_items)),
                issues="json_parse_failed",
                existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
                hard_requirements=hard_requirements,
                recovery_guidance=retry_guidance,
            )
            continue

        for item in parsed:
            if not isinstance(item, dict):
                malformed_items += 1
                issue_counter["non_dict_item"] += 1
                continue

            is_valid, issue = validator(item)
            if not is_valid:
                malformed_items += 1
                issue_counter[issue] += 1
                continue

            q_norm = _normalise_question(str(item.get("question", "")))
            question_text = str(item.get("question", "")).strip()
            if not q_norm or q_norm in dedup or _is_near_duplicate_question(question_text, seen_questions):
                malformed_items += 1
                issue_counter["duplicate_question"] += 1
                continue
            dedup.add(q_norm)
            seen_questions.append(question_text)
            valid_items.append(item)
            if len(valid_items) >= target_count:
                break

        if len(valid_items) >= target_count:
            break

        retries_used += 1
        top_issues = ", ".join([f"{k}:{v}" for k, v in issue_counter.most_common(5)]) or "insufficient_valid_items"
        prompt = _build_retry_prompt(
            base_prompt=base_prompt,
            missing_count=target_count - len(valid_items),
            issues=top_issues,
            existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
            hard_requirements=hard_requirements,
            recovery_guidance=retry_guidance,
        )

    if len(valid_items) < target_count:
        # Final pass: request one item at a time to reduce duplicate/schema drift.
        for _ in range(_recovery_budget(target_count - len(valid_items))):
            if len(valid_items) >= target_count:
                break

            single_prompt = _build_retry_prompt(
                base_prompt=base_prompt,
                missing_count=1,
                issues="final_recovery_fill_missing_items",
                existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
                hard_requirements=hard_requirements,
                recovery_guidance=retry_guidance,
            )
            result = await llm.completion(
                single_prompt,
                llm_config,
                user_identity=user_identity,
            )
            raise_if_policy_blocked_result(result)
            metadata = result.get("metadata", {})
            if not result.get("success"):
                retries_used += 1
                issue_counter["transport_or_provider_error"] += 1
                continue

            parsed = _parse_questions_json(result.get("analysis", ""))
            if not parsed:
                retries_used += 1
                malformed_items += 1
                issue_counter["json_parse_failed"] += 1
                continue

            added = False
            for item in parsed:
                if not isinstance(item, dict):
                    malformed_items += 1
                    issue_counter["non_dict_item"] += 1
                    continue
                is_valid, issue = validator(item)
                if not is_valid:
                    malformed_items += 1
                    issue_counter[issue] += 1
                    continue

                q_norm = _normalise_question(str(item.get("question", "")))
                question_text = str(item.get("question", "")).strip()
                if not q_norm or q_norm in dedup or _is_near_duplicate_question(question_text, seen_questions):
                    malformed_items += 1
                    issue_counter["duplicate_question"] += 1
                    continue

                dedup.add(q_norm)
                seen_questions.append(question_text)
                valid_items.append(item)
                added = True
                break

            if not added:
                retries_used += 1

    if len(valid_items) < target_count:
        logger.warning(
            "Generation produced %d/%d valid items after retries=%d issues=%s last_error=%s",
            len(valid_items),
            target_count,
            retries_used,
            dict(issue_counter),
            last_error,
        )

    stats = {
        "metadata": metadata,
        "retries_used": retries_used,
        "malformed_items_dropped": malformed_items,
    }
    return valid_items[:target_count], stats


class QuestionGenerator:
    """Uses LiteLLM to generate interview Q&A and quizzes."""

    def __init__(self, llm_client: LLMClient, mcp_gateway: MCPGateway | None = None):
        self.llm = llm_client
        self.mcp = mcp_gateway

    async def _mcp_context_for_flow(
        self,
        *,
        flow: str,
        query: str,
        topic_id: str = "",
        topic_title: str = "",
    ) -> str:
        if self.mcp is None:
            return ""
        return await self.mcp.gather_context(
            flow=flow,
            query=query,
            topic_id=topic_id,
            topic_title=topic_title,
        )

    @staticmethod
    def _to_question_answer_v2(topic_id: str, item: dict[str, Any]) -> QuestionAnswerV2:
        return QuestionAnswerV2(
            question_id=_question_id(topic_id, item["question"]),
            topic_id=topic_id,
            question=item["question"].strip(),
            answer=format_markdown_readable(item["answer"].strip()),
            difficulty=item["difficulty"],
            learning_objective=item["learning_objective"].strip(),
            source_section=item["source_section"].strip(),
            source_quote=item["source_quote"].strip(),
            misconception_trap=item["misconception_trap"].strip(),
            reasoning_summary=item["reasoning_summary"].strip(),
        )

    @staticmethod
    def _to_quiz_question_v2(item: dict[str, Any]) -> QuizQuestionV2:
        topic_id = str(item.get("topic_id", "")).strip()
        qid = _question_id(topic_id or "quiz", str(item.get("question", "")))
        choices = [QuizChoice(label=c["label"], text=c["text"]) for c in item["choices"]]
        return QuizQuestionV2(
            question_id=qid,
            question=str(item.get("question", "")).strip(),
            type=QuizQuestionType(str(item.get("type", "mcq")).strip().lower()),
            choices=choices,
            correct_answer=str(item.get("correct_answer", "")).strip(),
            explanation=format_markdown_readable(str(item.get("explanation", "")).strip()),
            difficulty=str(item.get("difficulty", "medium")).strip().lower() or "medium",
            topic_id=topic_id,
            source_quote=str(item.get("source_quote", "")).strip(),
            reasoning_summary=str(item.get("reasoning_summary", "")).strip(),
        )

    @staticmethod
    def _policy_error_payload(exc: Exception) -> tuple[str, str]:
        if isinstance(exc, LLMServiceApprovalRequiredError):
            return APPROVAL_REQUIRED_CODE, APPROVAL_REQUIRED_MESSAGE
        if isinstance(exc, StudyAppLLMNotAssignedError):
            return STUDY_APP_NOT_ASSIGNED_CODE, STUDY_APP_NOT_ASSIGNED_MESSAGE
        if isinstance(exc, PersonalCredentialRequiredError):
            return PERSONAL_CREDENTIAL_REQUIRED_CODE, PERSONAL_CREDENTIAL_REQUIRED_MESSAGE
        return "generation_failed", str(exc).strip() or "Question generation failed"

    async def generate_v2_stream(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 5,
        requested_total_count: Optional[int] = None,
        existing_questions: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
        response_detail: str = "concise",
        preferred_language: str = "",
        requires_programming: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        target_count = max(0, int(count))
        requested_total = max(1, int(requested_total_count or count or 1))
        seed_existing = _normalise_existing_questions(existing_questions)
        problem_solving_mode = is_problem_solving_topic(topic_id)
        selected_language = _problem_solving_language(preferred_language) if problem_solving_mode else (
            (preferred_language or "").strip().lower()
        )
        hard_requirements = (
            _problem_solving_required_markdown(selected_language) if problem_solving_mode else None
        )
        retry_guidance = _question_retry_guidance(problem_solving_mode)
        mcp_context = await self._mcp_context_for_flow(
            flow="questions",
            query=f"{topic_title} {section_title or ''} interview questions and practical examples",
            topic_id=topic_id,
            topic_title=topic_title,
        )
        dedup_norm: set[str] = {_normalise_question(q) for q in seed_existing}
        generated_question_texts: list[str] = [*seed_existing]
        problem_pattern_signatures: set[str] = set()
        max_unique_problem_patterns = (
            _problem_solving_unique_pattern_target(target_count)
            if problem_solving_mode
            else 0
        )

        async def _generate_one(*, include_section: bool, relaxed: bool = False) -> dict[str, Any]:
            retries_used = 0
            malformed_dropped = 0
            metadata: dict[str, Any] = {}
            issue_counter: Counter[str] = Counter()
            last_error = ""
            base_prompt = _build_prompt(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=1,
                difficulty=None if relaxed else difficulty,
                level=level,
                section_title=section_title if include_section else None,
                section_content=section_content if include_section else None,
                response_detail=response_detail,
                preferred_language=selected_language if (problem_solving_mode or not relaxed) else "",
                requires_programming=requires_programming if (problem_solving_mode or not relaxed) else False,
                requested_total_count=requested_total,
                existing_questions=generated_question_texts,
                mcp_context=mcp_context,
            )
            prompt = base_prompt

            try:
                for _ in range(_attempt_budget(1)):
                    result = await self.llm.completion(
                        prompt,
                        llm_config,
                        user_identity=user_identity,
                    )
                    raise_if_policy_blocked_result(result)
                    metadata = result.get("metadata", {})

                    if not result.get("success"):
                        retries_used += 1
                        last_error = str(result.get("error", "unknown_error"))
                        issue_counter["transport_or_provider_error"] += 1
                        prompt = _build_retry_prompt(
                            base_prompt=base_prompt,
                            missing_count=1,
                            issues=f"transport_or_provider_error: {last_error}",
                            existing_questions=generated_question_texts,
                            hard_requirements=hard_requirements,
                            recovery_guidance=retry_guidance,
                        )
                        continue

                    parsed = _parse_questions_json(result.get("analysis", ""))
                    if not parsed:
                        retries_used += 1
                        malformed_dropped += 1
                        issue_counter["json_parse_failed"] += 1
                        prompt = _build_retry_prompt(
                            base_prompt=base_prompt,
                            missing_count=1,
                            issues="json_parse_failed",
                            existing_questions=generated_question_texts,
                            hard_requirements=hard_requirements,
                            recovery_guidance=retry_guidance,
                        )
                        continue

                    selected_item: dict[str, Any] | None = None
                    for item in parsed:
                        if not isinstance(item, dict):
                            malformed_dropped += 1
                            issue_counter["non_dict_item"] += 1
                            continue

                        valid, issue = _validate_question_item(
                            item,
                            topic_id,
                            difficulty,
                            level,
                            preferred_language=selected_language,
                            requires_programming=requires_programming,
                        )
                        if not valid:
                            malformed_dropped += 1
                            issue_counter[issue] += 1
                            continue

                        q_text = str(item.get("question", "")).strip()
                        q_norm = _normalise_question(q_text)
                        if not q_norm:
                            malformed_dropped += 1
                            issue_counter["invalid_question"] += 1
                            continue

                        if q_norm in dedup_norm or _is_near_duplicate_question(q_text, generated_question_texts):
                            malformed_dropped += 1
                            issue_counter["duplicate_question"] += 1
                            continue

                        if problem_solving_mode:
                            signature = _problem_solving_pattern_signature(
                                q_text,
                                str(item.get("answer", "")),
                            )
                            if not _accept_problem_pattern_signature(
                                signature=signature,
                                seen_signatures=problem_pattern_signatures,
                                unique_target=max_unique_problem_patterns,
                            ):
                                malformed_dropped += 1
                                issue_counter["duplicate_problem_pattern"] += 1
                                continue

                        dedup_norm.add(q_norm)
                        generated_question_texts.append(q_text)
                        selected_item = item
                        break

                    if selected_item is not None:
                        qa = self._to_question_answer_v2(topic_id, selected_item)
                        return {
                            "question": qa.model_dump(),
                            "metadata": metadata,
                            "retries_used": retries_used,
                            "malformed_items_dropped": malformed_dropped,
                        }

                    retries_used += 1
                    top_issues = (
                        ", ".join(f"{k}:{v}" for k, v in issue_counter.most_common(5))
                        or "insufficient_valid_items"
                    )
                    prompt = _build_retry_prompt(
                        base_prompt=base_prompt,
                        missing_count=1,
                        issues=top_issues,
                        existing_questions=generated_question_texts,
                        hard_requirements=hard_requirements,
                        recovery_guidance=retry_guidance,
                    )

                if issue_counter:
                    logger.warning(
                        "Streaming slot failed to produce valid question after retries=%d issues=%s last_error=%s",
                        retries_used,
                        dict(issue_counter),
                        last_error,
                    )
                return {
                    "question": None,
                    "metadata": metadata,
                    "retries_used": retries_used,
                    "malformed_items_dropped": malformed_dropped,
                }
            except (
                LLMServiceApprovalRequiredError,
                StudyAppLLMNotAssignedError,
                PersonalCredentialRequiredError,
            ) as exc:
                code, message = self._policy_error_payload(exc)
                return {
                    "error": {"code": code, "message": message},
                    "metadata": metadata,
                    "retries_used": retries_used,
                    "malformed_items_dropped": malformed_dropped,
                }
            except Exception as exc:
                logger.exception("Streaming question generation failed: %s", exc)
                return {
                    "error": {
                        "code": "generation_failed",
                        "message": str(exc).strip() or "Question generation failed",
                    },
                    "metadata": metadata,
                    "retries_used": retries_used,
                    "malformed_items_dropped": malformed_dropped,
                }

        yield {
            "type": "start",
            "topic_id": topic_id,
            "topic_title": topic_title,
            "target_count": target_count,
        }

        if target_count == 0:
            yield {
                "type": "done",
                "topic_id": topic_id,
                "topic_title": topic_title,
                "generated_count": 0,
                "provider_used": "",
                "model_used": "",
                "retries_used": 0,
                "malformed_items_dropped": 0,
            }
            return

        provider_used = ""
        model_used = ""
        retries_used = 0
        malformed_items_dropped = 0
        generated_count = 0

        section_first = bool(section_title and section_content)
        phase_modes: list[tuple[bool, bool]] = (
            [(True, False), (False, False)]
            if section_first
            else [(False, False)]
        )
        if not problem_solving_mode:
            phase_modes = (
                [(True, False), (False, False), (False, True)]
                if section_first
                else [(False, False), (False, True)]
            )
        for include_section, relaxed in phase_modes:
            while generated_count < target_count:
                result = await _generate_one(include_section=include_section, relaxed=relaxed)
                retries_used += int(result.get("retries_used", 0))
                malformed_items_dropped += int(result.get("malformed_items_dropped", 0))
                metadata = result.get("metadata", {}) or {}
                if metadata.get("provider"):
                    provider_used = str(metadata.get("provider"))
                if metadata.get("model"):
                    model_used = str(metadata.get("model"))

                error = result.get("error")
                if error:
                    yield {
                        "type": "error",
                        "code": str(error.get("code", "generation_failed")),
                        "message": str(error.get("message", "Question generation failed")),
                    }
                    return

                question = result.get("question")
                if not isinstance(question, dict):
                    break

                generated_count += 1
                yield {
                    "type": "question",
                    "question": question,
                }

        if generated_count < target_count:
            for _ in range(_recovery_budget(target_count - generated_count)):
                if generated_count >= target_count:
                    break

                result = await _generate_one(
                    include_section=False,
                    relaxed=False if problem_solving_mode else True,
                )
                retries_used += int(result.get("retries_used", 0))
                malformed_items_dropped += int(result.get("malformed_items_dropped", 0))
                metadata = result.get("metadata", {}) or {}
                if metadata.get("provider"):
                    provider_used = str(metadata.get("provider"))
                if metadata.get("model"):
                    model_used = str(metadata.get("model"))

                error = result.get("error")
                if error:
                    yield {
                        "type": "error",
                        "code": str(error.get("code", "generation_failed")),
                        "message": str(error.get("message", "Question generation failed")),
                    }
                    return

                question = result.get("question")
                if not isinstance(question, dict):
                    continue

                generated_count += 1
                yield {
                    "type": "question",
                    "question": question,
                }

        if generated_count < target_count:
            missing_count = target_count - generated_count
            fallback_items = _build_fallback_question_items(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=_problem_solving_fallback_count_with_headroom(
                    missing_count,
                    problem_solving_mode,
                ),
                difficulty=difficulty,
                level=level,
                section_title=None,
                section_content=None,
                existing_questions=generated_question_texts,
                preferred_language=selected_language,
                requires_programming=requires_programming,
            )
            for item in fallback_items:
                q_text = str(item.get("question", "")).strip()
                q_norm = _normalise_question(q_text)
                if not q_norm or q_norm in dedup_norm or _is_near_duplicate_question(q_text, generated_question_texts):
                    continue
                if problem_solving_mode:
                    signature = _problem_solving_pattern_signature(
                        q_text,
                        str(item.get("answer", "")),
                    )
                    if not _accept_problem_pattern_signature(
                        signature=signature,
                        seen_signatures=problem_pattern_signatures,
                        unique_target=max_unique_problem_patterns,
                    ):
                        continue
                dedup_norm.add(q_norm)
                generated_question_texts.append(q_text)
                qa = self._to_question_answer_v2(topic_id, item)
                generated_count += 1
                yield {
                    "type": "question",
                    "question": qa.model_dump(),
                }
                if generated_count >= target_count:
                    break

        yield {
            "type": "done",
            "topic_id": topic_id,
            "topic_title": topic_title,
            "generated_count": generated_count,
            "provider_used": provider_used,
            "model_used": model_used,
            "retries_used": retries_used,
            "malformed_items_dropped": malformed_items_dropped,
        }

    async def generate(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 5,
        requested_total_count: Optional[int] = None,
        existing_questions: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
        response_detail: str = "concise",
        preferred_language: str = "",
        requires_programming: bool = False,
    ) -> GenerateQuestionsResponse:
        problem_solving_mode = is_problem_solving_topic(topic_id)
        selected_language = _problem_solving_language(preferred_language) if problem_solving_mode else (
            (preferred_language or "").strip().lower()
        )
        v2 = await self.generate_v2(
            topic_id=topic_id,
            topic_title=topic_title,
            doc_content=doc_content,
            count=count,
            requested_total_count=requested_total_count,
            existing_questions=existing_questions,
            difficulty=difficulty,
            level=level,
            llm_config=llm_config,
            user_identity=user_identity,
            section_title=section_title,
            section_content=section_content,
            response_detail=response_detail,
            preferred_language=selected_language,
            requires_programming=requires_programming,
        )
        target_count = max(1, int(count or 1))
        questions = [
            QuestionAnswer(question=q.question, answer=q.answer, difficulty=q.difficulty)
            for q in v2.questions
        ]
        problem_pattern_signatures: set[str] = set()
        max_unique_problem_patterns = (
            _problem_solving_unique_pattern_target(target_count)
            if problem_solving_mode
            else 0
        )
        if problem_solving_mode:
            for q in questions:
                signature = _problem_solving_pattern_signature(q.question, q.answer)
                if signature:
                    problem_pattern_signatures.add(signature)
        if len(questions) < target_count:
            missing_count = target_count - len(questions)
            fallback_items = _build_fallback_question_items(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=_problem_solving_fallback_count_with_headroom(
                    missing_count,
                    problem_solving_mode,
                ),
                difficulty=difficulty,
                level=level,
                section_title=section_title,
                section_content=section_content,
                existing_questions=[
                    *(_normalise_existing_questions(existing_questions)),
                    *[q.question for q in questions],
                ],
                preferred_language=selected_language,
                requires_programming=requires_programming,
            )
            for item in fallback_items:
                if problem_solving_mode:
                    signature = _problem_solving_pattern_signature(
                        str(item.get("question", "")).strip(),
                        str(item.get("answer", "")),
                    )
                    if not _accept_problem_pattern_signature(
                        signature=signature,
                        seen_signatures=problem_pattern_signatures,
                        unique_target=max_unique_problem_patterns,
                    ):
                        continue
                questions.append(
                    QuestionAnswer(
                        question=str(item.get("question", "")).strip(),
                        answer=str(item.get("answer", "")).strip(),
                        difficulty=str(item.get("difficulty", "medium")).strip().lower() or "medium",
                    )
                )
                if len(questions) >= target_count:
                    break
        return GenerateQuestionsResponse(
            topic_id=topic_id,
            topic_title=topic_title,
            questions=questions[:target_count],
            provider_used=v2.provider_used,
            model_used=v2.model_used,
        )

    async def generate_v2(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 5,
        requested_total_count: Optional[int] = None,
        existing_questions: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
        response_detail: str = "concise",
        preferred_language: str = "",
        requires_programming: bool = False,
    ) -> GenerateQuestionsV2Response:
        target_count = max(1, int(count or 1))
        problem_solving_mode = is_problem_solving_topic(topic_id)
        selected_language = _problem_solving_language(preferred_language) if problem_solving_mode else (
            (preferred_language or "").strip().lower()
        )
        retry_guidance = _question_retry_guidance(problem_solving_mode)
        mcp_context = await self._mcp_context_for_flow(
            flow="questions",
            query=f"{topic_title} {section_title or ''} interview questions and practical examples",
            topic_id=topic_id,
            topic_title=topic_title,
        )
        validator = lambda item: _validate_question_item(
            item,
            topic_id,
            difficulty,
            level,
            preferred_language=selected_language,
            requires_programming=requires_programming,
        )
        provider_used = ""
        model_used = ""
        retries_used = 0
        malformed_items_dropped = 0
        raw_items: list[dict[str, Any]] = []
        existing_seed = _normalise_existing_questions(existing_questions)
        dedup_norm: set[str] = {_normalise_question(q) for q in existing_seed}
        seen_questions: list[str] = [*existing_seed]
        problem_pattern_signatures: set[str] = set()
        max_unique_problem_patterns = (
            _problem_solving_unique_pattern_target(target_count)
            if problem_solving_mode
            else 0
        )

        pass_modes: list[dict[str, Any]] = [
            {
                "section_title": section_title,
                "section_content": section_content,
                "difficulty": difficulty,
                "preferred_language": selected_language,
                "requires_programming": requires_programming,
            }
        ]
        if section_title and section_content:
            pass_modes.append(
                {
                    "section_title": None,
                    "section_content": None,
                    "difficulty": difficulty,
                    "preferred_language": selected_language,
                    "requires_programming": requires_programming,
                }
            )
        if not problem_solving_mode:
            pass_modes.append(
                {
                    "section_title": None,
                    "section_content": None,
                    "difficulty": None,
                    "preferred_language": "",
                    "requires_programming": False,
                }
            )

        for mode in pass_modes:
            if len(raw_items) >= target_count:
                break
            remaining = target_count - len(raw_items)
            existing_for_pass = [
                *existing_seed,
                *[str(item.get("question", "")) for item in raw_items],
            ]
            prompt = _build_prompt(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=remaining,
                requested_total_count=requested_total_count,
                existing_questions=existing_for_pass,
                difficulty=mode["difficulty"],
                level=level,
                section_title=mode["section_title"],
                section_content=mode["section_content"],
                response_detail=response_detail,
                preferred_language=mode["preferred_language"],
                requires_programming=mode["requires_programming"],
                mcp_context=mcp_context,
            )
            pass_items, pass_stats = await _collect_with_retries(
                llm=self.llm,
                base_prompt=prompt,
                llm_config=llm_config,
                user_identity=user_identity,
                target_count=remaining,
                validator=validator,
                existing_questions=existing_for_pass,
                hard_requirements=(
                    _problem_solving_required_markdown(selected_language)
                    if problem_solving_mode
                    else None
                ),
                retry_guidance=retry_guidance,
            )
            retries_used += int(pass_stats.get("retries_used", 0))
            malformed_items_dropped += int(pass_stats.get("malformed_items_dropped", 0))
            pass_metadata = pass_stats.get("metadata", {}) or {}
            if pass_metadata.get("provider"):
                provider_used = str(pass_metadata.get("provider"))
            if pass_metadata.get("model"):
                model_used = str(pass_metadata.get("model"))

            for item in pass_items:
                question_text = str(item.get("question", "")).strip()
                q_norm = _normalise_question(question_text)
                if not q_norm or q_norm in dedup_norm or _is_near_duplicate_question(question_text, seen_questions):
                    continue
                if problem_solving_mode:
                    signature = _problem_solving_pattern_signature(
                        question_text,
                        str(item.get("answer", "")),
                    )
                    if not _accept_problem_pattern_signature(
                        signature=signature,
                        seen_signatures=problem_pattern_signatures,
                        unique_target=max_unique_problem_patterns,
                    ):
                        continue
                dedup_norm.add(q_norm)
                seen_questions.append(question_text)
                raw_items.append(item)
                if len(raw_items) >= target_count:
                    break

        if len(raw_items) < target_count:
            missing_count = target_count - len(raw_items)
            fallback_items = _build_fallback_question_items(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=_problem_solving_fallback_count_with_headroom(
                    missing_count,
                    problem_solving_mode,
                ),
                difficulty=difficulty,
                level=level,
                section_title=None,
                section_content=None,
                existing_questions=[
                    *existing_seed,
                    *[str(item.get("question", "")) for item in raw_items],
                ],
                preferred_language=selected_language,
                requires_programming=requires_programming,
            )
            for item in fallback_items:
                question_text = str(item.get("question", "")).strip()
                q_norm = _normalise_question(question_text)
                if not q_norm or q_norm in dedup_norm or _is_near_duplicate_question(question_text, seen_questions):
                    continue
                if problem_solving_mode:
                    signature = _problem_solving_pattern_signature(
                        question_text,
                        str(item.get("answer", "")),
                    )
                    if not _accept_problem_pattern_signature(
                        signature=signature,
                        seen_signatures=problem_pattern_signatures,
                        unique_target=max_unique_problem_patterns,
                    ):
                        continue
                dedup_norm.add(q_norm)
                seen_questions.append(question_text)
                raw_items.append(item)
                if len(raw_items) >= target_count:
                    break

        questions = [self._to_question_answer_v2(topic_id, item) for item in raw_items[:target_count]]

        return GenerateQuestionsV2Response(
            topic_id=topic_id,
            topic_title=topic_title,
            questions=questions,
            provider_used=provider_used,
            model_used=model_used,
            retries_used=retries_used,
            malformed_items_dropped=malformed_items_dropped,
        )

    async def generate_quiz_v2_stream(
        self,
        topics_content: list[dict],
        count: int = 10,
        question_types: list[str] | None = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        response_detail: str | None = None,
        preferred_language: str = "",
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        target_count = max(1, int(count or 1))
        effective_detail = "very_detailed" if response_detail == "very_detailed" else "concise"
        mcp_context = await self._mcp_context_for_flow(
            flow="quiz",
            query=" ".join(
                [
                    " ".join(str(tc.get("title", "")).strip() for tc in topics_content[:4]),
                    "technical quiz generation",
                ]
            ).strip(),
        )
        allowed_topics = {str(tc["id"]) for tc in topics_content}
        allowed_types = set(question_types or ["mcq", "true_false"])
        if not allowed_types:
            allowed_types = {"mcq", "true_false"}

        dedup_norm: set[str] = set()
        generated_question_texts: list[str] = []
        provider_used = ""
        model_used = ""
        retries_used = 0
        malformed_items_dropped = 0
        generated_count = 0

        async def _generate_one() -> dict[str, Any]:
            local_retries = 0
            local_malformed = 0
            metadata: dict[str, Any] = {}
            issue_counter: Counter[str] = Counter()
            prompt = _build_quiz_prompt(
                topics_content,
                1,
                question_types,
                difficulty,
                level,
                response_detail=effective_detail,
                preferred_language=(preferred_language or "").strip().lower(),
                mcp_context=mcp_context,
            )
            base_prompt = prompt
            try:
                for _ in range(_attempt_budget(1)):
                    result = await self.llm.completion(
                        prompt,
                        llm_config,
                        user_identity=user_identity,
                    )
                    raise_if_policy_blocked_result(result)
                    metadata = result.get("metadata", {})
                    if not result.get("success"):
                        local_retries += 1
                        issue_counter["transport_or_provider_error"] += 1
                        prompt = _build_retry_prompt(
                            base_prompt=base_prompt,
                            missing_count=1,
                            issues="transport_or_provider_error",
                            existing_questions=generated_question_texts,
                        )
                        continue

                    parsed = _parse_questions_json(result.get("analysis", ""))
                    if not parsed:
                        local_retries += 1
                        local_malformed += 1
                        issue_counter["json_parse_failed"] += 1
                        prompt = _build_retry_prompt(
                            base_prompt=base_prompt,
                            missing_count=1,
                            issues="json_parse_failed",
                            existing_questions=generated_question_texts,
                        )
                        continue

                    selected_item: dict[str, Any] | None = None
                    for item in parsed:
                        if not isinstance(item, dict):
                            local_malformed += 1
                            issue_counter["non_dict_item"] += 1
                            continue
                        is_valid, issue = _validate_quiz_item(
                            item,
                            allowed_topics,
                            allowed_types,
                            difficulty,
                            level,
                        )
                        if not is_valid:
                            local_malformed += 1
                            issue_counter[issue] += 1
                            continue
                        question_text = str(item.get("question", "")).strip()
                        q_norm = _normalise_question(question_text)
                        if (
                            not q_norm
                            or q_norm in dedup_norm
                            or _is_near_duplicate_question(question_text, generated_question_texts)
                        ):
                            local_malformed += 1
                            issue_counter["duplicate_question"] += 1
                            continue
                        dedup_norm.add(q_norm)
                        generated_question_texts.append(question_text)
                        selected_item = item
                        break

                    if selected_item is not None:
                        question = self._to_quiz_question_v2(selected_item)
                        return {
                            "question": question.model_dump(),
                            "metadata": metadata,
                            "retries_used": local_retries,
                            "malformed_items_dropped": local_malformed,
                        }

                    local_retries += 1
                    top_issues = (
                        ", ".join(f"{k}:{v}" for k, v in issue_counter.most_common(5))
                        or "insufficient_valid_items"
                    )
                    prompt = _build_retry_prompt(
                        base_prompt=base_prompt,
                        missing_count=1,
                        issues=top_issues,
                        existing_questions=generated_question_texts,
                    )

                return {
                    "question": None,
                    "metadata": metadata,
                    "retries_used": local_retries,
                    "malformed_items_dropped": local_malformed,
                }
            except (
                LLMServiceApprovalRequiredError,
                StudyAppLLMNotAssignedError,
                PersonalCredentialRequiredError,
            ) as exc:
                code, message = self._policy_error_payload(exc)
                return {
                    "error": {"code": code, "message": message},
                    "metadata": metadata,
                    "retries_used": local_retries,
                    "malformed_items_dropped": local_malformed,
                }
            except Exception as exc:
                logger.exception("Streaming quiz generation failed: %s", exc)
                return {
                    "error": {
                        "code": "generation_failed",
                        "message": str(exc).strip() or "Quiz generation failed",
                    },
                    "metadata": metadata,
                    "retries_used": local_retries,
                    "malformed_items_dropped": local_malformed,
                }

        yield {
            "type": "start",
            "target_count": target_count,
            "topics_used": sorted(allowed_topics),
            "question_types": sorted(allowed_types),
        }

        while generated_count < target_count:
            yield {
                "type": "progress",
                "stage": "generating",
                "message": f"Generating quiz question {generated_count + 1} of {target_count}.",
                "generated_count": generated_count,
                "target_count": target_count,
            }
            result = await _generate_one()
            retries_used += int(result.get("retries_used", 0))
            malformed_items_dropped += int(result.get("malformed_items_dropped", 0))
            metadata = result.get("metadata", {}) or {}
            if metadata.get("provider"):
                provider_used = str(metadata.get("provider"))
            if metadata.get("model"):
                model_used = str(metadata.get("model"))

            error = result.get("error")
            if error:
                yield {
                    "type": "error",
                    "code": str(error.get("code", "generation_failed")),
                    "message": str(error.get("message", "Quiz generation failed")),
                }
                return

            question = result.get("question")
            if not isinstance(question, dict):
                break
            generated_count += 1
            yield {"type": "question", "question": question}

        if generated_count < target_count:
            for _ in range(_recovery_budget(target_count - generated_count)):
                if generated_count >= target_count:
                    break
                yield {
                    "type": "progress",
                    "stage": "recovering",
                    "message": (
                        f"Recovering missing quiz questions: {generated_count}/{target_count} generated."
                    ),
                    "generated_count": generated_count,
                    "target_count": target_count,
                }
                result = await _generate_one()
                retries_used += int(result.get("retries_used", 0))
                malformed_items_dropped += int(result.get("malformed_items_dropped", 0))
                metadata = result.get("metadata", {}) or {}
                if metadata.get("provider"):
                    provider_used = str(metadata.get("provider"))
                if metadata.get("model"):
                    model_used = str(metadata.get("model"))
                error = result.get("error")
                if error:
                    yield {
                        "type": "error",
                        "code": str(error.get("code", "generation_failed")),
                        "message": str(error.get("message", "Quiz generation failed")),
                    }
                    return
                question = result.get("question")
                if not isinstance(question, dict):
                    continue
                generated_count += 1
                yield {"type": "question", "question": question}

        if generated_count < target_count:
            fallback_items = _build_fallback_quiz_items(
                topics_content=topics_content,
                count=target_count - generated_count,
                question_types=question_types,
                difficulty=difficulty,
                level=level,
                existing_questions=generated_question_texts,
            )
            for item in fallback_items:
                if generated_count >= target_count:
                    break
                question_text = str(item.get("question", "")).strip()
                q_norm = _normalise_question(question_text)
                if (
                    not q_norm
                    or q_norm in dedup_norm
                    or _is_near_duplicate_question(question_text, generated_question_texts)
                ):
                    continue
                dedup_norm.add(q_norm)
                generated_question_texts.append(question_text)
                question = self._to_quiz_question_v2(item)
                generated_count += 1
                yield {"type": "question", "question": question.model_dump()}

        yield {
            "type": "done",
            "generated_count": generated_count,
            "target_count": target_count,
            "topics_used": sorted(allowed_topics),
            "provider_used": provider_used,
            "model_used": model_used,
            "retries_used": retries_used,
            "malformed_items_dropped": malformed_items_dropped,
        }

    async def generate_quiz(
        self,
        topics_content: list[dict],
        count: int = 10,
        question_types: list[str] | None = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        response_detail: str | None = None,
        preferred_language: str = "",
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> GenerateQuizResponse:
        v2 = await self.generate_quiz_v2(
            topics_content=topics_content,
            count=count,
            question_types=question_types,
            difficulty=difficulty,
            level=level,
            response_detail=response_detail,
            preferred_language=preferred_language,
            llm_config=llm_config,
            user_identity=user_identity,
        )
        legacy_questions = [
            QuizQuestion(
                question=q.question,
                type=q.type,
                choices=q.choices,
                correct_answer=q.correct_answer,
                explanation=q.explanation,
                difficulty=q.difficulty,
                topic_id=q.topic_id,
            )
            for q in v2.questions
        ]
        return GenerateQuizResponse(
            questions=legacy_questions,
            topics_used=v2.topics_used,
            provider_used=v2.provider_used,
            model_used=v2.model_used,
        )

    async def generate_quiz_v2(
        self,
        topics_content: list[dict],
        count: int = 10,
        question_types: list[str] | None = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        response_detail: str | None = None,
        preferred_language: str = "",
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> GenerateQuizV2Response:
        effective_detail = "very_detailed" if response_detail == "very_detailed" else "concise"
        mcp_context = await self._mcp_context_for_flow(
            flow="quiz",
            query=" ".join(
                [
                    " ".join(str(tc.get("title", "")).strip() for tc in topics_content[:4]),
                    "technical quiz generation",
                ]
            ).strip(),
        )
        prompt = _build_quiz_prompt(
            topics_content,
            count,
            question_types,
            difficulty,
            level,
            response_detail=effective_detail,
            preferred_language=(preferred_language or "").strip().lower(),
            mcp_context=mcp_context,
        )
        allowed_topics = {str(tc["id"]) for tc in topics_content}
        allowed_types = set(question_types or ["mcq", "true_false"])
        validator = lambda item: _validate_quiz_item(
            item,
            allowed_topics,
            allowed_types,
            difficulty,
            level,
        )

        raw_items, stats = await _collect_with_retries(
            llm=self.llm,
            base_prompt=prompt,
            llm_config=llm_config,
            user_identity=user_identity,
            target_count=count,
            validator=validator,
        )

        questions = []
        for item in raw_items:
            topic_id = str(item.get("topic_id", "")).strip()
            qid = _question_id(topic_id or "quiz", item["question"])
            choices = [QuizChoice(label=c["label"], text=c["text"]) for c in item["choices"]]
            questions.append(
                QuizQuestionV2(
                    question_id=qid,
                    question=item["question"].strip(),
                    type=QuizQuestionType(item["type"]),
                    choices=choices,
                    correct_answer=str(item["correct_answer"]).strip(),
                    explanation=format_markdown_readable(item["explanation"].strip()),
                    difficulty=item["difficulty"],
                    topic_id=topic_id,
                    source_quote=item["source_quote"].strip(),
                    reasoning_summary=item["reasoning_summary"].strip(),
                )
            )

        metadata = stats.get("metadata", {})
        topic_ids = sorted({tc["id"] for tc in topics_content})
        return GenerateQuizV2Response(
            questions=questions,
            topics_used=topic_ids,
            provider_used=metadata.get("provider", ""),
            model_used=metadata.get("model", ""),
            retries_used=stats.get("retries_used", 0),
            malformed_items_dropped=stats.get("malformed_items_dropped", 0),
        )
