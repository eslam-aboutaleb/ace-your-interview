"""Skill taxonomies for the agentic LLM learning system.

The agentic LLM draws on three skill domains to raise the quality of
generated questions, evaluations, and curricula:

* **Technical skills** — problem-solving patterns, algorithmic thinking,
  data-structure and system-design fundamentals.
* **Interview skills** — communication frameworks, trick-question
  patterns, and behavioral strategies a real interviewer probes.
* **Educational skills** — pedagogy: Socratic questioning, scaffolding,
  misconception detection, and feedback technique.

Each domain maps stable skill ids to structured records. The render
helpers turn a selection into a fenced prompt block the generation
flows inject, so the model reasons with the same skill vocabulary the
learner is assessed against.
"""

from __future__ import annotations

from typing import Any, Iterable

# ── Technical / coding skills ────────────────────────────────
# Each entry: name, category, description, patterns (when to use),
# common_mistakes, interview_tricks (probes a strong interviewer asks),
# complexity.

TECHNICAL_SKILLS: dict[str, dict[str, Any]] = {
    "arrays-two-pointer": {
        "name": "Arrays & Two-Pointer Technique",
        "category": "data-structures",
        "description": "Traverse a sorted or unsorted array with two converging or same-direction pointers instead of nested loops.",
        "patterns": ["pair sum in sorted array", "remove duplicates in place", "container with most water", "3sum", "partition"],
        "common_mistakes": ["off-by-one when moving a pointer", "assuming sorted input without checking", "mutating while iterating"],
        "interview_tricks": ["What happens with duplicate values?", "How does it behave on an empty or single-element array?", "Can you do it in one pass?"],
        "complexity": "O(n) time, O(1) space",
    },
    "hash-map-frequency": {
        "name": "Hash Maps & Frequency Counting",
        "category": "data-structures",
        "description": "Use a dict to count occurrences or index values for O(1) lookups, turning nested scans into single passes.",
        "patterns": ["two sum", "first non-repeating char", "group anagrams", "top-k frequent", "subarray sum equals k"],
        "common_mistakes": ["forgetting to initialise a key", "confusing value with index", "not handling hash collisions conceptually"],
        "interview_tricks": ["What is the worst-case lookup?", "How would you handle a stream instead of a fixed array?"],
        "complexity": "O(n) time, O(n) space",
    },
    "sliding-window": {
        "name": "Sliding Window",
        "category": "techniques",
        "description": "Maintain a contiguous window over an array/string and expand/shrink it to satisfy a constraint in O(n).",
        "patterns": ["longest substring without repeating chars", "min window substring", "max sum subarray of size k", "permutation in string"],
        "common_mistakes": ["shrinking the window on the wrong condition", "forgetting to update the best answer before shrinking"],
        "interview_tricks": ["What if the window size is variable vs fixed?", "How do you handle negative numbers?"],
        "complexity": "O(n) time, O(1) space",
    },
    "stack-monotonic": {
        "name": "Stacks & Monotonic Variants",
        "category": "data-structures",
        "description": "Use a LIFO stack (often monotonic) to track pending elements and resolve them in one pass.",
        "patterns": ["valid parentheses", "next greater element", "largest rectangle in histogram", "min stack", "daily temperatures"],
        "common_mistakes": ["popping on the wrong comparison", "forgetting to push the index not the value"],
        "interview_tricks": ["Why a stack and not a queue?", "What is the invariant of the monotonic stack?"],
        "complexity": "O(n) time, O(n) space",
    },
    "binary-search": {
        "name": "Binary Search & Search Space",
        "category": "techniques",
        "description": "Halve a monotonic search space each step; applies to sorted arrays and to 'binary search on answer' problems.",
        "patterns": ["find target", "first/last occurrence", "search in rotated array", "koko eating bananas", "median of two sorted arrays"],
        "common_mistakes": ["infinite loop from mid calculation", "wrong boundary update (lo/hi)", "missing the post-loop check"],
        "interview_tricks": ["What is the loop invariant?", "How do you avoid integer overflow in mid?", "When does binary search on answer apply?"],
        "complexity": "O(log n) time, O(1) space",
    },
    "linked-list-fast-slow": {
        "name": "Linked Lists & Fast/Slow Pointers",
        "category": "data-structures",
        "description": "Traverse linked structures with two speeds to detect cycles or find midpoints without extra space.",
        "patterns": ["cycle detection", "find middle node", "merge two lists", "reverse a list", "remove nth from end"],
        "common_mistakes": ["null pointer on empty list", "losing the next reference when reversing"],
        "interview_tricks": ["How do you prove the fast/slow pointers meet?", "What is the space cost of a hash-set alternative?"],
        "complexity": "O(n) time, O(1) space",
    },
    "tree-dfs-bfs": {
        "name": "Trees & DFS/BFS Traversal",
        "category": "data-structures",
        "description": "Traverse trees depth-first (pre/in/post-order) or breadth-first (level order), recursively or iteratively.",
        "patterns": ["inorder traversal", "level order", "lowest common ancestor", "diameter", "validate BST", "path sum"],
        "common_mistakes": ["mixing up pre/in/post order", "forgetting the base case", "not tracking depth"],
        "interview_tricks": ["Recursive vs iterative — what is the call-stack cost?", "How do you handle an unbalanced tree?"],
        "complexity": "O(n) time, O(h) space",
    },
    "heap-priority-queue": {
        "name": "Heaps & Priority Queues",
        "category": "data-structures",
        "description": "Maintain a running top-k or median with a binary heap in O(log n) per operation.",
        "patterns": ["top-k frequent", "merge k sorted lists", "find median from stream", "task scheduler", "k closest points"],
        "common_mistakes": ["using a max-heap where a min-heap is needed", "forgetting heapify is O(n)"],
        "interview_tricks": ["Why two heaps for a running median?", "What is the cost of building a heap from an array?"],
        "complexity": "O(n log k) time, O(k) space",
    },
    "dynamic-programming": {
        "name": "Dynamic Programming",
        "category": "techniques",
        "description": "Solve overlapping subproblems once, memoise, and build up from a base case; recognise optimal substructure.",
        "patterns": ["fibonacci", "climbing stairs", "coin change", "longest common subsequence", "0/1 knapsack", "house robber"],
        "common_mistakes": ["not defining the state", "wrong recurrence", "off-by-one on the base case", "confusing top-down with bottom-up"],
        "interview_tricks": ["What is the state and the transition?", "Can you reduce to O(1) space?", "How do you prove optimal substructure?"],
        "complexity": "O(n*k) time, O(n) or O(1) space",
    },
    "graph-traversal": {
        "name": "Graph Traversal",
        "category": "techniques",
        "description": "BFS for shortest unweighted paths, DFS for connectivity and cycle detection; represent as adjacency list.",
        "patterns": ["number of islands", "course schedule", "clone graph", "word ladder", "rotting oranges"],
        "common_mistakes": ["not marking visited", "confusing BFS with DFS for shortest path", "ignoring disconnected components"],
        "interview_tricks": ["BFS or DFS — why?", "How do you detect a cycle in a directed graph?"],
        "complexity": "O(V+E) time, O(V) space",
    },
    "union-find": {
        "name": "Union-Find (Disjoint Set)",
        "category": "techniques",
        "description": "Track connected components with near-constant union and find using path compression and union by rank.",
        "patterns": ["number of connected components", "redundant connection", "accounts merge", "satisfiability of equality equations"],
        "common_mistakes": ["forgetting path compression", "not using union by rank"],
        "interview_tricks": ["What is the amortised cost with both optimisations?", "When is Union-Find better than DFS?"],
        "complexity": "O(alpha(n)) amortised per op",
    },
    "backtracking": {
        "name": "Backtracking",
        "category": "techniques",
        "description": "Build candidates incrementally and abandon a branch as soon as it violates a constraint.",
        "patterns": ["subsets", "permutations", "combination sum", "n-queens", "word search", "generate parentheses"],
        "common_mistakes": ["not undoing the choice (missing backtrack step)", "duplicate results from not sorting/skipping"],
        "interview_tricks": ["What is the decision tree?", "How do you avoid duplicates?"],
        "complexity": "O(2^n) or O(n!) time",
    },
    "bit-manipulation": {
        "name": "Bit Manipulation",
        "category": "techniques",
        "description": "Use XOR, AND, shifts and masks to solve problems in O(1) or O(n) with O(1) space.",
        "patterns": ["single number (XOR)", "counting bits", "power of two", "missing number", "reverse bits"],
        "common_mistakes": ["confusing arithmetic shift with logical shift", "off-by-one on bit positions"],
        "interview_tricks": ["Why does XOR cancel duplicates?", "How do you isolate the lowest set bit?"],
        "complexity": "O(n) time, O(1) space",
    },
    "recursion": {
        "name": "Recursion & Divide-and-Conquer",
        "category": "techniques",
        "description": "Reduce a problem to smaller instances of itself; combine results. Foundation for mergesort, quicksort, tree algorithms.",
        "patterns": ["mergesort", "quicksort", "power function", "subset generation", "tower of hanoi"],
        "common_mistakes": ["missing base case", "stack overflow on deep recursion", "redundant recomputation"],
        "interview_tricks": ["What is the recurrence relation?", "What is the recursion depth?"],
        "complexity": "O(n log n) typical for sorting",
    },
    "system-design-basics": {
        "name": "System Design Fundamentals",
        "category": "system-design",
        "description": "Reason about scale: load balancing, caching, database choice, consistency vs availability, and API design.",
        "patterns": ["design a URL shortener", "rate limiter", "cache (LRU)", "chat system", "news feed"],
        "common_mistakes": ["jumping to components before estimating load", "ignoring the consistency/availability tradeoff", "not stating assumptions"],
        "interview_tricks": ["What are the QPS and storage estimates?", "Where is the bottleneck?", "What happens on failure?"],
        "complexity": "N/A — tradeoff-driven",
    },
}

# ── Interview skills ─────────────────────────────────────────

INTERVIEW_SKILLS: dict[str, dict[str, Any]] = {
    "clarify-before-code": {
        "name": "Clarify Before Coding",
        "description": "Ask targeted questions about inputs, constraints, and edge cases before writing any code.",
        "when_to_use": "Every coding problem, first 2-3 minutes.",
        "technique": "Restate the problem, confirm input ranges, ask about duplicates/sortedness, and name the edge cases you will handle.",
        "example": "So I need to return indices of the two numbers that sum to target. Are there duplicates? Is there exactly one solution? Can the array be empty?",
    },
    "brute-force-first": {
        "name": "Brute Force First, Then Optimise",
        "description": "State the naive solution, its complexity, then improve — showing you understand the tradeoff.",
        "when_to_use": "When the optimal approach is not immediately obvious.",
        "technique": "Describe the O(n^2) approach, name its cost, then introduce the optimisation and why it is better.",
        "example": "A brute-force check of every pair is O(n^2). We can do better with a hash map: one pass, O(n) time and O(n) space.",
    },
    "complexity-verbalization": {
        "name": "Verbalise Time & Space Complexity",
        "description": "State the complexity of every approach and explain the dominant term.",
        "when_to_use": "After describing any approach.",
        "technique": "Name time and space separately, identify the bottleneck operation, and mention the tradeoff.",
        "example": "This is O(n) time because we scan once, and O(n) space for the hash map. The space is the tradeoff for the speed.",
    },
    "edge-case-enumeration": {
        "name": "Enumerate Edge Cases",
        "description": "Proactively list boundary inputs and handle them explicitly.",
        "when_to_use": "Before and during implementation.",
        "technique": "List empty input, single element, duplicates, negatives, overflow, and already-sorted cases; test at least two.",
        "example": "Edge cases: empty array, one element, all duplicates, target at the ends. I will handle empty and single-element up front.",
    },
    "tradeoff-discussion": {
        "name": "Discuss Tradeoffs Explicitly",
        "description": "Compare two valid approaches on time, space, readability, and maintainability.",
        "when_to_use": "When two approaches are viable.",
        "technique": "Give a pros/cons comparison and state which you would ship and why.",
        "example": "The two-pointer approach is O(n) time and O(1) space but needs sorted input; the hash map is O(n) time and O(n) space but works unsorted. I would pick the hash map for clarity.",
    },
    "star-structure": {
        "name": "STAR for Behavioral Answers",
        "description": "Answer behavioral questions with Situation, Task, Action, Result.",
        "when_to_use": "Behavioral questions ('tell me about a time...').",
        "technique": "One sentence of situation, one of task, the bulk on your specific actions, and a quantified result plus a reflection.",
        "example": "Situation: our deploy pipeline broke nightly. Task: I owned reducing failures. Action: I added automated rollback and canary checks. Result: failures dropped 70% and I documented the runbook.",
    },
    "behavioral-ownership": {
        "name": "Own Your Contribution",
        "description": "Use 'I' for your actions; credit the team without hiding your role.",
        "when_to_use": "Behavioral and project questions.",
        "technique": "Describe what you specifically did, why you chose it, and what you would repeat or change.",
        "example": "I proposed the caching layer because I measured the read path; the team reviewed it, and I implemented it behind a feature flag.",
    },
    "estimation-guesstimate": {
        "name": "Back-of-Envelope Estimation",
        "description": "Estimate load, storage, or bandwidth with stated assumptions and round numbers.",
        "when_to_use": "System-design questions.",
        "technique": "State assumptions, round to powers of ten, compute QPS and storage, and sanity-check the result.",
        "example": "Assume 100M users, 10% daily active, 5 requests each: ~50K QPS average, ~10x peak. At 1KB per event, that is ~50MB/s to ingest.",
    },
    "debugging-narrative": {
        "name": "Structured Debugging",
        "description": "Localise a bug with a hypothesis, a test, and a fix — not random changes.",
        "when_to_use": "Debugging or 'find the bug' questions.",
        "technique": "Reproduce, form a hypothesis, isolate with a minimal test, fix, and verify; mention logging and bisecting.",
        "example": "I would reproduce it with the smallest input, add a log at the boundary, and bisect the change set to find the offending commit.",
    },
    "follow-up-questions": {
        "name": "Ask Insightful Follow-ups",
        "description": "Turn an answer into a dialogue by asking about tradeoffs, scale, or failure modes.",
        "when_to_use": "End of an answer, or when the interviewer hints.",
        "technique": "Ask one specific question that shows depth: 'How would this change at 10x traffic?' or 'What is the failure mode here?'",
        "example": "If this service hit 10x traffic, the database would be the bottleneck — would you shard, read-replica, or cache first?",
    },
}

# ── Educational / pedagogical skills ─────────────────────────

EDUCATIONAL_SKILLS: dict[str, dict[str, Any]] = {
    "socratic-questioning": {
        "name": "Socratic Questioning",
        "description": "Guide the learner to the answer with questions instead of giving it away.",
        "technique": "Ask one question at a time, building from what the learner already knows; never supply the next step until they attempt it.",
        "example": "You know a hash map gives O(1) lookup — how could that remove the inner loop?",
    },
    "scaffolding": {
        "name": "Scaffolding",
        "description": "Break a hard problem into smaller steps and support each, then fade the support.",
        "technique": "Provide a skeleton, a hint, or a worked first step; reduce help as competence grows.",
        "example": "Start by writing the function signature and the base case. Then think about the recursive call on the smaller list.",
    },
    "misconception-detection": {
        "name": "Misconception Detection",
        "description": "Spot the specific wrong mental model behind an error and correct that, not just the answer.",
        "technique": "Identify the pattern in the mistake (e.g. thinking a list is passed by value) and address the root belief.",
        "example": "It looks like you think list.append returns a new list. It mutates in place and returns None — that is why your variable is None.",
    },
    "retrieval-practice": {
        "name": "Retrieval Practice",
        "description": "Have the learner recall from memory before re-reading; recall strengthens retention.",
        "technique": "Ask the learner to explain the concept or solve the problem from memory before showing any reference.",
        "example": "Before we look at the code, tell me in your own words how a monotonic stack keeps the next-greater element.",
    },
    "worked-example": {
        "name": "Worked Example",
        "description": "Show a complete, annotated solution, then have the learner solve a near-transfer variant.",
        "technique": "Walk one example step by step, naming each decision, then give a similar problem for the learner.",
        "example": "Watch me solve two-sum with a hash map on [2,7,11,15]. Now you try [3,2,4] and tell me the map state at each step.",
    },
    "deliberate-practice": {
        "name": "Deliberate Practice",
        "description": "Target the learner's weakest sub-skill with focused, feedback-rich reps just above their level.",
        "technique": "Diagnose the weak point, design a narrow drill, and give immediate specific feedback.",
        "example": "Your recurrences are solid but your base cases slip. Let's do five small DP problems focusing only on the base case.",
    },
    "feedback-sandwich": {
        "name": "Actionable Feedback",
        "description": "Give specific, evidence-based feedback: what was strong, what to fix, and how.",
        "technique": "Cite the exact line or reasoning, state the impact, and give one concrete next step — not vague praise.",
        "example": "Your complexity analysis was correct (strong). You missed the duplicate case (fix): add a check when the pointer value equals its neighbour (how).",
    },
    "analogical-reasoning": {
        "name": "Analogical Reasoning",
        "description": "Anchor an abstract concept to a concrete analogy the learner already understands.",
        "technique": "Pick an analogy, map each part explicitly, and note where the analogy breaks.",
        "example": "A cache is like a desk drawer: fast to reach but small. The bookshelf (database) is slow but holds everything. The analogy breaks because a drawer does not evict.",
    },
    "metacognition": {
        "name": "Metacognition & Reflection",
        "description": "Prompt the learner to reflect on how they solved a problem and what they would change.",
        "technique": "After a solution, ask: what was the hardest step? what would you do differently next time?",
        "example": "What was the trickiest part of this problem, and what pattern would you now look for first?",
    },
    "spaced-repetition": {
        "name": "Spaced Repetition",
        "description": "Schedule review of a concept at increasing intervals to move it into long-term memory.",
        "technique": "Flag a concept for review at 1 day, 3 days, 1 week, and 1 month; tie it to a flashcard or follow-up question.",
        "example": "We will revisit monotonic stacks in three days with a harder variant, so the pattern sticks.",
    },
}

# ── Render helpers ───────────────────────────────────────────

def _skill_lines(skill_ids: Iterable[str], source: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for skill_id in skill_ids:
        record = source.get(str(skill_id).strip())
        if not record:
            continue
        lines.append(f"- {record['name']}: {record['description']}")
        for key in ("patterns", "common_mistakes", "interview_tricks"):
            values = record.get(key)
            if values:
                lines.append(f"  - {key.replace('_', ' ')}: " + "; ".join(str(v) for v in values))
        for key in ("complexity", "when_to_use", "technique", "example"):
            value = record.get(key)
            if value:
                lines.append(f"  - {key.replace('_', ' ')}: {value}")
    return lines


def render_skill_block(title: str, skill_ids: Iterable[str], source: dict[str, dict[str, Any]]) -> str:
    """Render a fenced skill block for prompt injection."""
    lines = _skill_lines(skill_ids, source)
    if not lines:
        return ""
    return f"{title}:\n" + "\n".join(lines)


def technical_skill_block(skill_ids: Iterable[str]) -> str:
    return render_skill_block("Technical & coding skills to apply and assess", skill_ids, TECHNICAL_SKILLS)


def interview_skill_block(skill_ids: Iterable[str]) -> str:
    return render_skill_block("Interview strategies to model and probe", skill_ids, INTERVIEW_SKILLS)


def educational_skill_block(skill_ids: Iterable[str]) -> str:
    return render_skill_block("Educational techniques to apply", skill_ids, EDUCATIONAL_SKILLS)


# ── Track/level-aware selection ──────────────────────────────

#: Core technical skills every learner should reach, by level.
_LEVEL_TECHNICAL: dict[str, tuple[str, ...]] = {
    "junior": (
        "arrays-two-pointer", "hash-map-frequency", "sliding-window",
        "binary-search", "recursion", "tree-dfs-bfs",
    ),
    "mid": (
        "stack-monotonic", "linked-list-fast-slow", "heap-priority-queue",
        "graph-traversal", "backtracking", "dynamic-programming",
    ),
    "senior": (
        "dynamic-programming", "union-find", "bit-manipulation",
        "graph-traversal", "system-design-basics",
    ),
}

#: Interview strategies by session type.
_SESSION_INTERVIEW: dict[str, tuple[str, ...]] = {
    "coding": (
        "clarify-before-code", "brute-force-first", "complexity-verbalization",
        "edge-case-enumeration", "tradeoff-discussion", "debugging-narrative",
    ),
    "behavioral": (
        "star-structure", "behavioral-ownership", "follow-up-questions",
    ),
    "system-design": (
        "estimation-guesstimate", "tradeoff-discussion", "follow-up-questions",
        "complexity-verbalization",
    ),
    "general": (
        "clarify-before-code", "edge-case-enumeration", "tradeoff-discussion",
        "follow-up-questions",
    ),
}

#: Educational techniques that improve any learning flow.
_CORE_EDUCATIONAL: tuple[str, ...] = (
    "socratic-questioning", "scaffolding", "misconception-detection",
    "retrieval-practice", "feedback-sandwich", "metacognition",
)


def skills_for_session(
    *,
    level: str = "mid",
    session_type: str = "general",
    include_educational: bool = True,
) -> dict[str, list[str]]:
    """Select the skill ids relevant to a session's level and type."""
    level_key = str(level or "").strip().lower()
    if level_key not in _LEVEL_TECHNICAL:
        level_key = "mid"
    session_key = str(session_type or "").strip().lower()
    if session_key not in _SESSION_INTERVIEW:
        session_key = "general"
    result: dict[str, list[str]] = {
        "technical": list(_LEVEL_TECHNICAL[level_key]),
        "interview": list(_SESSION_INTERVIEW[session_key]),
    }
    if include_educational:
        result["educational"] = list(_CORE_EDUCATIONAL)
    return result


def skill_context_block(
    *,
    level: str = "mid",
    session_type: str = "general",
    include_educational: bool = True,
) -> str:
    """Render the full skill context for a session as one fenced block."""
    selected = skills_for_session(
        level=level, session_type=session_type, include_educational=include_educational
    )
    parts = [
        technical_skill_block(selected["technical"]),
        interview_skill_block(selected["interview"]),
    ]
    if include_educational:
        parts.append(educational_skill_block(selected["educational"]))
    return "\n\n".join(part for part in parts if part)
