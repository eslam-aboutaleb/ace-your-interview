"""Skill taxonomies for the agentic LLM learning system.

The agentic LLM draws on three skill domains to raise the quality of
generated questions, evaluations, and curricula:

* **Technical skills** — problem-solving patterns, algorithmic thinking,
  data-structure and system-design fundamentals, plus per-track packs
  (backend, frontend, system design, AI stack) so a learner practising
  a stack topic is assessed against that stack's vocabulary rather
  than data-structures-and-algorithms patterns.
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

# ── Per-track technical skill packs ───────────────────────────
# Same record shape as ``TECHNICAL_SKILLS`` (name, category,
# description, patterns, common_mistakes, interview_tricks,
# complexity). One pack per interview track, so a session whose topic
# is backend, frontend, system design or the AI stack reasons with the
# vocabulary of that stack instead of DSA patterns.

BACKEND_SKILLS: dict[str, dict[str, Any]] = {
    "http-semantics": {
        "name": "HTTP Semantics & Methods",
        "category": "protocols",
        "description": "Use methods, status codes, and headers with their safety and idempotency semantics, and reason about how client, proxies, and server share state.",
        "patterns": ["map an operation to GET/POST/PUT/PATCH/DELETE with the right safety", "choose 200 vs 201 vs 202 vs 204 by the effect produced", "validate a cached response with ETag/If-None-Match and 304", "set Cache-Control max-age for a cacheable resource", "propagate a request id across services"],
        "common_mistakes": ["using POST for a read because a body is easier", "returning 200 with an error body instead of a 4xx/5xx", "treating PUT as non-idempotent", "assuming a 3xx is always followed automatically"],
        "interview_tricks": ["Which methods are safe and idempotent, and why does retry depend on it?", "Where does the cache live and who is allowed to invalidate it?", "How would you make a retried POST safe?"],
        "complexity": "N/A — semantics-driven",
    },
    "api-contract-design": {
        "name": "API Contract Design",
        "category": "api-design",
        "description": "Design resource-oriented endpoints with stable schemas, explicit versioning, predictable pagination, and a consistent error envelope.",
        "patterns": ["model collections and sub-resources as nested paths", "paginate with cursor/limit instead of offset", "return a typed error envelope with a machine-readable code", "version a breaking change behind a new path or media type", "document partial failure for batch endpoints"],
        "common_mistakes": ["exposing internal table names as URLs", "returning 500 for a client mistake", "leaking nullable fields whose meaning is undefined", "adding a required field to an existing response without a version"],
        "interview_tricks": ["How do you evolve this contract without breaking existing clients?", "What does a caller do with a 429, and what should the response say?", "Where would you draw the boundary between two services here?"],
        "complexity": "N/A — tradeoff-driven",
    },
    "data-modeling-persistence": {
        "name": "Data Modeling & Persistence",
        "category": "data",
        "description": "Choose entities, keys, and indexes for the access pattern; decide between normalised relational rows and embedded documents deliberately.",
        "patterns": ["normalise to the third normal form then denormalise a hot read path", "add a covering/composite index for a filter-plus-sort query", "use a junction table for many-to-many", "enforce invariants with constraints not application code", "plan a migration that is online and reversible"],
        "common_mistakes": ["indexing every column", "storing computed state that can go stale", "using an autoincrement key where a natural key is required", "forgetting the nullability and default of a new column"],
        "interview_tricks": ["Which query is this model optimised for, and which one does it hurt?", "How do you add a NOT NULL column to a billion-row table?", "What is the isolation level this code assumes?"],
        "complexity": "O(log n) reads with a B-tree index; O(1) by primary key",
    },
    "authn-authz": {
        "name": "Authentication & Authorization",
        "category": "security",
        "description": "Verify identity with sessions or tokens, then authorise every action with object-level checks that do not depend on the client hiding UI.",
        "patterns": ["issue short-lived access tokens with a rotating refresh token", "authorise per object, not per endpoint", "store passwords with a slow salted hash", "scope machine-to-machine calls with a least-privilege token", "audit privilege changes"],
        "common_mistakes": ["trusting a user id from the request body", "storing a session token in a predictable place", "checking authorisation only in the frontend", "rolling your own crypto"],
        "interview_tricks": ["Where is the authorisation decision made and who can bypass it?", "What happens when a token is stolen — how do you revoke it?", "How would you model roles that vary per tenant?"],
        "complexity": "O(1) per check with an indexed policy/role lookup",
    },
    "observability": {
        "name": "Observability & Operability",
        "category": "operability",
        "description": "Instrument logs, metrics, and traces with stable dimensions so an incident can be localised without a redeploy.",
        "patterns": ["emit structured logs with a correlation id", "track p50/p95/p99 latency and error ratio per endpoint", "add a trace span across an outbound call", "define an SLO with an error budget", "alert on symptoms, page on causes"],
        "common_mistakes": ["logging secrets or whole request bodies", "using an unbounded label cardinality that explodes metrics", "alerting on CPU instead of user-visible latency", "tracing synchronously in a hot path"],
        "interview_tricks": ["Which single metric tells you this is broken before users do?", "How do you debug a request that only fails in production?", "What would you put on the dashboards for this service?"],
        "complexity": "N/A — operability-driven",
    },
    "concurrency-async": {
        "name": "Concurrency & Async Execution",
        "category": "concurrency",
        "description": "Share mutable state safely with locks, queues, or isolation, and keep blocking work off the event loop.",
        "patterns": ["protect a critical section with a lock sized to the invariant", "use a connection/worker pool instead of unbounded threads", "make an update idempotent so a retry is safe", "serialise conflicting work through a queue", "detect and break a deadlock cycle"],
        "common_mistakes": ["holding a lock across an await or an I/O call", "read-modify-write without a compare-and-set", "blocking the event loop with CPU work", "assuming a thread-safe library makes the composition safe"],
        "interview_tricks": ["What is the invariant the lock protects, and what breaks without it?", "Where is the race window in this read-modify-write?", "How does this behave under 10x concurrency?"],
        "complexity": "O(1) amortised per op; contention bounded by critical-section length",
    },
    "testing-strategy": {
        "name": "Testing Strategy",
        "category": "quality",
        "description": "Choose the cheapest test that can catch each defect: unit for logic, integration for contracts, contract tests across service boundaries, and a few end-to-end journeys.",
        "patterns": ["test the behaviour, not the implementation", "stub at the boundary you own", "use a real database for a migration or transaction test", "pin a contract with a schema check", "make the suite deterministic with fixed clocks and seeds"],
        "common_mistakes": ["mocking the code under test", "asserting on log lines instead of outcomes", "a suite that needs a specific run order", "chasing coverage instead of risk"],
        "interview_tricks": ["Which failure would this test catch that a unit test cannot?", "How do you test a retry path without sleeping?", "What is your strategy for a flaky test you cannot fix today?"],
        "complexity": "N/A — risk-driven",
    },
    "caching-databases": {
        "name": "Caching & Database Access Patterns",
        "category": "data",
        "description": "Put a cache in front of a hot path with an explicit invalidation rule, and keep connection and transaction boundaries tight.",
        "patterns": ["cache-aside with a TTL plus explicit invalidation", "use a write-through cache for read-heavy counters", "batch and index the query behind a N+1", "use a read replica for stale-tolerant reads", "set a statement and transaction timeout"],
        "common_mistakes": ["cache without an invalidation or eviction story", "caching per-user secrets", "letting a transaction span an external call", "assuming a replica read is immediately current"],
        "interview_tricks": ["What is the staleness budget, and what makes this value incorrect?", "How do you handle a cache stampede?", "When would you drop the cache entirely?"],
        "complexity": "O(1) hit; miss costs the origin query, usually O(log n) indexed",
    },
}

FRONTEND_SKILLS: dict[str, dict[str, Any]] = {
    "react-state-and-rendering": {
        "name": "React State & Rendering",
        "category": "frontend",
        "description": "Separate server cache, URL state, and local UI state, and derive values instead of duplicating them, so renders stay predictable.",
        "patterns": ["lift or colocate state to the closest common owner", "derive a value during render instead of mirroring it in state", "key a list by a stable id, not the index", "memoise only a measured expensive subtree", "batch related updates"],
        "common_mistakes": ["state that duplicates props", "mutating state in place", "an effect that copies props into state", "a key that changes on every render"],
        "interview_tricks": ["Why did this component render twice, and which render is wasted?", "Where should this value live: server cache, URL, or local state?", "What happens when two updates land in the same tick?"],
        "complexity": "O(n) per render of the affected subtree",
    },
    "component-architecture": {
        "name": "Component Architecture",
        "category": "frontend",
        "description": "Compose small presentational components with containers that own data access, and keep prop surfaces explicit.",
        "patterns": ["split container from presentational components", "compose with children and slots instead of prop drilling", "extract a hook for reusable stateful logic", "co-locate a component with its styles and tests"],
        "common_mistakes": ["a god component that fetches and renders and validates", "boolean props that select a whole different UI", "a context that re-renders the whole tree on every change"],
        "interview_tricks": ["Which part of this component would you reuse elsewhere, and how?", "What is the public API of this component and what does it not know?", "How would you add a second variant without another prop?"],
        "complexity": "N/A — structure-driven",
    },
    "data-fetching-caching": {
        "name": "Data Fetching & Caching",
        "category": "frontend",
        "description": "Own server state with a cache keyed by request, so loading, refetching, and invalidation are explicit and race-free.",
        "patterns": ["key the cache by the real request identity", "invalidate or update the cache after a mutation", "cancel or ignore a stale response", "prefetch on hover or focus", "optimistic update with a rollback"],
        "common_mistakes": ["fetching in an effect without a cleanup, so an old response wins", "refetching on every mount with no invalidation", "storing server data in component state", "blocking the whole page on one request"],
        "interview_tricks": ["What happens if the response arrives after the user navigates away?", "How do you keep this list fresh without polling?", "Where does the cache live and who invalidates it?"],
        "complexity": "O(1) cache lookup; render cost O(n) of the result set",
    },
    "performance-web-vitals": {
        "name": "Performance & Web Vitals",
        "category": "frontend",
        "description": "Move work off the critical path: cut the main-thread and payload cost that decides LCP, INP, and CLS.",
        "patterns": ["code-split a route or a heavy widget", "lazy-load below-the-fold media", "preload the hero asset", "defer a non-critical third-party script", "reserve space to avoid layout shift"],
        "common_mistakes": ["measuring on a throttled dev build", "shipping a whole icon library for three icons", "a layout shift from an un-sized image or late font", "a long task that blocks the first interaction"],
        "interview_tricks": ["Which metric would you improve first here, and how would you measure it in the field?", "What is on the critical path to first paint?", "What changed if INP regressed after that feature?"],
        "complexity": "Budget-driven: LCP < 2.5s, INP < 200ms, CLS < 0.1",
    },
    "accessibility": {
        "name": "Accessibility",
        "category": "frontend",
        "description": "Expose correct semantics, names, and focus order so the UI is operable by keyboard and assistive technology.",
        "patterns": ["use a native control before a div", "label every input and name every icon button", "manage focus in a modal or route change", "respect prefers-reduced-motion", "check contrast and target size"],
        "common_mistakes": ["a custom widget with no keyboard support", "an aria-role that fights the native semantics", "removing the focus ring", "announcing a live region that never changes"],
        "interview_tricks": ["How would you complete this flow with only a keyboard?", "What does a screen reader announce for this button?", "Which WCAG rule would a user complain about first?"],
        "complexity": "N/A — compliance-driven (WCAG 2.2 AA)",
    },
    "css-layout-systems": {
        "name": "CSS & Layout Systems",
        "category": "frontend",
        "description": "Build layouts from the browser's own model — normal flow, flex, grid, and container queries — with tokens instead of ad-hoc values.",
        "patterns": ["grid for the page, flex for the component", "use clamp/container queries over media-query breakpoints", "express spacing and colour as design tokens", "scope a component's styles"],
        "common_mistakes": ["absolute positioning for whole layouts", "magic pixel values that break at a new viewport", "specificity escalation with nested selectors", "a fixed height that clips translated content"],
        "interview_tricks": ["How does this layout degrade at 320px and at 2560px?", "Why does the container overflow and what is the cheapest fix?", "What is the cascade order here?"],
        "complexity": "N/A — layout-driven",
    },
    "typescript-type-design": {
        "name": "TypeScript Type Design",
        "category": "frontend",
        "description": "Model the domain so illegal states cannot be constructed, and let the compiler carry the invariants across component boundaries.",
        "patterns": ["discriminated unions over optional flags", "brand or template types for ids and units", "narrow at the boundary, trust afterwards", "share the API type with the backend contract", "exhaustive switch on a union"],
        "common_mistakes": ["any at a boundary that then spreads", "a type that asserts instead of verifying", "overloaded generics nobody can read", "duplicating a type the API already owns"],
        "interview_tricks": ["What state can this type represent that the domain cannot?", "Where does the unknown become a known?", "How would you make the compiler reject this call?"],
        "complexity": "N/A — type-driven (compile-time cost only)",
    },
    "testing-ui": {
        "name": "UI Testing",
        "category": "frontend",
        "description": "Test what a user can observe — roles, text, and behaviour — instead of implementation details that refactors break.",
        "patterns": ["query by role and accessible name", "fire user events, not internal calls", "await an appearance assertion rather than a timeout", "mock at the network boundary", "test keyboard and focus behaviour"],
        "common_mistakes": ["selecting by class name or snapshot text", "waiting on a fixed sleep", "mocking the component's own hook", "asserting on a store's internal shape"],
        "interview_tricks": ["What would break if this markup changed but the behaviour did not?", "How do you test a flow that depends on a slow request?", "Which test would catch a regression a snapshot cannot?"],
        "complexity": "N/A — risk-driven",
    },
}

SYSTEM_DESIGN_SKILLS: dict[str, dict[str, Any]] = {
    "load-estimation": {
        "name": "Load Estimation",
        "category": "system-design",
        "description": "Turn a product description into QPS, storage, and bandwidth numbers before naming any component.",
        "patterns": ["state assumptions then round to powers of ten", "derive QPS from DAU and actions per user", "size storage from write rate times retention", "apply a peak-to-average factor", "sanity-check the result against a known system"],
        "common_mistakes": ["skipping assumptions and jumping to a component", "ignoring the read/write ratio", "forgetting replication factor in storage", "designing for the average, not the peak"],
        "interview_tricks": ["What are your QPS and storage estimates, and what assumption drives them?", "How does this change at 10x the users you assumed?", "Where does the peak come from — time of day or an event?"],
        "complexity": "N/A — estimation-driven",
    },
    "caching-strategies": {
        "name": "Caching Strategies",
        "category": "system-design",
        "description": "Place a cache where it removes the most load, with a policy for staleness, eviction, and failure.",
        "patterns": ["cache-aside with TTL plus explicit invalidation", "write-through for read-heavy counters", "CDN or edge cache for immutable assets", "request coalescing to stop a stampede", "negative caching for misses"],
        "common_mistakes": ["no invalidation or eviction story", "caching personalised data on a shared key", "a cache that fails open with no origin protection", "measuring hit rate while the origin still saturates"],
        "interview_tricks": ["What is the staleness budget for this value?", "What happens to the database when the cache goes cold?", "How would you detect that the cache is silently wrong?"],
        "complexity": "O(1) hit; miss pays the origin cost",
    },
    "database-scaling": {
        "name": "Database Scaling",
        "category": "system-design",
        "description": "Scale reads and writes with replicas, sharding, or partitioning, and keep the key choice and joins honest.",
        "patterns": ["read replicas for stale-tolerant reads", "partition by a key that spreads load", "shard by tenant or hash with a resharding plan", "move a hot counter to an approximate sketch", "archive cold rows to cheaper storage"],
        "common_mistakes": ["a shard key that creates a hotspot", "cross-shard joins and transactions", "replication lag treated as an error", "scaling up forever with no exit plan"],
        "interview_tricks": ["What is your shard key and what breaks when it is uneven?", "How do you reshard without downtime?", "Which query cannot be served by this design?"],
        "complexity": "O(log n) indexed reads; shards scale linearly in count",
    },
    "consistency-tradeoffs": {
        "name": "Consistency Tradeoffs",
        "category": "system-design",
        "description": "Choose the consistency level the feature actually needs, and make the visible anomalies explicit.",
        "patterns": ["strong consistency for money, eventual for feeds", "linearisable writes via a single leader or consensus", "read-your-writes with a session pin", "idempotent consumers for at-least-once delivery", "a saga with compensating actions"],
        "common_mistakes": ["claiming consistency without proving the failure case", "ignoring that a clock read is not an oracle", "a distributed transaction with no compensation path", "assuming a quorum read implies a quorum write"],
        "interview_tricks": ["What does a user see during a partition here?", "Which invariant would you never relax, and why?", "How do you reconcile two divergent writes?"],
        "complexity": "Consensus adds a round trip: O(1) ops, latency-bound",
    },
    "queues-backpressure": {
        "name": "Queues & Backpressure",
        "category": "system-design",
        "description": "Decouple producers and consumers with a durable queue, and bound the in-flight work so a slow consumer cannot take the system down.",
        "patterns": ["durable queue with an explicit DLQ", "bounded concurrency on the consumer", "retry with jittered backoff and an idempotency key", "shed load at the edge when full", "drain and replay after an incident"],
        "common_mistakes": ["an unbounded queue that hides failure until it is fatal", "at-least-once delivery without idempotency", "a retry storm with no backoff", "no visibility into queue depth or consumer lag"],
        "interview_tricks": ["What happens when the consumer falls an hour behind?", "How do you make a redelivered message safe?", "Where do you drop load first, and what does the user see?"],
        "complexity": "O(1) enqueue; throughput bounded by consumer concurrency",
    },
    "infrastructure-as-code": {
        "name": "Infrastructure as Code",
        "category": "system-design",
        "description": "Describe every environment declaratively, review it like code, and roll out changes gradually.",
        "patterns": ["one module per environment with shared variables", "state stored remotely with locking", "plan/apply with a human review", "immutable images over in-place mutation", "drift detection against the declared state"],
        "common_mistakes": ["hand edits that drift from the declared state", "secrets in the repository or the state file", "an apply that cannot be rolled back", "a single state file everyone writes to at once"],
        "interview_tricks": ["How do you promote this change from staging to production?", "What is your rollback if the apply half-succeeds?", "How do you prevent drift between environments?"],
        "complexity": "N/A — process-driven",
    },
    "kubernetes-orchestration": {
        "name": "Kubernetes Orchestration",
        "category": "system-design",
        "description": "Run stateless workloads as deployments behind a service, with probes, limits, and a rollout strategy that never drops the floor.",
        "patterns": ["deployment plus service for stateless tiers", "readiness vs liveness probes with distinct purposes", "CPU/memory requests and limits per container", "rolling update with maxUnavailable and maxSurge", "HPA on the metric that is actually saturated"],
        "common_mistakes": ["a liveness probe that restarts a slow-but-healthy pod", "no resource limits so one pod starves the node", "state in the container filesystem", "a single replica behind a PDB that allows zero"],
        "interview_tricks": ["Why did this pod get OOM-killed and how would you size it?", "What happens to in-flight requests during a rollout?", "How do you drain a node without dropping traffic?"],
        "complexity": "O(1) control loop per object; scheduling cost O(nodes) per pod",
    },
    "cloud-service-selection": {
        "name": "Cloud Service Selection",
        "category": "system-design",
        "description": "Pick the managed service that removes the most undifferentiated work, and state what you give up for it.",
        "patterns": ["managed relational for transactional state", "object storage plus a queue for async work", "a managed cache in front of a hot read", "serverless for spiky low-duty-cycle jobs", "a managed observability stack instead of self-hosted"],
        "common_mistakes": ["self-hosting a database to avoid a vendor", "ignoring egress and request costs", "choosing a queue with no DLQ or delay", "a vendor default that silently imposes a quota"],
        "interview_tricks": ["What does this managed service cost you in control?", "Which part of this pipeline would you keep self-hosted?", "What is the exit path if you need to leave this vendor?"],
        "complexity": "N/A — selection-driven",
    },
}

AI_STACK_SKILLS: dict[str, dict[str, Any]] = {
    "prompt-design": {
        "name": "Prompt Design",
        "category": "ai-stack",
        "description": "Write an instruction with an explicit contract, role, constraints, and output schema so the model's degrees of freedom are closed.",
        "patterns": ["state the task, the audience, and the output schema", "show one worked example of the required shape", "bound the answer length and allowed values", "put untrusted content in a marked block with an ignore-instructions rule", "iterate against a fixed eval set, not one lucky call"],
        "common_mistakes": ["a vague verb with no accept/reject criteria", "an output schema the validator then rejects", "instructions buried under untrusted text", "tuning the prompt against a single anecdote"],
        "interview_tricks": ["What makes this prompt deterministic, and what is still random?", "How would you handle a model that returns prose when JSON was required?", "What is your eval for this prompt and what is its pass bar?"],
        "complexity": "N/A — contract-driven",
    },
    "rag-retrieval": {
        "name": "RAG & Retrieval",
        "category": "ai-stack",
        "description": "Retrieve the right chunks, ground the answer in them, and make the failure modes (miss, wrong chunk, stale index) visible.",
        "patterns": ["chunk by semantic unit with overlap", "hybrid search: dense plus lexical", "rerank the top-k before answering", "cite the source chunk for each claim", "evaluate retrieval separately from generation"],
        "common_mistakes": ["chunking mid-sentence so evidence is split", "top-k with no relevance threshold, so a miss still answers", "no citation, so a hallucination is indistinguishable", "index that is never refreshed after a source edit"],
        "interview_tricks": ["What happens when no chunk is relevant — do you still answer?", "How do you measure retrieval quality without reading every result?", "How would you keep the index fresh and the cost bounded?"],
        "complexity": "O(log n) approximate search; cost dominated by rerank",
    },
    "agent-tool-use": {
        "name": "Agent & Tool Use",
        "category": "ai-stack",
        "description": "Give a model a small, typed tool set with a stop condition, and keep the loop bounded and auditable.",
        "patterns": ["bound the loop with a step and time budget", "typed tool schemas with narrow arguments", "validate arguments before the side effect", "one tool per intent, no grab-bag endpoint", "log every step for replay"],
        "common_mistakes": ["tools that overlap so the model picks randomly", "a side-effecting tool with no confirmation", "an unbounded loop that retries forever", "no trace, so a bad outcome cannot be explained"],
        "interview_tricks": ["When does this agent stop, and what does it do when it is wrong?", "How do you keep a tool call from reaching production by accident?", "What would you log to debug this run after the fact?"],
        "complexity": "O(steps x tool cost), bounded by an explicit budget",
    },
    "evaluation-metrics": {
        "name": "Evaluation & Metrics",
        "category": "ai-stack",
        "description": "Decide with an offline eval set and a rubric, then watch a live metric that can regress.",
        "patterns": ["a golden set that covers the failure classes you care about", "deterministic checks for schema and grounding", "LLM-as-judge with a rubric and a human spot check", "a held-out regression set per release", "track cost and latency beside quality"],
        "common_mistakes": ["judging with the same prompt you tuned on", "a single accuracy number that hides the bad class", "no baseline, so a change cannot be shown to help", "evaluating on the training examples"],
        "interview_tricks": ["How would you know this change made the system better?", "Which failure mode does your metric miss?", "What is the cheapest eval that would have caught this regression?"],
        "complexity": "N/A — measurement-driven",
    },
    "serving-cost-optimization": {
        "name": "Serving & Cost Optimization",
        "category": "ai-stack",
        "description": "Serve the model behind a path that keeps latency and spend predictable as traffic varies.",
        "patterns": ["route by task size to a cheaper model", "cache deterministic or repeated responses", "batch requests where latency allows", "cap the output token budget per call", "track cost per request and per user"],
        "common_mistakes": ["one model for every task", "no cap on context, so every call pays for a huge prompt", "ignoring the p95 latency of the fallback path", "cost dashboards that arrive a month late"],
        "interview_tricks": ["What is the cost per request and what dominates it?", "How do you keep p95 latency stable when traffic triples?", "Where would you cache without making answers stale?"],
        "complexity": "Latency O(computed tokens); cost O(input + output tokens)",
    },
    "guardrails-safety": {
        "name": "Guardrails & Safety",
        "category": "ai-stack",
        "description": "Constrain inputs, tool calls, and outputs, and fail closed when a constraint is violated.",
        "patterns": ["classify the request before it reaches the model", "validate and escape anything before execution", "redact PII and secrets from prompts and logs", "refuse with a fixed, tested message", "keep an audit trail of blocked requests"],
        "common_mistakes": ["filtering only the output, never the input", "a prompt-level rule with no enforcement", "secrets or PII in the trace log", "a jailbreak test that only checks one phrasing"],
        "interview_tricks": ["Where does an untrusted instruction stop being data?", "What happens if the classifier fails — does the system fail open?", "How would you test this guard against a novel attack?"],
        "complexity": "O(1) per request; cost is the classifier plus the base call",
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


def technical_skill_block(skill_ids: Iterable[str], track: str = "") -> str:
    """Render the technical block for ``track`` (DSA pack when unknown)."""
    return render_skill_block(
        "Technical & coding skills to apply and assess",
        skill_ids,
        track_technical_skills(track),
    )


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

#: Per-track technical skill packs, keyed by the canonical track id
#: used across the topic catalogue and interview sessions.
_TRACK_SKILLS: dict[str, dict[str, dict[str, Any]]] = {
    "backend": BACKEND_SKILLS,
    "frontend": FRONTEND_SKILLS,
    "system_design": SYSTEM_DESIGN_SKILLS,
    "ai_stack": AI_STACK_SKILLS,
}

#: Per-track technical skills by level, mirroring ``_LEVEL_TECHNICAL``.
#: A track that is absent here (or an unknown track) falls back to the
#: DSA pack, which the problem-solving topic depends on.
_TRACK_TECHNICAL: dict[str, dict[str, tuple[str, ...]]] = {
    "backend": {
        "junior": (
            "http-semantics", "api-contract-design",
            "data-modeling-persistence", "testing-strategy",
        ),
        "mid": (
            "api-contract-design", "data-modeling-persistence",
            "authn-authz", "concurrency-async", "caching-databases",
        ),
        "senior": (
            "concurrency-async", "authn-authz", "observability",
            "caching-databases", "data-modeling-persistence", "testing-strategy",
        ),
    },
    "frontend": {
        "junior": (
            "react-state-and-rendering", "component-architecture",
            "css-layout-systems", "accessibility",
        ),
        "mid": (
            "react-state-and-rendering", "component-architecture",
            "data-fetching-caching", "css-layout-systems", "accessibility",
        ),
        "senior": (
            "react-state-and-rendering", "performance-web-vitals",
            "typescript-type-design", "data-fetching-caching",
            "testing-ui", "accessibility",
        ),
    },
    "system_design": {
        "junior": (
            "load-estimation", "caching-strategies",
            "cloud-service-selection", "database-scaling",
        ),
        "mid": (
            "load-estimation", "caching-strategies", "database-scaling",
            "consistency-tradeoffs", "kubernetes-orchestration",
        ),
        "senior": (
            "consistency-tradeoffs", "database-scaling",
            "queues-backpressure", "kubernetes-orchestration",
            "infrastructure-as-code", "cloud-service-selection",
        ),
    },
    "ai_stack": {
        "junior": (
            "prompt-design", "evaluation-metrics", "guardrails-safety",
        ),
        "mid": (
            "prompt-design", "rag-retrieval", "evaluation-metrics",
            "agent-tool-use", "guardrails-safety",
        ),
        "senior": (
            "agent-tool-use", "rag-retrieval", "evaluation-metrics",
            "serving-cost-optimization", "guardrails-safety",
        ),
    },
}


def normalise_track(track: str | None) -> str:
    """Canonicalise a track id, or ``""`` when it names no known pack.

    Mirrors the topic catalogue's normalisation (case, ``-``/``_``,
    whitespace) so a track that arrives as ``"System-Design"`` or
    ``"ai stack"`` still selects its pack.
    """
    raw = str(track or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw if raw in _TRACK_SKILLS else ""


def track_technical_skills(track: str | None) -> dict[str, dict[str, Any]]:
    """Return the technical pack for ``track`` (DSA pack when unknown)."""
    return _TRACK_SKILLS.get(normalise_track(track), TECHNICAL_SKILLS)


def _level_key(level: str) -> str:
    level_key = str(level or "").strip().lower()
    if level_key not in _LEVEL_TECHNICAL:
        level_key = "mid"
    return level_key


def skills_for_session(
    *,
    level: str = "mid",
    session_type: str = "general",
    track: str = "",
    include_educational: bool = True,
) -> dict[str, list[str]]:
    """Select the skill ids relevant to a session's level and type.

    ``track`` selects the per-track technical pack (backend, frontend,
    system design, AI stack). An empty or unknown track keeps the DSA
    selection, which the problem-solving topic relies on.
    """
    level_key = _level_key(level)
    session_key = str(session_type or "").strip().lower()
    if session_key not in _SESSION_INTERVIEW:
        session_key = "general"
    track_levels = _TRACK_TECHNICAL.get(normalise_track(track))
    technical = list(track_levels[level_key]) if track_levels else list(_LEVEL_TECHNICAL[level_key])
    result: dict[str, list[str]] = {
        "technical": technical,
        "interview": list(_SESSION_INTERVIEW[session_key]),
    }
    if include_educational:
        result["educational"] = list(_CORE_EDUCATIONAL)
    return result


def skill_context_block(
    *,
    level: str = "mid",
    session_type: str = "general",
    track: str = "",
    include_educational: bool = True,
) -> str:
    """Render the full skill context for a session as one fenced block."""
    selected = skills_for_session(
        level=level,
        session_type=session_type,
        track=track,
        include_educational=include_educational,
    )
    parts = [
        technical_skill_block(selected["technical"], track),
        interview_skill_block(selected["interview"]),
    ]
    if include_educational:
        parts.append(educational_skill_block(selected["educational"]))
    return "\n\n".join(part for part in parts if part)
