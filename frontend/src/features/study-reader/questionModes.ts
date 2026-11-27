import type { AttemptOutcome, ReaderMode } from "@/types";

/**
 * Unified mode policies for question presentation (issue #13).
 *
 * The mode travels in the URL query string (?mode=) so deep links and
 * back/forward preserve it. Policies are pure data so the page, the
 * attempt panel and tests all read the same rules:
 *
 * - study: rich content visible immediately; explicit completion toggle.
 * - practice: an attempt is required before reveal; outcome self-grade.
 * - review: attempt + FSRS-style grade (Again/Hard/Good/Easy) mapped to
 *   outcome + confidence for the journey attempts API. FSRS scheduling
 *   itself stays with the existing review backend.
 * - interview: a free response must be submitted before reveal; the
 *   learner then self-grades (LLM evaluation is out of scope here).
 */

export interface ModePolicy {
  mode: ReaderMode;
  label: string;
  /** Content and source sections are visible without an attempt. */
  showContentImmediately: boolean;
  /** Reveal is disabled until the learner has written an attempt. */
  requireAttemptBeforeReveal: boolean;
  /** Reveal additionally requires submitting the attempt (interview). */
  requireSubmissionBeforeReveal: boolean;
  /** Grading offered after reveal. */
  selfGrading: "none" | "outcome" | "fsrs";
  /** Explicit completion toggle available. */
  completionToggle: boolean;
}

export const MODE_POLICIES: Record<ReaderMode, ModePolicy> = {
  study: {
    mode: "study",
    label: "Study",
    showContentImmediately: true,
    requireAttemptBeforeReveal: false,
    requireSubmissionBeforeReveal: false,
    selfGrading: "none",
    completionToggle: true,
  },
  practice: {
    mode: "practice",
    label: "Practice",
    showContentImmediately: false,
    requireAttemptBeforeReveal: true,
    requireSubmissionBeforeReveal: false,
    selfGrading: "outcome",
    completionToggle: false,
  },
  review: {
    mode: "review",
    label: "Review",
    showContentImmediately: false,
    requireAttemptBeforeReveal: true,
    requireSubmissionBeforeReveal: false,
    selfGrading: "fsrs",
    completionToggle: false,
  },
  interview: {
    mode: "interview",
    label: "Interview",
    showContentImmediately: false,
    requireAttemptBeforeReveal: true,
    requireSubmissionBeforeReveal: true,
    selfGrading: "outcome",
    completionToggle: false,
  },
};

export const READER_MODES: ReaderMode[] = ["study", "practice", "review", "interview"];

export function parseReaderMode(raw: string | null | undefined): ReaderMode {
  return READER_MODES.find((mode) => mode === raw) ?? "study";
}

export interface ReviewGrade {
  id: string;
  label: string;
  outcome: AttemptOutcome;
  confidence: number;
}

/** FSRS-style four-point grade mapped onto the attempts API contract. */
export const REVIEW_GRADES: ReviewGrade[] = [
  { id: "again", label: "Again", outcome: "incorrect", confidence: 1 },
  { id: "hard", label: "Hard", outcome: "correct", confidence: 2 },
  { id: "good", label: "Good", outcome: "correct", confidence: 4 },
  { id: "easy", label: "Easy", outcome: "correct", confidence: 5 },
];

export interface OutcomeGrade {
  id: AttemptOutcome;
  label: string;
  outcome: AttemptOutcome;
  confidence: number;
}

/** Plain three-way self-grade for practice and interview modes. */
export const OUTCOME_GRADES: OutcomeGrade[] = [
  { id: "correct", label: "I got it", outcome: "correct", confidence: 4 },
  { id: "incorrect", label: "I missed it", outcome: "incorrect", confidence: 1 },
  { id: "skipped", label: "Skip", outcome: "skipped", confidence: 0 },
];

/** A reveal request is allowed when the policy's preconditions are met. */
export function canReveal(policy: ModePolicy, hasAttempt: boolean, submitted: boolean): boolean {
  if (policy.showContentImmediately) return true;
  if (policy.requireSubmissionBeforeReveal && !submitted) return false;
  if (policy.requireAttemptBeforeReveal && !hasAttempt) return false;
  return true;
}
