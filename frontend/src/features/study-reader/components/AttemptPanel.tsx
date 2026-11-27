import { useRef, useState } from "react";
import type { AttemptOutcome } from "@/types";
import { canReveal, OUTCOME_GRADES, REVIEW_GRADES, type ModePolicy } from "../questionModes";

interface AttemptPanelProps {
  policy: ModePolicy;
  /** Current server-side completion flag for this slot. */
  completed: boolean;
  /** Whether a progress mutation is in flight. */
  submitting: boolean;
  /** Last submission error, if any. */
  submitError: string | null;
  onCompletionToggle: (completed: boolean) => void;
  /** Returns whether the attempt was recorded; grading resets on failure. */
  onGrade: (outcome: AttemptOutcome, confidence: number, responseTimeMs: number) => Promise<boolean>;
  /** Notifies the page when gated content may be shown. */
  onRevealChange: (revealed: boolean) => void;
}

/**
 * Mode-specific attempt and grading panel (issue #13). Attempt state is
 * local to this panel instance — the page keys it by slot id and mode, so
 * attempts never leak across questions or modes.
 */
export default function AttemptPanel({
  policy,
  completed,
  submitting,
  submitError,
  onCompletionToggle,
  onGrade,
  onRevealChange,
}: AttemptPanelProps) {
  const [attemptText, setAttemptText] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [graded, setGraded] = useState(false);
  const startedAt = useRef(Date.now());

  const elapsedMs = () => Math.min(Math.max(Date.now() - startedAt.current, 0), 3_600_000);

  const revealNow = () => {
    setRevealed(true);
    onRevealChange(true);
  };

  if (policy.completionToggle) {
    return (
      <section aria-label="Study completion" className="mt-6 border-t border-udemy-border pt-4">
        <button
          type="button"
          onClick={() => onCompletionToggle(!completed)}
          disabled={submitting}
          aria-pressed={completed}
          className="btn-primary disabled:opacity-50"
        >
          {completed ? "Completed — undo" : "Mark as complete"}
        </button>
        {submitError && (
          <p role="alert" className="mt-2 text-sm text-udemy-danger">
            {submitError}
          </p>
        )}
      </section>
    );
  }

  const revealAllowed = canReveal(policy, attemptText.trim().length > 0, submitted);

  return (
    <section aria-label={`${policy.label} attempt`} className="mt-6 border-t border-udemy-border pt-4">
      <label className="block">
        <span className="text-sm font-medium text-udemy-text">
          {policy.requireSubmissionBeforeReveal ? "Your response" : "Your attempt"}
        </span>
        <textarea
          value={attemptText}
          onChange={(event) => setAttemptText(event.target.value)}
          rows={policy.requireSubmissionBeforeReveal ? 6 : 3}
          disabled={submitted || graded}
          className="mt-1 w-full rounded-md border border-udemy-border p-2 text-sm text-udemy-text"
          placeholder={
            policy.requireSubmissionBeforeReveal
              ? "Answer as if you were in the interview…"
              : "Write what you remember before revealing…"
          }
        />
      </label>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {policy.requireSubmissionBeforeReveal && !submitted && (
          <button
            type="button"
            onClick={() => setSubmitted(true)}
            disabled={attemptText.trim().length === 0}
            className="btn-primary disabled:opacity-50"
          >
            Submit response
          </button>
        )}
        {!policy.requireSubmissionBeforeReveal && !revealed && (
          <button
            type="button"
            onClick={revealNow}
            disabled={!revealAllowed}
            className="btn-primary disabled:opacity-50"
          >
            Reveal answer
          </button>
        )}
        {policy.requireSubmissionBeforeReveal && submitted && !revealed && (
          <button type="button" onClick={revealNow} className="btn-primary">
            Reveal answer
          </button>
        )}
      </div>

      {revealed && !graded && policy.selfGrading !== "none" && (
        <div role="group" aria-label="Grade your answer" className="mt-3 flex flex-wrap gap-2">
          {(policy.selfGrading === "fsrs" ? REVIEW_GRADES : OUTCOME_GRADES).map((grade) => (
            <button
              key={grade.id}
              type="button"
              onClick={() => {
                void (async () => {
                  const ok = await onGrade(grade.outcome, grade.confidence, elapsedMs());
                  if (ok) setGraded(true);
                })();
              }}
              disabled={submitting}
              className="rounded-md border border-udemy-border px-3 py-1.5 text-sm hover:bg-udemy-bg disabled:opacity-50"
            >
              {grade.label}
            </button>
          ))}
        </div>
      )}

      {graded && !submitError && (
        <p className="mt-3 text-sm text-udemy-success" role="status">
          Attempt recorded.
        </p>
      )}
      {submitError && (
        <p role="alert" className="mt-2 text-sm text-udemy-danger">
          {submitError}
        </p>
      )}
    </section>
  );
}
