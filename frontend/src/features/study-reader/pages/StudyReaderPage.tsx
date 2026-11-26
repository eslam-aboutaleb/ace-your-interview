import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import axios from "axios";
import {
  fetchJourney,
  fetchReaderState,
  patchJourneyCheckpoint,
} from "@/services/api";
import type {
  JourneyDetail,
  JourneyModule,
  JourneyObjective,
  JourneyPhase,
  JourneyQuestionSlot,
  ReaderStateResponse,
} from "@/types";
import CurriculumRail from "../components/CurriculumRail";
import ReaderBreadcrumbs from "../components/ReaderBreadcrumbs";
import ReaderErrorState from "../components/ReaderErrorState";
import ReaderShell from "../components/ReaderShell";
import ReaderSkeleton from "../components/ReaderSkeleton";
import ReaderToolbar from "../components/ReaderToolbar";

interface ObjectiveLocation {
  objective: JourneyObjective;
  module: JourneyModule;
  phase: JourneyPhase;
}

function findObjective(journey: JourneyDetail, objectiveId: string): ObjectiveLocation | null {
  for (const phase of journey.phases) {
    for (const module of phase.modules) {
      const objective = module.objectives.find((o) => o.id === objectiveId);
      if (objective) return { objective, module, phase };
    }
  }
  return null;
}

function newMutationId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `mutation-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/**
 * Study reader page (issue #7). The URL owns position:
 * /journeys/:journeyId/batches/:batchId/questions/:questionId. On entry
 * without a question id, the checkpoint restores the last position.
 *
 * Scope note: this slice ships the shell, navigation, and checkpoint
 * restore. Question content rendering lands with #9/#10, so the center
 * pane surfaces the objective's plan data, not a placeholder.
 */
export default function StudyReaderPage() {
  const { journeyId, questionId } = useParams<{
    journeyId: string;
    batchId: string;
    questionId: string;
  }>();
  const navigate = useNavigate();

  const [journey, setJourney] = useState<JourneyDetail | null>(null);
  const [readerState, setReaderState] = useState<ReaderStateResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [railOpen, setRailOpen] = useState(true);
  const [mobileRailOpen, setMobileRailOpen] = useState(false);
  const lastPatchedQuestion = useRef<string | null>(null);

  const load = useCallback(async () => {
    if (!journeyId) {
      setError("No journey was selected.");
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const [journeyData, stateData] = await Promise.all([
        fetchJourney(journeyId),
        fetchReaderState(journeyId),
      ]);
      setJourney(journeyData);
      setReaderState(stateData);
    } catch (err) {
      const message = axios.isAxiosError(err) && err.response?.status === 404
        ? "This journey could not be found."
        : "Check your connection and try again.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [journeyId]);

  useEffect(() => {
    void load();
  }, [load]);

  const slots = useMemo<JourneyQuestionSlot[]>(() => {
    if (!readerState) return [];
    return readerState.questions
      .filter((slot) => slot.status === "accepted")
      .sort((a, b) => a.position - b.position);
  }, [readerState]);

  const activeSlot = useMemo(
    () => slots.find((slot) => slot.id === questionId) ?? null,
    [slots, questionId],
  );

  // URL owns position: when it is missing or stale, restore from the
  // checkpoint (or fall back to the first accepted slot) and replace.
  useEffect(() => {
    if (!journey || !readerState || slots.length === 0 || activeSlot) return;
    const checkpointQuestionId = readerState.checkpoint.last_question_id;
    const restoreSlot =
      slots.find((slot) => slot.id === checkpointQuestionId) ?? slots[0];
    navigate(
      `/journeys/${journey.id}/batches/${restoreSlot.batch_id}/questions/${restoreSlot.id}`,
      { replace: true },
    );
  }, [journey, readerState, slots, activeSlot, navigate]);

  // Keyboard-safe prev/next (issue #8): plain arrow keys turn pages, but
  // never while the user is typing, editing, or selecting text, and never
  // with modifier keys held.
  const slotsRef = useRef(slots);
  slotsRef.current = slots;
  const activeSlotRef = useRef(activeSlot);
  activeSlotRef.current = activeSlot;
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.closest("input, textarea, select, [contenteditable]:not([contenteditable=false]), pre, code") ||
          target.isContentEditable)
      ) {
        return;
      }
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed) return;
      const current = activeSlotRef.current;
      if (!current) return;
      const list = slotsRef.current;
      const index = list.findIndex((slot) => slot.id === current.id);
      const next = event.key === "ArrowLeft" ? list[index - 1] : list[index + 1];
      if (!next) return;
      event.preventDefault();
      navigate(
        `/journeys/${journey?.id}/batches/${next.batch_id}/questions/${next.id}`,
      );
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [navigate, journey?.id]);

  // Persist the checkpoint when the active question changes (optimistic
  // concurrency: the server rejects stale revisions with 409, in which
  // case we re-sync from the server rather than clobbering it).
  useEffect(() => {
    if (!journey || !readerState || !activeSlot) return;
    if (readerState.checkpoint.last_question_id === activeSlot.id) return;
    if (lastPatchedQuestion.current === activeSlot.id) return;
    lastPatchedQuestion.current = activeSlot.id;
    const revision = readerState.checkpoint.revision;
    void (async () => {
      try {
        const response = await patchJourneyCheckpoint(journey.id, {
          expected_revision: revision,
          mutation_id: newMutationId(),
          last_batch_id: activeSlot.batch_id,
          last_question_id: activeSlot.id,
        });
        setReaderState((current) =>
          current ? { ...current, checkpoint: response.checkpoint } : current,
        );
      } catch (err) {
        if (axios.isAxiosError(err) && err.response?.status === 409) {
          // Another client moved first; re-sync instead of clobbering.
          try {
            const fresh = await fetchReaderState(journey.id);
            setReaderState(fresh);
          } catch {
            // leave the current state; the next navigation retries
          }
        }
      }
    })();
  }, [journey, readerState, activeSlot]);

  const announcement = activeSlot
    ? `Question ${slots.findIndex((s) => s.id === activeSlot.id) + 1} of ${slots.length}: ${activeSlot.objective_title}`
    : "";

  if (loading) {
    return <ReaderSkeleton />;
  }
  if (error || !journey || !readerState) {
    return (
      <ReaderErrorState
        message={error ?? "The reader could not load."}
        onRetry={() => void load()}
      />
    );
  }

  const activeIndex = activeSlot ? slots.findIndex((s) => s.id === activeSlot.id) : -1;
  const goToSlot = (slot: JourneyQuestionSlot) => {
    navigate(`/journeys/${journey.id}/batches/${slot.batch_id}/questions/${slot.id}`);
    setMobileRailOpen(false);
  };

  const batch = readerState.latest_batch;
  const location = activeSlot ? findObjective(journey, activeSlot.objective_id) : null;

  return (
    <ReaderShell
      rail={
        <CurriculumRail
          journey={journey}
          slots={slots}
          activeQuestionId={activeSlot?.id ?? null}
        />
      }
      breadcrumbs={
        <ReaderBreadcrumbs
          items={[
            { label: journey.title },
            ...(batch ? [{ label: `Batch ${batch.batch_number}` }] : []),
            ...(activeSlot ? [{ label: activeSlot.objective_title }] : []),
          ]}
        />
      }
      toolbar={
        activeSlot ? (
          <ReaderToolbar
            title={activeSlot.objective_title}
            hasPrev={activeIndex > 0}
            hasNext={activeIndex >= 0 && activeIndex < slots.length - 1}
            onPrev={() => goToSlot(slots[activeIndex - 1])}
            onNext={() => goToSlot(slots[activeIndex + 1])}
          />
        ) : undefined
      }
      railOpen={railOpen}
      onToggleRail={() => setRailOpen((open) => !open)}
      mobileRailOpen={mobileRailOpen}
      onToggleMobileRail={() => setMobileRailOpen((open) => !open)}
    >
      <div aria-live="polite" role="status" className="sr-only">
        {announcement}
      </div>
      {slots.length === 0 ? (
        <div className="flex h-full flex-col items-center justify-center gap-2 p-8 text-center">
          <h1 className="text-udemy-text font-medium">No questions generated yet.</h1>
          <p className="max-w-md text-sm text-udemy-text-muted">
            This journey has a plan, but its first batch of questions has not
            been generated. Question generation ships behind its own feature
            flag with the generation pipeline.
          </p>
        </div>
      ) : activeSlot && location ? (
        <article aria-label="Question overview" className="mx-auto max-w-3xl p-6">
          <p className="text-xs font-semibold uppercase tracking-wide text-udemy-text-muted">
            {location.phase.title} / {location.module.title}
          </p>
          <h1 className="mt-1 text-2xl font-bold text-udemy-text">
            {location.objective.title}
          </h1>
          <dl className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="rounded-lg border border-udemy-border p-3">
              <dt className="text-xs uppercase text-udemy-text-muted">Objective type</dt>
              <dd className="mt-1 text-sm font-medium text-udemy-text">
                {location.objective.kind}
              </dd>
            </div>
            <div className="rounded-lg border border-udemy-border p-3">
              <dt className="text-xs uppercase text-udemy-text-muted">Source section</dt>
              <dd className="mt-1 text-sm font-medium text-udemy-text">
                {location.objective.source_section || "Custom"}
              </dd>
            </div>
            <div className="rounded-lg border border-udemy-border p-3">
              <dt className="text-xs uppercase text-udemy-text-muted">Question</dt>
              <dd className="mt-1 text-sm font-medium text-udemy-text">
                {activeIndex + 1} of {slots.length}
              </dd>
            </div>
          </dl>
        </article>
      ) : null}
    </ReaderShell>
  );
}
