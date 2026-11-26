import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import axios from "axios";
import {
  AlertTriangle,
  Check,
  CloudOff,
  Loader2,
  Maximize2,
  Minimize2,
  Minus,
  MoveHorizontal,
  Plus,
} from "lucide-react";
import {
  fetchJourneyProgress,
  patchJourneyCheckpoint,
  recordQuestionAttempt,
  setQuestionCompletion,
} from "@/services/api";
import { useApi } from "@/hooks/useApi";
import { useAuthStore } from "@/store/authStore";
import type {
  AttemptOutcome,
  JourneyDetail,
  JourneyModule,
  JourneyObjective,
  JourneyPhase,
  JourneyQuestionSlot,
} from "@/types";
import {
  getReaderData,
  invalidateReaderData,
  updateCachedCheckpoint,
  type ReaderData,
} from "../readerData";
import {
  createCheckpointSaver,
  type CheckpointSaver,
  type CheckpointSaveStatus,
} from "../checkpointSaver";
import AttemptPanel from "../components/AttemptPanel";
import { MODE_POLICIES, parseReaderMode, READER_MODES } from "../questionModes";
import { nextFontScale, useReaderUiStore } from "../store/readerUiStore";
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

function newMutationId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `mutation-${Date.now()}-${Math.random().toString(36).slice(2)}`;
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

/**
 * Study reader page (issues #7, #8, #11).
 *
 * State ownership contract: the URL owns the selected
 * journey/batch/question; the reader-data cache (per-user scoped,
 * TTL-bound, abort-aware via useApi) owns server payloads; Zustand
 * holds only harmless UI preferences (sidebar, focus mode, font
 * scale, content width). No question payload is mirrored into a
 * global store.
 */
export default function StudyReaderPage() {
  const { journeyId, questionId } = useParams<{
    journeyId: string;
    batchId: string;
    questionId: string;
  }>();
  const navigate = useNavigate();
  const userKey = useAuthStore((s) => s.user?.user ?? "anonymous");

  const sidebarOpen = useReaderUiStore(
    (s) => s.prefsByUser[userKey]?.sidebarOpen ?? true,
  );
  const focusMode = useReaderUiStore(
    (s) => s.prefsByUser[userKey]?.focusMode ?? false,
  );
  const fontScale = useReaderUiStore(
    (s) => s.prefsByUser[userKey]?.fontScale ?? 1,
  );
  const contentWidth = useReaderUiStore(
    (s) => s.prefsByUser[userKey]?.contentWidth ?? "normal",
  );
  const setReaderPrefs = useReaderUiStore((s) => s.setReaderPrefs);

  const [mobileRailOpen, setMobileRailOpen] = useState(false);
  const [saveStatus, setSaveStatus] = useState<CheckpointSaveStatus>("idle");
  const [searchParams, setSearchParams] = useSearchParams();
  const mode = parseReaderMode(searchParams.get("mode"));
  const policy = MODE_POLICIES[mode];
  const [progressSubmitting, setProgressSubmitting] = useState(false);
  const [progressError, setProgressError] = useState<string | null>(null);
  // Reveal state is per slot+mode so answers never leak across questions.
  const [revealedByKey, setRevealedByKey] = useState<Record<string, boolean>>({});
  const saverRef = useRef<CheckpointSaver | null>(null);
  const revisionRef = useRef(0);

  const fetcher = useCallback(
    async (signal: AbortSignal): Promise<ReaderData> => {
      if (!journeyId) throw new Error("No journey was selected.");
      try {
        return await getReaderData(journeyId, userKey, signal);
      } catch (err) {
        if (axios.isAxiosError(err) && err.response?.status === 404) {
          throw new Error("This journey could not be found.");
        }
        throw err;
      }
    },
    [journeyId, userKey],
  );

  const { data, error, loading, retry, setData } = useApi<ReaderData>(fetcher, {
    deps: [journeyId, userKey],
    errorMessage: "Check your connection and try again.",
  });

  const journey = data?.journey ?? null;
  const readerState = data?.readerState ?? null;

  const progressFetcher = useCallback(
    (signal: AbortSignal) => {
      if (!journeyId) return Promise.reject(new Error("No journey selected."));
      return fetchJourneyProgress(journeyId, signal);
    },
    [journeyId],
  );
  const { data: progress, setData: setProgress } = useApi(progressFetcher, {
    deps: [journeyId],
    immediate: Boolean(journeyId),
    errorMessage: "Progress could not be loaded.",
  });

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

  // Track the server revision for optimistic concurrency.
  useEffect(() => {
    if (readerState) revisionRef.current = readerState.checkpoint.revision;
  }, [readerState]);

  // Checkpoint saver lifecycle (issue #12): debounced position autosave,
  // durable offline queue, 409 reconcile, pagehide best-effort beacon.
  useEffect(() => {
    if (!journeyId) return;
    const saver = createCheckpointSaver({
      journeyId,
      userKey,
      getRevision: () => revisionRef.current,
      patch: patchJourneyCheckpoint,
      fetchState: async () => {
        invalidateReaderData(journeyId);
        const fresh = await getReaderData(journeyId, userKey);
        return fresh.readerState;
      },
      onCheckpoint: (checkpoint) => {
        updateCachedCheckpoint(journeyId, checkpoint);
        revisionRef.current = checkpoint.revision;
        setData((current) =>
          current
            ? {
                ...current,
                readerState: { ...current.readerState, checkpoint },
              }
            : current,
        );
      },
      onStatus: setSaveStatus,
      debounceMs: 1500,
    });
    saverRef.current = saver;
    const onPageHide = () => saver.flushBestEffort();
    window.addEventListener("pagehide", onPageHide);
    return () => {
      window.removeEventListener("pagehide", onPageHide);
      saver.destroy();
      saverRef.current = null;
    };
  }, [journeyId, userKey, setData]);

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
        (target.closest(
          "input, textarea, select, [contenteditable]:not([contenteditable=false]), pre, code",
        ) ||
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

  // Schedule a checkpoint autosave when the active question changes;
  // the saver debounces, queues offline and reconciles 409s (#12).
  useEffect(() => {
    if (!journey || !readerState || !activeSlot) return;
    if (readerState.checkpoint.last_question_id === activeSlot.id) return;
    saverRef.current?.position({
      last_batch_id: activeSlot.batch_id,
      last_question_id: activeSlot.id,
      scroll_offset: 0,
    });
  }, [journey, readerState, activeSlot]);

  const setMode = (next: string) => {
    const params = new URLSearchParams(searchParams);
    if (next === "study") {
      params.delete("mode");
    } else {
      params.set("mode", next);
    }
    setSearchParams(params, { replace: true });
  };

  const activeState = activeSlot
    ? progress?.states.find((state) => state.slot_id === activeSlot.id) ?? null
    : null;

  const handleCompletionToggle = async (completed: boolean) => {
    if (!journey || !activeSlot || progressSubmitting) return;
    setProgressSubmitting(true);
    setProgressError(null);
    try {
      const response = await setQuestionCompletion(journey.id, activeSlot.id, {
        completed,
        mutation_id: newMutationId(),
      });
      setProgress(response.summary);
    } catch {
      setProgressError("The completion update did not save. Try again.");
    } finally {
      setProgressSubmitting(false);
    }
  };

  const handleGrade = async (
    outcome: AttemptOutcome,
    confidence: number,
    responseTimeMs: number,
  ): Promise<boolean> => {
    if (!journey || !activeSlot || progressSubmitting) return false;
    setProgressSubmitting(true);
    setProgressError(null);
    try {
      const response = await recordQuestionAttempt(journey.id, activeSlot.id, {
        outcome,
        confidence,
        response_time_ms: responseTimeMs,
        mutation_id: newMutationId(),
      });
      setProgress(response.summary);
      return true;
    } catch {
      setProgressError("The attempt did not save. Try again.");
      return false;
    } finally {
      setProgressSubmitting(false);
    }
  };

  const revealKey = activeSlot ? `${mode}:${activeSlot.id}` : "";
  const contentVisible =
    policy.showContentImmediately || (revealKey !== "" && revealedByKey[revealKey] === true);

  const announcement = activeSlot
    ? `Question ${slots.findIndex((s) => s.id === activeSlot.id) + 1} of ${slots.length}: ${activeSlot.objective_title}`
    : "";

  if (loading) {
    return <ReaderSkeleton />;
  }
  if (error || !journey || !readerState) {
    return (
      <ReaderErrorState
        message={error || "The reader could not load."}
        onRetry={() => void retry()}
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
  const railVisible = sidebarOpen && !focusMode;

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
          >
            <div role="group" aria-label="Reader mode" className="flex items-center gap-1">
              {READER_MODES.map((readerMode) => (
                <button
                  key={readerMode}
                  type="button"
                  onClick={() => setMode(readerMode)}
                  aria-pressed={mode === readerMode}
                  className={`rounded-md px-2 py-1 text-xs font-medium ${
                    mode === readerMode
                      ? "bg-udemy-purple text-white"
                      : "text-udemy-text-muted hover:bg-udemy-bg"
                  }`}
                >
                  {MODE_POLICIES[readerMode].label}
                </button>
              ))}
            </div>
            <span
              role="status"
              aria-label="Save status"
              aria-live="polite"
              className="flex items-center gap-1 text-xs text-udemy-text-muted"
            >
              {(saveStatus === "pending" || saveStatus === "saving") && (
                <>
                  <Loader2 className="w-3.5 h-3.5 motion-safe:animate-spin" aria-hidden="true" />
                  Saving…
                </>
              )}
              {saveStatus === "saved" && (
                <>
                  <Check className="w-3.5 h-3.5 text-udemy-success" aria-hidden="true" />
                  Saved
                </>
              )}
              {saveStatus === "offline" && (
                <>
                  <CloudOff className="w-3.5 h-3.5" aria-hidden="true" />
                  Offline — will retry
                </>
              )}
              {saveStatus === "error" && (
                <>
                  <AlertTriangle className="w-3.5 h-3.5 text-udemy-warning" aria-hidden="true" />
                  Save failed — will retry
                </>
              )}
            </span>
            <button
              type="button"
              onClick={() =>
                setReaderPrefs(userKey, { fontScale: nextFontScale(fontScale, -1) })
              }
              aria-label="Decrease font size"
              className="p-2 rounded-md hover:bg-udemy-bg"
            >
              <Minus className="w-4 h-4" aria-hidden="true" />
            </button>
            <button
              type="button"
              onClick={() =>
                setReaderPrefs(userKey, { fontScale: nextFontScale(fontScale, 1) })
              }
              aria-label="Increase font size"
              className="p-2 rounded-md hover:bg-udemy-bg"
            >
              <Plus className="w-4 h-4" aria-hidden="true" />
            </button>
            <button
              type="button"
              onClick={() =>
                setReaderPrefs(userKey, {
                  contentWidth: contentWidth === "normal" ? "wide" : "normal",
                })
              }
              aria-label="Toggle wide layout"
              aria-pressed={contentWidth === "wide"}
              className="p-2 rounded-md hover:bg-udemy-bg"
            >
              <MoveHorizontal className="w-4 h-4" aria-hidden="true" />
            </button>
            <button
              type="button"
              onClick={() => setReaderPrefs(userKey, { focusMode: !focusMode })}
              aria-label={focusMode ? "Exit focus mode" : "Enter focus mode"}
              aria-pressed={focusMode}
              className="p-2 rounded-md hover:bg-udemy-bg"
            >
              {focusMode ? (
                <Minimize2 className="w-4 h-4" aria-hidden="true" />
              ) : (
                <Maximize2 className="w-4 h-4" aria-hidden="true" />
              )}
            </button>
          </ReaderToolbar>
        ) : undefined
      }
      railOpen={railVisible}
      onToggleRail={() => setReaderPrefs(userKey, { sidebarOpen: !sidebarOpen })}
      mobileRailOpen={mobileRailOpen}
      onToggleMobileRail={() => setMobileRailOpen((open) => !open)}
      onScroll={(event) => {
        if (activeSlot) {
          saverRef.current?.position({
            last_batch_id: activeSlot.batch_id,
            last_question_id: activeSlot.id,
            scroll_offset: Math.round(event.currentTarget.scrollTop),
          });
        }
      }}
    >
      <div aria-live="polite" role="status" aria-label="Current question" className="sr-only">
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
        <article
          aria-label="Question overview"
          style={{ zoom: fontScale }}
          className={`mx-auto p-6 ${contentWidth === "wide" ? "max-w-5xl" : "max-w-3xl"}`}
        >
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
                {contentVisible
                  ? location.objective.source_section || "Custom"
                  : "Hidden until you reveal"}
              </dd>
            </div>
            <div className="rounded-lg border border-udemy-border p-3">
              <dt className="text-xs uppercase text-udemy-text-muted">Question</dt>
              <dd className="mt-1 text-sm font-medium text-udemy-text">
                {activeIndex + 1} of {slots.length}
              </dd>
            </div>
          </dl>
          <AttemptPanel
            key={`${mode}:${activeSlot.id}`}
            policy={policy}
            completed={activeState?.completed ?? false}
            submitting={progressSubmitting}
            submitError={progressError}
            onCompletionToggle={(completed) => void handleCompletionToggle(completed)}
            onGrade={handleGrade}
            onRevealChange={(revealed) => {
              if (revealed) setRevealedByKey((current) => ({ ...current, [revealKey]: true }));
            }}
          />
        </article>
      ) : null}
    </ReaderShell>
  );
}
