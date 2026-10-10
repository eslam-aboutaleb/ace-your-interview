import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  Lightbulb,
  Loader2,
  Mic,
  MicOff,
  Radio,
  Send,
  SkipForward,
  FileText,
  Volume2,
} from "lucide-react";
import { pageVariants, pageTransition } from "@/utils/animations";
import {
  fetchHint,
  fetchInterviewSession,
  generateNextInterviewQuestion,
  generateNextInterviewQuestionStream,
  isInterviewStreamError,
  submitInterviewAnswer,
  submitInterviewAnswerStream,
} from "@/services/api";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import { useSettingsStore } from "@/store/settingsStore";
import { useInterviewSessionStore } from "@/store/interviewSessionStore";
import { useStreamGeneration } from "@/hooks/useStreamGeneration";
import { normalizeEscapedMultilineText } from "@/utils/textNormalization";
import { useVoice } from "@/hooks/useVoice";
import type {
  InterviewQuestionResponse,
  InterviewQuestionStreamDoneEvent,
  InterviewTurnResponse,
  InterviewTurnStreamDoneEvent,
} from "@/types";

function questionIdFor(question: string): string {
  let hash = 0;
  for (let i = 0; i < question.length; i += 1) {
    hash = (hash * 31 + question.charCodeAt(i)) | 0;
  }
  return `iq-${Math.abs(hash).toString(36)}`;
}

/**
 * A stream error may fall back to the non-streaming
 * endpoint unless it explicitly opted out
 * (`fallbackEligible === false`). Any other error
 * (network, abort) also falls back, matching the
 * original recovery semantics.
 */
function interviewStreamFallbackEligible(error: unknown): boolean {
  return !(isInterviewStreamError(error) && !error.fallbackEligible);
}

export default function InterviewSessionPage() {
  const { sessionId = "" } = useParams();
  const navigate = useNavigate();
  const settings = useSettingsStore();
  const store = useInterviewSessionStore();
  const {
    session,
    loading,
    error,
    streamProgress,
    loadingNext,
    submitting,
    startedAt,
    lastTurn,
    hintLevel,
    hintResult,
    hintLoading,
    hintError,
    hintLevelUsed,
  } = store;

  // Purely local UI state: the answer draft.
  const [answerDraft, setAnswerDraft] = useState("");

  const nextQuestionStream = useStreamGeneration<
    InterviewQuestionStreamDoneEvent,
    InterviewQuestionResponse
  >();
  const submitAnswerStream = useStreamGeneration<
    InterviewTurnStreamDoneEvent,
    InterviewTurnResponse
  >();

  const questionHeadingRef = useRef<HTMLHeadingElement>(null);
  const firstQuestionRef = useRef(true);

  // Voice integration
  const voice = useVoice();

  // Reset the session store when leaving the page so a
  // later visit starts fresh (mirrors local useState).
  useEffect(() => {
    return () => {
      useInterviewSessionStore.getState().reset();
    };
  }, []);

  // Reset hint state when the current question changes
  useEffect(() => {
    store.resetHintState();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.session.current_question]);

  // Move focus to newly generated questions (skip the
  // initial load so the page title keeps focus).
  useEffect(() => {
    if (!session?.session.current_question) return;
    if (firstQuestionRef.current) {
      firstQuestionRef.current = false;
      return;
    }
    questionHeadingRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.session.current_question]);

  // Auto-fill answer draft from voice transcript
  useEffect(() => {
    if (voice.turns.length > 0) {
      const lastUserTurn = [...voice.turns]
        .reverse()
        .find((t) => t.role === "user");
      if (lastUserTurn && lastUserTurn.text) {
        setAnswerDraft((prev) =>
          prev ? prev + " " + lastUserTurn.text : lastUserTurn.text,
        );
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voice.turns.length]);

  const loadSession = useCallback(async () => {
    if (!sessionId) return;
    store.setLoading(true);
    store.setError("");
    try {
      const res = await fetchInterviewSession(sessionId);
      store.setSession(res);
      if (res.session.current_question) {
        store.setStartedAt(Date.now());
      }
    } catch {
      store.setError("Could not load interview session.");
    } finally {
      store.setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  useEffect(() => {
    loadSession();
  }, [loadSession]);

  const handleLoadHint = async () => {
    const question = session?.session.current_question;
    if (!sessionId || !question) return;
    store.setHintLoading(true);
    store.setHintError("");
    try {
      const res = await fetchHint(
        questionIdFor(question),
        hintLevel,
        { question_text: question },
      );
      store.setHintResult(res);
      store.setHintLevelUsed(Math.max(hintLevelUsed, res.level));
    } catch (err) {
      const detail = (
        err as {
          response?: { data?: { detail?: unknown } };
          message?: string;
        }
      )?.response?.data?.detail;
      store.setHintError(
        typeof detail === "string"
          ? detail
          : "Could not load a hint for this question.",
      );
    } finally {
      store.setHintLoading(false);
    }
  };

  const ensureNextQuestion = useCallback(async () => {
    if (!sessionId || !session || session.session.status !== "active") return;
    if (session.session.current_question) return;

    store.setLoadingNext(true);
    store.setError("");
    store.setStreamProgress("Generating first interview question...");
    const payload = {
      llm_config: {
        provider: settings.provider,
        model: settings.model,
        temperature: settings.temperature,
        max_tokens: settings.maxTokens,
      },
    };
    try {
      const result = await nextQuestionStream.run({
        stream: (signal) =>
          generateNextInterviewQuestionStream(sessionId, payload, {
            onStart: (event) => {
              store.setStreamProgress(
                event.message || "Generating interview question...",
              );
            },
            onProgress: (event) => {
              store.setStreamProgress(
                event.message || "Generating interview question...",
              );
            },
            onDone: (event) => {
              store.setCurrentQuestion(event.question);
            },
            onError: (event) => {
              store.setError(
                event.message || "Could not generate the next question.",
              );
            },
          }, signal),
        fallback: () => generateNextInterviewQuestion(sessionId, payload),
        isFallbackEligible: interviewStreamFallbackEligible,
      });
      if (result.aborted) return;
      if (result.error) {
        store.setError(
          isInterviewStreamError(result.error) && !result.error.fallbackEligible
            ? result.error.message || "Could not generate the next question."
            : "Could not generate the next question.",
        );
        return;
      }
      if (result.usedFallback && result.data) {
        store.setCurrentQuestion(result.data.question);
      }
      store.setStartedAt(Date.now());
    } finally {
      store.setStreamProgress("");
      store.setLoadingNext(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, session, settings, nextQuestionStream]);

  useEffect(() => {
    ensureNextQuestion();
  }, [ensureNextQuestion]);

  const handleSubmit = async () => {
    if (!sessionId || !session || !answerDraft.trim()) return;
    store.setSubmitting(true);
    store.setError("");
    store.setStreamProgress("Evaluating your answer...");
    const payload = {
      user_answer: answerDraft.trim(),
      response_time_ms: Math.max(0, Date.now() - startedAt),
      question_id: session.session.current_question
        ? questionIdFor(session.session.current_question)
        : undefined,
      hint_level: hintLevelUsed > 0 ? hintLevelUsed : undefined,
      llm_config: {
        provider: settings.provider,
        model: settings.model,
        temperature: settings.temperature,
        max_tokens: settings.maxTokens,
      },
    };
    try {
      const result = await submitAnswerStream.run({
        stream: (signal) =>
          submitInterviewAnswerStream(sessionId, payload, {
            onStart: (event) => {
              store.setStreamProgress(
                event.message || "Evaluating your answer...",
              );
            },
            onProgress: (event) => {
              store.setStreamProgress(
                event.message || "Evaluating your answer...",
              );
            },
            onDone: () => {
              store.setStreamProgress("Answer evaluation completed.");
            },
            onError: (event) => {
              store.setError(
                event.message || "Failed to submit answer. Please retry.",
              );
            },
          }, signal),
        fallback: () => submitInterviewAnswer(sessionId, payload),
        isFallbackEligible: interviewStreamFallbackEligible,
      });
      if (result.aborted) return;
      if (result.error) {
        store.setError(
          isInterviewStreamError(result.error) && !result.error.fallbackEligible
            ? result.error.message || "Failed to submit answer. Please retry."
            : "Failed to submit answer. Please retry.",
        );
        return;
      }
      const res = result.data;
      if (!res) return;
      store.appendTurn(res);
      setAnswerDraft("");
      if (res.session.status === "completed") {
        navigate(`/interview/${sessionId}/report`);
      }
    } finally {
      store.setStreamProgress("");
      store.setSubmitting(false);
    }
  };

  const handleNext = async () => {
    if (!sessionId) return;
    store.setLoadingNext(true);
    store.setError("");
    store.setStreamProgress("Generating next interview question...");
    const payload = {
      llm_config: {
        provider: settings.provider,
        model: settings.model,
        temperature: settings.temperature,
        max_tokens: settings.maxTokens,
      },
    };
    try {
      const result = await nextQuestionStream.run({
        stream: (signal) =>
          generateNextInterviewQuestionStream(sessionId, payload, {
            onStart: (event) => {
              store.setStreamProgress(
                event.message || "Generating next interview question...",
              );
            },
            onProgress: (event) => {
              store.setStreamProgress(
                event.message || "Generating next interview question...",
              );
            },
            onDone: () => {
              store.setStreamProgress("Next question ready.");
            },
            onError: (event) => {
              store.setError(
                event.message || "Could not fetch next question.",
              );
            },
          }, signal),
        fallback: () => generateNextInterviewQuestion(sessionId, payload),
        isFallbackEligible: interviewStreamFallbackEligible,
      });
      if (result.aborted) return;
      if (result.error) {
        store.setError(
          isInterviewStreamError(result.error) && !result.error.fallbackEligible
            ? result.error.message || "Could not fetch next question."
            : "Could not fetch next question.",
        );
        return;
      }
      if (result.usedFallback && result.data) {
        store.setCurrentQuestion(result.data.question);
      }
      store.setStartedAt(Date.now());
      store.setLastTurn(null);
    } finally {
      store.setStreamProgress("");
      store.setLoadingNext(false);
    }
  };

  const progressPct = useMemo(() => {
    if (!session) return 0;
    return Math.round(
      (session.session.turns_completed / session.session.turn_count) * 100,
    );
  }, [session]);
  const isCoding = session?.session.interview_type === "coding";

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-udemy-purple" />
      </div>
    );
  }

  if (!session) {
    return (
      <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-12">
        <p className="text-udemy-text-muted">Interview session not found.</p>
      </div>
    );
  }

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
    >
      <div className="bg-udemy-dark text-white">
        <div className="max-w-[1000px] mx-auto px-4 sm:px-6 py-6">
          <h1 className="text-2xl font-bold">Live Mock Interview</h1>
          <p className="text-gray-400 text-sm mt-1">
            {session.session.track} · {session.session.level} ·{" "}
            {session.session.interview_type}
          </p>
          <p className="text-gray-400 text-xs mt-1">
            style: {session.session.interviewer_style} · feedback:{" "}
            {session.session.feedback_mode}
          </p>
          <div className="mt-4 bg-white/15 h-2 rounded-full overflow-hidden">
            <div
              className="bg-udemy-purple-light h-full"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-2">
            {session.session.turns_completed}/{session.session.turn_count} turns
            completed
          </p>
        </div>
      </div>

      <div className="max-w-[1000px] mx-auto px-4 sm:px-6 py-8 space-y-6">
        {error && (
          <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2" role="alert">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
        {(submitting || loadingNext) && streamProgress && (
          <div className="rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900 flex items-center gap-2" aria-live="polite" role="status">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>{streamProgress}</span>
          </div>
        )}

        <div className="udemy-card p-6">
          <h2 className="text-sm font-bold text-udemy-text-muted uppercase mb-2">
            Current Question
          </h2>
          <p
            ref={questionHeadingRef}
            tabIndex={-1}
            className="text-lg font-medium leading-relaxed whitespace-pre-line outline-none"
          >
            {normalizeEscapedMultilineText(
              session.session.current_question || "Generating next question...",
            )}
          </p>
          {isCoding && (
            <p className="text-xs text-udemy-text-muted mt-2">
              Include your approach, complexity analysis, and edge-case
              handling.
            </p>
          )}
          {voice.voiceEnabled && session.session.current_question && (
            <button
              onClick={() =>
                voice.speak(
                  normalizeEscapedMultilineText(
                    session.session.current_question || "",
                  ),
                )
              }
              disabled={voice.isSpeaking}
              className="mt-3 inline-flex items-center gap-1.5 text-xs text-udemy-purple
                hover:text-udemy-purple-light transition-colors disabled:opacity-50"
            >
              <Volume2 className="w-3.5 h-3.5" />
              {voice.isSpeaking ? "Speaking..." : "Read aloud"}
            </button>
          )}

          <div className="mt-4 pt-4 border-t border-udemy-border">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-1.5 text-sm font-medium">
                <Lightbulb className="w-4 h-4 text-udemy-purple" />
                Hints
              </div>
              <select
                value={hintLevel}
                onChange={(e) => store.setHintLevel(Number(e.target.value))}
                disabled={hintLoading}
                className="border border-udemy-border rounded px-2 py-1.5 text-sm"
              >
                <option value={1}>Level 1 — gentle nudge</option>
                <option value={2}>Level 2 — approach</option>
                <option value={3}>Level 3 — partial solution</option>
                <option value={4}>Level 4 — full solution</option>
              </select>
              <button
                onClick={handleLoadHint}
                disabled={hintLoading || !session.session.current_question}
                className="btn-secondary inline-flex items-center gap-1.5 text-sm disabled:opacity-50"
              >
                {hintLoading ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Lightbulb className="w-3.5 h-3.5" />
                )}
                Get hint
              </button>
              {hintResult && (
                <span className="text-xs text-udemy-text-muted">
                  {hintResult.hints_used} hint
                  {hintResult.hints_used === 1 ? "" : "s"} used ·{" "}
                  {hintResult.hints_remaining} remaining
                </span>
              )}
            </div>
            <p className="text-xs text-udemy-text-muted mt-1.5">
              Using hints lowers the reasoning-depth score for this answer.
            </p>
            {hintError && (
              <p className="text-xs text-red-600 mt-2 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                {hintError}
              </p>
            )}
            {hintResult && (
              <div className="mt-3 rounded-lg bg-amber-50 border border-amber-200 p-3">
                <p className="text-xs font-medium text-amber-700 mb-1">
                  Hint — level {hintResult.level}
                </p>
                <p className="text-sm whitespace-pre-line">
                  {hintResult.hint}
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="udemy-card p-6">
          <label className="block text-sm font-medium mb-2">Your Answer</label>
          <div className="relative">
            <textarea
              value={answerDraft}
              onChange={(e) => setAnswerDraft(e.target.value)}
              rows={isCoding ? 12 : 8}
              placeholder={
                isCoding
                  ? "Write your solution and explain complexity + edge cases..."
                  : voice.voiceEnabled
                    ? "Write or speak your interview answer..."
                    : "Write your interview answer..."
              }
              className={`w-full border border-udemy-border rounded px-3 py-2.5 text-sm ${
                isCoding ? "font-mono" : ""
              }`}
            />
            {voice.voiceEnabled && (
              <button
                onClick={async () => {
                  if (voice.isRecording) {
                    voice.stopRecording();
                    // After stopping, the transcript appears in voice.turns — grab it
                    // For browser tier, the transcript was already put into currentTranscript
                  } else {
                    // Start a voice session for interview if not already active
                    if (!voice.sessionActive) {
                      // No system prompt: the server owns the interview-coach persona.
                      await voice.startSession("interview", { sessionId });
                    }
                    voice.startRecording();
                  }
                }}
                className={`absolute bottom-3 right-3 p-2 rounded-full transition-colors ${
                  voice.isRecording
                    ? "bg-red-500 hover:bg-red-600 animate-pulse"
                    : "bg-gray-200 hover:bg-gray-300"
                }`}
                title={
                  voice.isRecording ? "Stop recording" : "Speak your answer"
                }
                aria-label={
                  voice.isRecording ? "Stop recording" : "Speak your answer"
                }
              >
                {voice.isRecording ? (
                  <MicOff className="w-4 h-4 text-white" />
                ) : (
                  <Mic className="w-4 h-4 text-gray-600" />
                )}
              </button>
            )}
          </div>
          {voice.currentTranscript && (
            <p className="text-xs text-gray-500 mt-1 italic">
              🎙 {voice.currentTranscript}...
            </p>
          )}

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              onClick={handleSubmit}
              disabled={
                submitting ||
                loadingNext ||
                !answerDraft.trim() ||
                !session.session.current_question
              }
              className="btn-primary w-full sm:w-auto flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {submitting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
              Submit Answer
            </button>

            <button
              onClick={handleNext}
              disabled={
                loadingNext || submitting || session.session.status !== "active"
              }
              className="btn-secondary w-full sm:w-auto flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loadingNext ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <SkipForward className="w-4 h-4" />
              )}
              Next Question
            </button>

            {session.session.report_ready && (
              <Link
                to={`/interview/${session.session.session_id}/report`}
                className="btn-secondary w-full sm:w-auto inline-flex items-center justify-center gap-2"
              >
                <FileText className="w-4 h-4" />
                View Report
              </Link>
            )}
          </div>
        </div>

        {lastTurn && (
          <div className="udemy-card p-6">
            <h2 className="text-sm font-bold text-udemy-text-muted uppercase mb-3">
              Rubric Feedback
            </h2>
            {lastTurn.turn.degraded || lastTurn.degraded ? (
              <div className="flex items-start gap-2 text-sm text-udemy-text-muted" role="alert">
                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                <p>
                  Evaluation unavailable — your answer was saved. Submit the next
                  question or repeat this turn to get scored feedback.
                </p>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 mb-4">
                  {Object.entries(lastTurn.turn.rubric)
                    .filter(([k]) => k !== "degraded")
                    .map(([k, v]) => (
                      <div key={k} className="bg-udemy-bg rounded px-3 py-2">
                        <p className="text-[11px] text-udemy-text-muted uppercase">
                          {k.replace(/_/g, " ")}
                        </p>
                        <p className="font-bold">{v}</p>
                      </div>
                    ))}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-sm font-semibold mb-2">Strengths</h3>
                    <ul className="text-sm text-udemy-text-muted list-disc pl-5 space-y-1">
                      {lastTurn.turn.strengths.map((s, i) => (
                        <li key={`${s}-${i}`}>{s}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold mb-2">Improvements</h3>
                    <ul className="text-sm text-udemy-text-muted list-disc pl-5 space-y-1">
                      {lastTurn.turn.improvements.map((s, i) => (
                        <li key={`${s}-${i}`}>{s}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="mt-4">
                  <MarkdownRenderer
                    content={lastTurn.turn.follow_up_note}
                    className="text-sm"
                    compact
                  />
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}
