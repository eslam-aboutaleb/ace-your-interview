import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { AlertTriangle, Loader2, Send, SkipForward, FileText } from "lucide-react";
import { pageVariants, pageTransition } from "@/utils/animations";
import {
  fetchInterviewSession,
  generateNextInterviewQuestion,
  submitInterviewAnswer,
} from "@/services/api";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import { useSettingsStore } from "@/store/settingsStore";
import type { InterviewSessionResponse, InterviewTurnResponse } from "@/types";

export default function InterviewSessionPage() {
  const { sessionId = "" } = useParams();
  const navigate = useNavigate();
  const settings = useSettingsStore();

  const [data, setData] = useState<InterviewSessionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");
  const [answerDraft, setAnswerDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [loadingNext, setLoadingNext] = useState(false);
  const [startedAt, setStartedAt] = useState<number>(Date.now());
  const [lastTurn, setLastTurn] = useState<InterviewTurnResponse | null>(null);

  const loadSession = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const res = await fetchInterviewSession(sessionId);
      setData(res);
      if (res.session.current_question) {
        setStartedAt(Date.now());
      }
    } catch {
      setErrorMsg("Could not load interview session.");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    loadSession();
  }, [loadSession]);

  const ensureNextQuestion = useCallback(async () => {
    if (!sessionId || !data || data.session.status !== "active") return;
    if (data.session.current_question) return;

    setLoadingNext(true);
    setErrorMsg("");
    try {
      const next = await generateNextInterviewQuestion(sessionId, {
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
      });
      setData((prev) =>
        prev
          ? {
              ...prev,
              session: {
                ...prev.session,
                current_question: next.question,
              },
            }
          : prev,
      );
      setStartedAt(Date.now());
    } catch {
      setErrorMsg("Could not generate the next question.");
    } finally {
      setLoadingNext(false);
    }
  }, [sessionId, data, settings]);

  useEffect(() => {
    ensureNextQuestion();
  }, [ensureNextQuestion]);

  const handleSubmit = async () => {
    if (!sessionId || !data || !answerDraft.trim()) return;
    setSubmitting(true);
    setErrorMsg("");
    try {
      const responseTimeMs = Math.max(0, Date.now() - startedAt);
      const res = await submitInterviewAnswer(sessionId, {
        user_answer: answerDraft.trim(),
        response_time_ms: responseTimeMs,
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
      });
      setLastTurn(res);
      setData((prev) =>
        prev
          ? {
              ...prev,
              session: res.session,
              turns: [...prev.turns, res.turn],
            }
          : prev,
      );
      setAnswerDraft("");
      if (res.session.status === "completed") {
        navigate(`/interview/${sessionId}/report`);
      }
    } catch {
      setErrorMsg("Failed to submit answer. Please retry.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleNext = async () => {
    if (!sessionId) return;
    setLoadingNext(true);
    setErrorMsg("");
    try {
      const next = await generateNextInterviewQuestion(sessionId, {
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
      });
      setData((prev) =>
        prev
          ? {
              ...prev,
              session: {
                ...prev.session,
                current_question: next.question,
              },
            }
          : prev,
      );
      setStartedAt(Date.now());
      setLastTurn(null);
    } catch {
      setErrorMsg("Could not fetch next question.");
    } finally {
      setLoadingNext(false);
    }
  };

  const progressPct = useMemo(() => {
    if (!data) return 0;
    return Math.round((data.session.turns_completed / data.session.turn_count) * 100);
  }, [data]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-udemy-purple" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="max-w-[1340px] mx-auto px-6 py-12">
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
        <div className="max-w-[1000px] mx-auto px-6 py-6">
          <h1 className="text-2xl font-bold">Live Mock Interview</h1>
          <p className="text-gray-400 text-sm mt-1">
            {data.session.track} · {data.session.level} · {data.session.interview_type}
          </p>
          <div className="mt-4 bg-white/15 h-2 rounded-full overflow-hidden">
            <div
              className="bg-udemy-purple-light h-full"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-2">
            {data.session.turns_completed}/{data.session.turn_count} turns completed
          </p>
        </div>
      </div>

      <div className="max-w-[1000px] mx-auto px-6 py-8 space-y-6">
        {errorMsg && (
          <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        <div className="udemy-card p-6">
          <h2 className="text-sm font-bold text-udemy-text-muted uppercase mb-2">Current Question</h2>
          <p className="text-lg font-medium leading-relaxed">
            {data.session.current_question || "Generating next question..."}
          </p>
        </div>

        <div className="udemy-card p-6">
          <label className="block text-sm font-medium mb-2">Your Answer</label>
          <textarea
            value={answerDraft}
            onChange={(e) => setAnswerDraft(e.target.value)}
            rows={8}
            placeholder="Write your interview answer..."
            className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
          />

          <div className="mt-4 flex items-center gap-3">
            <button
              onClick={handleSubmit}
              disabled={submitting || loadingNext || !answerDraft.trim() || !data.session.current_question}
              className="btn-primary flex items-center gap-2 disabled:opacity-50"
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
              disabled={loadingNext || submitting || data.session.status !== "active"}
              className="btn-secondary flex items-center gap-2 disabled:opacity-50"
            >
              {loadingNext ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <SkipForward className="w-4 h-4" />
              )}
              Next Question
            </button>

            {data.session.report_ready && (
              <Link
                to={`/interview/${data.session.session_id}/report`}
                className="btn-secondary inline-flex items-center gap-2"
              >
                <FileText className="w-4 h-4" />
                View Report
              </Link>
            )}
          </div>
        </div>

        {lastTurn && (
          <div className="udemy-card p-6">
            <h2 className="text-sm font-bold text-udemy-text-muted uppercase mb-3">Rubric Feedback</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
              {Object.entries(lastTurn.turn.rubric).map(([k, v]) => (
                <div key={k} className="bg-udemy-bg rounded px-3 py-2">
                  <p className="text-[11px] text-udemy-text-muted uppercase">{k.replace(/_/g, " ")}</p>
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
          </div>
        )}
      </div>
    </motion.div>
  );
}
