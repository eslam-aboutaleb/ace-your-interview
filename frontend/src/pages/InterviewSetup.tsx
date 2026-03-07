import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Loader2, MessageSquare, Sparkles, Clock, ArrowRight } from "lucide-react";
import {
  pageVariants,
  pageTransition,
  containerVariants,
  cardVariants,
} from "@/utils/animations";
import {
  createInterviewSession,
  fetchInterviewSessions,
} from "@/services/api";
import { useSettingsStore } from "@/store/settingsStore";
import type {
  InterviewLevel,
  InterviewSession,
  InterviewType,
  LearningTrack,
} from "@/types";

const TRACK_LABELS: Record<LearningTrack, string> = {
  backend: "Backend",
  frontend: "Frontend",
  system_design: "System Design",
  ai_stack: "AI Stack",
};

export default function InterviewSetup() {
  const navigate = useNavigate();
  const settings = useSettingsStore();

  const [track, setTrack] = useState<LearningTrack>("backend");
  const [level, setLevel] = useState<InterviewLevel>("mid");
  const [interviewType, setInterviewType] = useState<InterviewType>("mixed");
  const [turnCount, setTurnCount] = useState(5);
  const [targetRole, setTargetRole] = useState("");
  const [jobDescriptionText, setJobDescriptionText] = useState("");
  const [resumeSummaryText, setResumeSummaryText] = useState("");
  const [focusAreasRaw, setFocusAreasRaw] = useState("");

  const [starting, setStarting] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [recent, setRecent] = useState<InterviewSession[]>([]);
  const [loadingRecent, setLoadingRecent] = useState(true);

  useEffect(() => {
    fetchInterviewSessions(6, 0)
      .then((res) => setRecent(res.sessions))
      .catch(() => setRecent([]))
      .finally(() => setLoadingRecent(false));
  }, []);

  const focusAreas = focusAreasRaw
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean)
    .slice(0, 8);

  const handleStart = async () => {
    setStarting(true);
    setErrorMsg("");
    try {
      const res = await createInterviewSession({
        track,
        level,
        interview_type: interviewType,
        turn_count: turnCount,
        target_role: targetRole.trim(),
        job_description_text: jobDescriptionText.trim(),
        resume_summary_text: resumeSummaryText.trim(),
        focus_areas: focusAreas,
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
      });
      navigate(`/interview/${res.session.session_id}`);
    } catch {
      setErrorMsg(
        "Could not start mock interview. Check backend feature flag and LLM settings.",
      );
    } finally {
      setStarting(false);
    }
  };

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
    >
      <div className="bg-udemy-dark text-white">
        <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-8">
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
            <MessageSquare className="w-7 h-7 text-udemy-purple-light" />
            Mock Interview
          </h1>
          <p className="text-gray-400 mt-2">
            Configure a personalized interview session and practice turn-by-turn.
          </p>
        </div>
      </div>

      <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-8">
        {errorMsg && (
          <div className="mb-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
            {errorMsg}
          </div>
        )}

        <motion.div
          className="grid grid-cols-1 lg:grid-cols-3 gap-6"
          variants={containerVariants}
          initial="hidden"
          animate="show"
        >
          <motion.div variants={cardVariants} className="lg:col-span-2">
            <div className="udemy-card p-6 space-y-5">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Track</label>
                  <select
                    value={track}
                    onChange={(e) => setTrack(e.target.value as LearningTrack)}
                    className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  >
                    <option value="backend">Backend</option>
                    <option value="frontend">Frontend</option>
                    <option value="system_design">System Design</option>
                    <option value="ai_stack">AI Stack</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Level</label>
                  <select
                    value={level}
                    onChange={(e) => setLevel(e.target.value as InterviewLevel)}
                    className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  >
                    <option value="junior">Junior</option>
                    <option value="mid">Mid</option>
                    <option value="senior">Senior</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Interview Type</label>
                  <select
                    value={interviewType}
                    onChange={(e) => setInterviewType(e.target.value as InterviewType)}
                    className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  >
                    <option value="mixed">Mixed</option>
                    <option value="technical">Technical</option>
                    <option value="behavioral">Behavioral</option>
                    <option value="system_design">System Design</option>
                    <option value="ai_fundamentals">AI Fundamentals</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Turn Count</label>
                  <select
                    value={turnCount}
                    onChange={(e) => setTurnCount(Number(e.target.value))}
                    className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  >
                    {[3, 5, 7, 10, 12].map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Target Role (optional)</label>
                <input
                  type="text"
                  value={targetRole}
                  onChange={(e) => setTargetRole(e.target.value)}
                  placeholder="e.g. Senior Backend Engineer"
                  className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Focus Areas (comma-separated, optional)
                </label>
                <input
                  type="text"
                  value={focusAreasRaw}
                  onChange={(e) => setFocusAreasRaw(e.target.value)}
                  placeholder="api design, caching, tradeoffs"
                  className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Job Description Context (optional)</label>
                <textarea
                  value={jobDescriptionText}
                  onChange={(e) => setJobDescriptionText(e.target.value)}
                  rows={5}
                  className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  placeholder="Paste key responsibilities or required skills..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Resume Summary Context (optional)</label>
                <textarea
                  value={resumeSummaryText}
                  onChange={(e) => setResumeSummaryText(e.target.value)}
                  rows={4}
                  className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  placeholder="Paste a concise summary of your experience..."
                />
              </div>

              <button
                onClick={handleStart}
                disabled={starting}
                className="btn-primary w-full sm:w-auto flex items-center gap-2 disabled:opacity-50"
              >
                {starting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4" />
                )}
                {starting ? "Starting..." : "Start Mock Interview"}
              </button>
            </div>
          </motion.div>

          <motion.div variants={cardVariants}>
            <div className="udemy-card p-6">
              <h2 className="text-lg font-bold mb-3">Recent Sessions</h2>
              {loadingRecent ? (
                <div className="space-y-2">
                  {Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="skeleton h-16 rounded" />
                  ))}
                </div>
              ) : recent.length === 0 ? (
                <p className="text-sm text-udemy-text-muted">No sessions yet.</p>
              ) : (
                <div className="space-y-2">
                  {recent.map((s) => (
                    <Link
                      key={s.session_id}
                      to={`/interview/${s.session_id}`}
                      className="block rounded border border-udemy-border px-3 py-2 hover:bg-gray-50"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-medium line-clamp-1">
                          {TRACK_LABELS[s.track]} · {s.level}
                        </span>
                        <span
                          className={`text-xs px-2 py-0.5 rounded-full ${
                            s.status === "completed"
                              ? "bg-green-100 text-green-700"
                              : "bg-blue-100 text-blue-700"
                          }`}
                        >
                          {s.status}
                        </span>
                      </div>
                      <div className="text-xs text-udemy-text-muted mt-1 flex items-center gap-2">
                        <Clock className="w-3.5 h-3.5" />
                        {s.turns_completed}/{s.turn_count} turns
                      </div>
                    </Link>
                  ))}
                </div>
              )}

              <div className="mt-5 text-xs text-udemy-text-muted">
                Powered by {settings.provider}
              </div>

              <Link
                to="/quiz"
                className="mt-4 inline-flex items-center gap-1 text-sm text-udemy-purple"
              >
                Practice quick quiz
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </motion.div>
  );
}
