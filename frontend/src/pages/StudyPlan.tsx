import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  CalendarDays,
  Loader2,
  RefreshCw,
  Save,
  Sparkles,
  Target,
} from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import {
  fetchLearnerProfile,
  fetchRecommendations,
  fetchStudyPlan,
  fetchTopics,
  runProfileDiagnostic,
  updateLearnerProfile,
} from "@/services/api";
import { normalizeEscapedSingleLineText } from "@/utils/textNormalization";
import type {
  LearnerProfileUpdateRequest,
  LearningTrack,
  ProfileDiagnosticResponse,
  RecommendationItem,
  RecommendationsResponse,
  StudyPlanResponse,
  TopicSummary,
} from "@/types";

type ProfileDraft = LearnerProfileUpdateRequest & {
  focus_topic_ids: string[];
  target_companies: string[];
  preferred_modalities: Array<"study" | "quiz" | "interview" | "voice">;
  confidence_by_track: Partial<Record<LearningTrack, number>>;
};

const TRACK_LABELS: Record<LearningTrack, string> = {
  backend: "Backend",
  frontend: "Frontend",
  system_design: "System Design",
  ai_stack: "AI Stack",
};

const MODALITY_OPTIONS: Array<"study" | "quiz" | "interview" | "voice"> = [
  "study",
  "quiz",
  "interview",
  "voice",
];

function emptyDraft(): ProfileDraft {
  return {
    target_role: "",
    target_date: "",
    weekly_minutes: 240,
    preferred_session_minutes: 30,
    current_level: "mid",
    target_level: "senior",
    primary_track: "backend",
    focus_topic_ids: [],
    target_companies: [],
    preferred_modalities: ["study", "quiz", "interview"],
    confidence_by_track: {
      backend: 3,
      frontend: 3,
      system_design: 3,
      ai_stack: 3,
    },
  };
}

function profileToDraft(profile: RecommendationsResponse["profile"]): ProfileDraft {
  return {
    target_role: profile.target_role,
    target_date: profile.target_date,
    weekly_minutes: profile.weekly_minutes,
    preferred_session_minutes: profile.preferred_session_minutes,
    current_level: profile.current_level,
    target_level: profile.target_level,
    primary_track: profile.primary_track,
    focus_topic_ids: profile.focus_topic_ids || [],
    target_companies: profile.target_companies || [],
    preferred_modalities: profile.preferred_modalities || ["study", "quiz", "interview"],
    confidence_by_track: {
      backend: profile.confidence_by_track.backend ?? 3,
      frontend: profile.confidence_by_track.frontend ?? 3,
      system_design: profile.confidence_by_track.system_design ?? 3,
      ai_stack: profile.confidence_by_track.ai_stack ?? 3,
    },
  };
}

function daysUntilTarget(value: string): number | null {
  if (!value) return null;
  const target = new Date(`${value}T00:00:00`);
  if (Number.isNaN(target.getTime())) return null;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return Math.round((target.getTime() - today.getTime()) / 86_400_000);
}

export default function StudyPlanPage() {
  const [plan, setPlan] = useState<StudyPlanResponse | null>(null);
  const [topics, setTopics] = useState<TopicSummary[]>([]);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [diagnostic, setDiagnostic] = useState<ProfileDiagnosticResponse | null>(null);
  const [draft, setDraft] = useState<ProfileDraft>(emptyDraft());
  const [loading, setLoading] = useState(true);
  const [savingProfile, setSavingProfile] = useState(false);
  const [runningDiagnostic, setRunningDiagnostic] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [message, setMessage] = useState("");

  const load = async () => {
    setLoading(true);
    setErrorMsg("");
    try {
      const [studyPlan, topicList, profile, nextRecommendations] = await Promise.all([
        fetchStudyPlan(7, 3),
        fetchTopics(),
        fetchLearnerProfile(),
        fetchRecommendations(6),
      ]);
      setPlan(studyPlan);
      setTopics(topicList);
      setDraft(profileToDraft(profile));
      setRecommendations(nextRecommendations.items || []);

      if (profile.diagnostic_updated_at) {
        runProfileDiagnostic()
          .then(setDiagnostic)
          .catch(() => {
            // keep page usable even if the diagnostic refresh fails
          });
      } else {
        setDiagnostic(null);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg("Could not load your study planning workspace. Please retry.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const topicMap = useMemo(
    () =>
      new Map(
        topics.map((topic) => [topic.id, normalizeEscapedSingleLineText(topic.title)]),
      ),
    [topics],
  );

  const targetCountdown = useMemo(
    () => daysUntilTarget(draft.target_date || ""),
    [draft.target_date],
  );

  const handleSaveProfile = async () => {
    setSavingProfile(true);
    setMessage("");
    setErrorMsg("");
    try {
      const saved = await updateLearnerProfile({
        ...draft,
        focus_topic_ids: draft.focus_topic_ids,
        target_companies: draft.target_companies,
      });
      setDraft(profileToDraft(saved));
      setMessage("Learner profile saved.");
      const [studyPlan, nextRecommendations] = await Promise.all([
        fetchStudyPlan(7, 3),
        fetchRecommendations(6),
      ]);
      setPlan(studyPlan);
      setRecommendations(nextRecommendations.items || []);
    } catch (err) {
      console.error(err);
      setErrorMsg("Could not save your learner profile.");
    } finally {
      setSavingProfile(false);
    }
  };

  const handleRunDiagnostic = async () => {
    setRunningDiagnostic(true);
    setMessage("");
    setErrorMsg("");
    try {
      const [diagnosticResult, studyPlan, nextRecommendations] = await Promise.all([
        runProfileDiagnostic(),
        fetchStudyPlan(7, 3),
        fetchRecommendations(6),
      ]);
      setDiagnostic(diagnosticResult);
      setPlan(studyPlan);
      setRecommendations(nextRecommendations.items || []);
      setMessage("Diagnostic refreshed.");
    } catch (err) {
      console.error(err);
      setErrorMsg("Could not run the learning diagnostic.");
    } finally {
      setRunningDiagnostic(false);
    }
  };

  const updateConfidence = (track: LearningTrack, value: number) => {
    setDraft((prev) => ({
      ...prev,
      confidence_by_track: {
        ...prev.confidence_by_track,
        [track]: value,
      },
    }));
  };

  const toggleModality = (modality: "study" | "quiz" | "interview" | "voice") => {
    setDraft((prev) => {
      const current = new Set(prev.preferred_modalities);
      if (current.has(modality)) current.delete(modality);
      else current.add(modality);
      const next = Array.from(current) as Array<
        "study" | "quiz" | "interview" | "voice"
      >;
      return {
        ...prev,
        preferred_modalities: next.length ? next : ["study"],
      };
    });
  };

  const planSummary = plan
    ? [
        `${plan.days} days`,
        `${plan.daily_items} item(s)/day`,
        `${plan.total_tasks} total task(s)`,
      ].join(" · ")
    : "";

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
      className="max-w-[1240px] mx-auto px-4 sm:px-6 py-8"
    >
      <div className="udemy-card p-6">
        <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <CalendarDays className="w-6 h-6 text-udemy-purple" />
              Goal-Aware Study Plan
            </h1>
            <p className="text-sm text-udemy-text-muted mt-1">
              Recommendations now consider mastery, due review pressure, your target role,
              and deadline instead of only weak areas.
            </p>
            {planSummary && (
              <p className="text-xs text-udemy-text-muted mt-2">{planSummary}</p>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={load}
              disabled={loading}
              className="btn-secondary inline-flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              Refresh
            </button>
            <button
              onClick={handleRunDiagnostic}
              disabled={runningDiagnostic || loading}
              className="btn-primary inline-flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {runningDiagnostic ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              Run Diagnostic
            </button>
          </div>
        </div>

        {message && <p className="mt-3 text-sm text-green-700">{message}</p>}
        {errorMsg && (
          <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1.3fr,0.9fr] gap-6 mt-6">
        <div className="space-y-6">
          <div className="udemy-card p-6">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-bold flex items-center gap-2">
                  <Target className="w-5 h-5 text-udemy-purple" />
                  Learner Profile
                </h2>
                <p className="text-sm text-udemy-text-muted mt-1">
                  This becomes the ranking input for your next-best actions.
                </p>
              </div>
              <button
                onClick={handleSaveProfile}
                disabled={savingProfile || loading}
                className="btn-secondary inline-flex items-center gap-2 disabled:opacity-50"
              >
                {savingProfile ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                Save Profile
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-5">
              <div>
                <label className="block text-sm font-medium mb-2">Target Role</label>
                <input
                  value={draft.target_role || ""}
                  onChange={(e) =>
                    setDraft((prev) => ({ ...prev, target_role: e.target.value }))
                  }
                  placeholder="e.g. Senior Backend Engineer"
                  className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Target Date</label>
                <input
                  type="date"
                  value={draft.target_date || ""}
                  onChange={(e) =>
                    setDraft((prev) => ({ ...prev, target_date: e.target.value }))
                  }
                  className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Primary Track</label>
                <select
                  value={draft.primary_track || "backend"}
                  onChange={(e) =>
                    setDraft((prev) => ({
                      ...prev,
                      primary_track: e.target.value as LearningTrack,
                    }))
                  }
                  className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
                >
                  <option value="backend">Backend</option>
                  <option value="frontend">Frontend</option>
                  <option value="system_design">System Design</option>
                  <option value="ai_stack">AI Stack</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Weekly Time Budget</label>
                <input
                  type="number"
                  min={30}
                  max={2400}
                  value={draft.weekly_minutes || 240}
                  onChange={(e) =>
                    setDraft((prev) => ({
                      ...prev,
                      weekly_minutes: Number(e.target.value),
                    }))
                  }
                  className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Current Level</label>
                <select
                  value={draft.current_level || "mid"}
                  onChange={(e) =>
                    setDraft((prev) => ({
                      ...prev,
                      current_level: e.target.value as ProfileDraft["current_level"],
                    }))
                  }
                  className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
                >
                  <option value="junior">Junior</option>
                  <option value="mid">Mid</option>
                  <option value="senior">Senior</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Target Level</label>
                <select
                  value={draft.target_level || "senior"}
                  onChange={(e) =>
                    setDraft((prev) => ({
                      ...prev,
                      target_level: e.target.value as ProfileDraft["target_level"],
                    }))
                  }
                  className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
                >
                  <option value="junior">Junior</option>
                  <option value="mid">Mid</option>
                  <option value="senior">Senior</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Preferred Session Length (minutes)
                </label>
                <input
                  type="number"
                  min={10}
                  max={180}
                  value={draft.preferred_session_minutes || 30}
                  onChange={(e) =>
                    setDraft((prev) => ({
                      ...prev,
                      preferred_session_minutes: Number(e.target.value),
                    }))
                  }
                  className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Target Companies</label>
                <input
                  value={(draft.target_companies || []).join(", ")}
                  onChange={(e) =>
                    setDraft((prev) => ({
                      ...prev,
                      target_companies: e.target.value
                        .split(",")
                        .map((item) => item.trim())
                        .filter(Boolean),
                    }))
                  }
                  placeholder="Stripe, Datadog, OpenAI"
                  className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
                />
              </div>
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium mb-2">Focus Topics</label>
              <input
                value={(draft.focus_topic_ids || []).join(", ")}
                onChange={(e) =>
                  setDraft((prev) => ({
                    ...prev,
                    focus_topic_ids: e.target.value
                      .split(",")
                      .map((item) => item.trim())
                      .filter(Boolean),
                  }))
                }
                placeholder="01-backend-api-design-and-contracts, 11-system-design-scaling-and-reliability"
                className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
              />
              <p className="text-xs text-udemy-text-muted mt-1">
                Use topic ids when you want to hard-weight specific areas.
              </p>
            </div>

            <div className="mt-5">
              <label className="block text-sm font-medium mb-2">Preferred Modalities</label>
              <div className="flex flex-wrap gap-2">
                {MODALITY_OPTIONS.map((option) => {
                  const active = (draft.preferred_modalities || []).includes(option);
                  return (
                    <button
                      key={option}
                      type="button"
                      onClick={() => toggleModality(option)}
                      className={`px-3 py-2 rounded-full text-sm border transition-colors ${
                        active
                          ? "bg-udemy-purple text-white border-transparent"
                          : "bg-white text-udemy-text-muted border-udemy-border"
                      }`}
                    >
                      {option}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3 mt-5">
              {(Object.keys(TRACK_LABELS) as LearningTrack[]).map((track) => (
                <div key={track} className="rounded-lg border border-udemy-border p-3 bg-white">
                  <label className="block text-xs text-udemy-text-muted mb-1">
                    Confidence: {TRACK_LABELS[track]}
                  </label>
                  <select
                    value={draft.confidence_by_track[track] || 3}
                    onChange={(e) => updateConfidence(track, Number(e.target.value))}
                    className="w-full border border-udemy-border rounded px-2 py-2 text-sm"
                  >
                    {[1, 2, 3, 4, 5].map((value) => (
                      <option key={value} value={value}>
                        {value}/5
                      </option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
          </div>

          <div className="udemy-card p-6">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-bold">7-Day Plan</h2>
                <p className="text-sm text-udemy-text-muted mt-1">
                  Scheduled from your recommendation stack.
                </p>
              </div>
              {plan && (
                <span className="text-xs text-udemy-text-muted">
                  Generated {new Date(plan.generated_at).toLocaleDateString()}
                </span>
              )}
            </div>

            <div className="mt-5">
              {loading ? (
                <div className="flex items-center justify-center gap-2 text-sm text-udemy-text-muted py-10">
                  <Loader2 className="w-5 h-5 animate-spin text-udemy-purple" />
                  Building your study plan...
                </div>
              ) : !plan || plan.days_plan.length === 0 ? (
                <div className="text-center py-10">
                  <p className="font-semibold">No plan available yet.</p>
                  <p className="text-sm text-udemy-text-muted mt-1">
                    Save a learner profile or complete more practice to generate one.
                  </p>
                  <Link to="/topics" className="btn-primary mt-4 inline-flex">
                    Start studying
                  </Link>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {plan.days_plan.map((day) => (
                    <div key={`${day.day_index}:${day.date}`} className="rounded-xl border border-udemy-border bg-white p-4">
                      <div className="flex items-center justify-between gap-2 mb-3">
                        <h3 className="text-lg font-bold">{day.label}</h3>
                        <span className="text-xs text-udemy-text-muted">{day.date}</span>
                      </div>

                      {day.tasks.length === 0 ? (
                        <p className="text-sm text-udemy-text-muted">
                          No tasks scheduled for this day.
                        </p>
                      ) : (
                        <div className="space-y-3">
                          {day.tasks.map((task, idx) => {
                            const taskTitle = normalizeEscapedSingleLineText(task.title);
                            const taskReason = normalizeEscapedSingleLineText(task.reason);
                            const taskTopicTitle = task.topic_id
                              ? topicMap.get(task.topic_id) || normalizeEscapedSingleLineText(task.topic_id)
                              : "Cross-topic";
                            return (
                              <div
                                key={`${day.day_index}:${idx}:${task.task_type}:${task.cta_route}`}
                                className="rounded-lg border border-udemy-border p-3"
                              >
                                <p className="text-sm font-semibold">{taskTitle}</p>
                                <p className="text-xs text-udemy-text-muted mt-1">
                                  {taskTopicTitle} · {task.task_type.replace("_", " ")} · ~
                                  {task.estimated_minutes} min
                                </p>
                                <p className="text-xs text-udemy-text-muted mt-1">
                                  {taskReason}
                                </p>
                                <Link to={task.cta_route} className="btn-secondary mt-3 inline-flex text-sm">
                                  Open task
                                </Link>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="udemy-card p-6">
            <h2 className="text-lg font-bold">Recommendation Snapshot</h2>
            <p className="text-sm text-udemy-text-muted mt-1">
              Highest-priority actions right now.
            </p>
            <div className="grid grid-cols-2 gap-3 mt-5">
              <Metric label="Primary Track" value={TRACK_LABELS[draft.primary_track || "backend"]} />
              <Metric
                label="Target"
                value={
                  targetCountdown == null
                    ? "Unset"
                    : targetCountdown < 0
                      ? `${Math.abs(targetCountdown)}d overdue`
                      : `${targetCountdown}d left`
                }
              />
              <Metric label="Weekly Budget" value={`${draft.weekly_minutes || 0} min`} />
              <Metric
                label="Session Length"
                value={`${draft.preferred_session_minutes || 0} min`}
              />
            </div>

            <div className="space-y-3 mt-5">
              {recommendations.length === 0 ? (
                <p className="text-sm text-udemy-text-muted">
                  Save a learner profile or complete more practice to unlock ranked actions.
                </p>
              ) : (
                recommendations.map((item) => {
                  const topicTitle = item.topic_id
                    ? topicMap.get(item.topic_id) || normalizeEscapedSingleLineText(item.topic_id)
                    : TRACK_LABELS[(item.track as LearningTrack) || "backend"] || "Interview";
                  return (
                    <div key={item.recommendation_id} className="rounded-lg border border-udemy-border bg-white p-3">
                      <p className="text-sm font-semibold">
                        {normalizeEscapedSingleLineText(item.title)}
                      </p>
                      <p className="text-xs text-udemy-text-muted mt-1">
                        {topicTitle} · priority {Math.round(item.priority)} · ~
                        {item.estimated_minutes} min
                      </p>
                      <p className="text-xs text-udemy-text-muted mt-1">
                        {normalizeEscapedSingleLineText(item.reason)}
                      </p>
                      <Link to={item.cta_route} className="btn-secondary mt-3 inline-flex text-sm">
                        Open
                      </Link>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          <div className="udemy-card p-6">
            <h2 className="text-lg font-bold">Diagnostic</h2>
            <p className="text-sm text-udemy-text-muted mt-1">
              Confidence, mastery, and interview evidence combined.
            </p>

            {!diagnostic ? (
              <p className="text-sm text-udemy-text-muted mt-5">
                Run the diagnostic to compute readiness and priority competencies.
              </p>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-3 mt-5">
                  <Metric
                    label="Readiness"
                    value={`${Math.round(diagnostic.readiness_score)}%`}
                  />
                  <Metric
                    label="Urgent Gaps"
                    value={`${diagnostic.urgent_competencies.length}`}
                  />
                </div>

                <div className="mt-4 rounded-lg border border-udemy-border bg-white p-3">
                  <p className="text-xs text-udemy-text-muted">Strongest</p>
                  <p className="text-sm font-medium mt-1">
                    {diagnostic.strongest_competencies.join(", ") || "Not enough evidence yet"}
                  </p>
                </div>

                <div className="space-y-2 mt-4">
                  {diagnostic.competencies.slice(0, 8).map((item) => (
                    <div key={item.competency_id} className="rounded-lg border border-udemy-border bg-white p-3">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="text-sm font-semibold">{item.label}</p>
                          <p className="text-xs text-udemy-text-muted mt-1">
                            {item.competency_type.replace("_", " ")} · evidence {item.evidence_count}
                          </p>
                        </div>
                        <span className="text-sm font-bold text-udemy-purple">
                          {Math.round(item.score)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-udemy-border px-3 py-2 bg-white">
      <p className="text-xs text-udemy-text-muted">{label}</p>
      <p className="text-lg font-bold">{value}</p>
    </div>
  );
}
