import { useEffect, useState, useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  BookOpen,
  Clock,
  TrendingUp,
  Sparkles,
  ChevronRight,
  MessageSquare,
  Target,
} from "lucide-react";
import {
  containerVariants,
  cardVariants,
  pageVariants,
  pageTransition,
} from "@/utils/animations";
import {
  fetchLearnerProfile,
  fetchRecommendations,
  fetchInterviewStats,
  fetchTopicMastery,
  fetchTopics,
  fetchForecast,
} from "@/services/api";
import { useProgressStore } from "@/store/progressStore";
import { useAuthStore } from "@/store/authStore";
import ProgressBar from "@/components/common/ProgressBar";
import SkeletonCards from "@/components/common/SkeletonCards";
import { normalizeEscapedSingleLineText } from "@/utils/textNormalization";
import type {
  ForecastResponse,
  LearnerProfile,
  RecommendationItem,
  TopicSummary,
} from "@/types";

const TOPIC_ICONS: Record<string, string> = {
  "01-backend-fundamentals-and-http": "🌐",
  "02-backend-api-design-and-contracts": "🧾",
  "03-backend-data-modeling-and-persistence": "🗄️",
  "04-backend-auth-security-observability": "🔐",
  "05-backend-testing-performance-concurrency": "⚡",
  "06-frontend-core-architecture": "🧱",
  "07-frontend-state-data-fetching": "🔄",
  "08-frontend-performance-accessibility": "♿",
  "09-frontend-testing-and-ui-systems": "🧪",
  "10-system-design-foundations": "🏛️",
  "11-system-design-scaling-and-reliability": "📈",
  "12-system-design-data-consistency-and-tradeoffs": "⚖️",
  "13-ai-stack-llm-and-prompting": "🧠",
  "14-ai-stack-rag-and-evaluation": "📚",
  "15-ai-stack-agents-tools-and-guardrails": "🛡️",
  "16-ai-stack-serving-monitoring-and-cost": "💸",
  "17-infrastructure-docker-deep-dive": "🐳",
  "18-infrastructure-terraform-infrastructure-as-code": "🏗️",
  "19-infrastructure-kubernetes-orchestration": "☸️",
  "20-cloud-aws-associate-exam-panel": "☁️",
  "21-cloud-gcp-associate-exam-panel": "🌍",
  "22-cloud-azure-associate-exam-panel": "🔷",
};

const TRACK_LABELS: Record<string, string> = {
  backend: "Backend",
  frontend: "Frontend",
  system_design: "System Design",
  ai_stack: "AI Stack",
};

export default function Dashboard() {
  const navigate = useNavigate();
  const [topics, setTopics] = useState<TopicSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [customTopic, setCustomTopic] = useState("");
  const [customError, setCustomError] = useState("");
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [profile, setProfile] = useState<LearnerProfile | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [interviewStats, setInterviewStats] = useState({
    total_sessions: 0,
    completed_sessions: 0,
    interview_readiness_score: 0,
  });
  const completedTopics = useProgressStore((s) => s.completedTopics);
  const totalProgress = useProgressStore((s) => s.totalProgress);
  const getTopicProgress = useProgressStore((s) => s.getTopicProgress);
  const setTopicCount = useProgressStore((s) => s.setTopicCount);
  const setMastery = useProgressStore((s) => s.setMastery);
  const curriculumNoticePending = useProgressStore(
    (s) => s.curriculumNoticePending,
  );
  const dismissCurriculumNotice = useProgressStore(
    (s) => s.dismissCurriculumNotice,
  );
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);

  const loadTopics = useCallback(async () => {
    setLoading(true);
    try {
      const loaded = await fetchTopics();
      setTopics(loaded);
      setTopicCount(loaded.length);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [setTopicCount]);

  useEffect(() => {
    loadTopics();
    fetchTopicMastery()
      .then((res) => {
        res.topics.forEach((t) => setMastery(t.topic_id, t.mastery_score));
      })
      .catch(() => {
        // ignore mastery bootstrap failures
      });
    fetchInterviewStats()
      .then(setInterviewStats)
      .catch(() => {
        // ignore mock interview stats failures (feature flag may be off)
      });
    fetchForecast(7)
      .then(setForecast)
      .catch(() => {
        // ignore forecast failures (feature flag may be off)
      });
    if (!isAuthenticated) {
      setProfile(null);
      setRecommendations([]);
      return;
    }
    fetchLearnerProfile()
      .then(setProfile)
      .catch(() => {
        setProfile(null);
      });
    fetchRecommendations(3)
      .then((res) => setRecommendations(res.items || []))
      .catch(() => {
        setRecommendations([]);
      });
  }, [isAuthenticated, loadTopics, setMastery]);

  const handleCreateCustomTopic = () => {
    const topic = customTopic.trim();
    if (topic.length < 2) {
      setCustomError(
        "Please enter at least 2 characters for the custom topic.",
      );
      return;
    }

    setCustomError("");
    setCustomTopic("");
    navigate(`/topics/custom/build?topic=${encodeURIComponent(topic)}`);
  };

  const progress = totalProgress();
  const completedCount = completedTopics.length;

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
    >
      {/* Hero banner */}
      <div className="bg-udemy-dark text-white">
        <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <h1 className="text-3xl md:text-4xl font-bold mb-3">
              Too lazy for this interview
            </h1>
            <p className="text-gray-300 text-lg max-w-2xl mb-6">
              Practice backend, frontend, system design (including
              infrastructure/cloud), and AI stack interview topics with adaptive
              questions and quizzes.
            </p>
          </motion.div>

          {/* Stats row */}
          <motion.div
            className="flex flex-wrap gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Stat
              icon={<BookOpen className="w-5 h-5" />}
              label="Topics"
              value={`${topics.length}`}
            />
            <Stat
              icon={<TrendingUp className="w-5 h-5" />}
              label="Completed"
              value={`${completedCount}`}
            />
            <Stat
              icon={<Clock className="w-5 h-5" />}
              label="Progress"
              value={`${progress}%`}
            />
            <Stat
              icon={<Sparkles className="w-5 h-5" />}
              label="AI-Powered"
              value="Enabled"
            />
            <Stat
              icon={<MessageSquare className="w-5 h-5" />}
              label="Mock Interviews"
              value={`${interviewStats.completed_sessions}`}
            />
            <Stat
              icon={<TrendingUp className="w-5 h-5" />}
              label="Readiness"
              value={`${Math.round(interviewStats.interview_readiness_score)}%`}
            />
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
            className="mt-4"
          >
            <Link
              to="/interview/trends"
              className="inline-flex items-center gap-1 text-sm text-udemy-purple-light hover:text-white"
            >
              View readiness trends
              <ChevronRight className="w-4 h-4" />
            </Link>
          </motion.div>
        </div>
      </div>

      {curriculumNoticePending && (
        <div className="max-w-[1340px] mx-auto px-4 sm:px-6 pt-4">
          <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <span>Curriculum updated; progress restarted.</span>
            <button
              onClick={dismissCurriculumNotice}
              className="font-semibold underline"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Overall progress */}
      <div className="max-w-[1340px] mx-auto px-4 sm:px-6 -mt-3">
        <div className="bg-white rounded-lg shadow-card p-4 flex items-center gap-4">
          <span className="text-sm font-semibold text-udemy-text-muted whitespace-nowrap">
            Overall Progress
          </span>
          <div className="flex-1">
            <ProgressBar percent={progress} />
          </div>
          <span className="text-sm font-bold text-udemy-purple">
            {progress}%
          </span>
        </div>
      </div>

      {/* Due forecast */}
      {forecast && (
        <div className="max-w-[1340px] mx-auto px-4 sm:px-6 pt-4">
          <div className="udemy-card p-4">
            <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
              <div>
                <h2 className="text-sm font-bold mb-1">
                  Due in the Next 7 Days
                </h2>
                <p className="text-xs text-udemy-text-muted">
                  {forecast.total_due} card
                  {forecast.total_due === 1 ? "" : "s"} due
                </p>
              </div>
              <Link to="/review" className="btn-secondary text-sm inline-flex">
                Review now
              </Link>
            </div>
            <div className="flex items-end gap-1.5 mt-4 h-24">
              {forecast.due_counts.map((day) => {
                const maxCount = Math.max(
                  1,
                  ...forecast.due_counts.map((d) => d.count),
                );
                const pct = Math.round((day.count / maxCount) * 100);
                const label = new Date(
                  `${day.date}T00:00:00`,
                ).toLocaleDateString(undefined, {
                  weekday: "short",
                  month: "numeric",
                  day: "numeric",
                });
                return (
                  <div
                    key={day.date}
                    className="flex-1 flex flex-col items-center justify-end gap-1 h-full"
                    title={`${day.date}: ${day.count} due`}
                  >
                    <span className="text-[10px] text-udemy-text-muted">
                      {day.count}
                    </span>
                    <div className="w-full rounded-t bg-udemy-purple/15 h-16 flex items-end">
                      <div
                        className="w-full rounded-t bg-udemy-purple"
                        style={{
                          height: `${day.count > 0 ? Math.max(pct, 6) : 0}%`,
                        }}
                      />
                    </div>
                    <span className="text-[10px] text-udemy-text-muted text-center leading-tight">
                      {label}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      <div className="max-w-[1340px] mx-auto px-4 sm:px-6 pt-6">
        <div className="grid grid-cols-1 xl:grid-cols-[1.05fr,0.95fr] gap-4">
          <div className="udemy-card p-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-bold mb-1 flex items-center gap-2">
                  <Target className="w-4 h-4 text-udemy-purple" />
                  Goal Snapshot
                </h2>
                <p className="text-xs text-udemy-text-muted">
                  The recommendation engine now uses your learner profile and deadline.
                </p>
              </div>
              <Link to="/study-plan" className="btn-secondary text-sm inline-flex">
                Open plan
              </Link>
            </div>

            {!isAuthenticated ? (
              <p className="text-sm text-udemy-text-muted mt-4">
                Sign in to save a target role, deadline, and personalized recommendation stack.
              </p>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                <MiniStat
                  label="Target role"
                  value={profile?.target_role || "Unset"}
                />
                <MiniStat
                  label="Track"
                  value={
                    profile?.primary_track
                      ? TRACK_LABELS[profile.primary_track] || profile.primary_track
                      : "Unset"
                  }
                />
                <MiniStat
                  label="Weekly budget"
                  value={profile ? `${profile.weekly_minutes} min` : "Unset"}
                />
                <MiniStat
                  label="Deadline"
                  value={profile?.target_date || "Unset"}
                />
              </div>
            )}
          </div>

          <div className="udemy-card p-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-bold mb-1">Next Best Actions</h2>
                <p className="text-xs text-udemy-text-muted">
                  Highest-priority actions from your study planner.
                </p>
              </div>
              <Link to="/study-plan" className="text-sm text-udemy-purple font-medium">
                See all
              </Link>
            </div>

            <div className="space-y-3 mt-4">
              {!isAuthenticated ? (
                <p className="text-sm text-udemy-text-muted">
                  Sign in to unlock ranked next-step recommendations.
                </p>
              ) : recommendations.length === 0 ? (
                <p className="text-sm text-udemy-text-muted">
                  Save your learner profile in Study Plan to unlock recommendations.
                </p>
              ) : (
                recommendations.map((item) => (
                  <Link
                    key={item.recommendation_id}
                    to={item.cta_route}
                    className="block rounded-lg border border-udemy-border bg-white p-3 hover:bg-gray-50 transition-colors"
                  >
                    <p className="text-sm font-semibold">
                      {normalizeEscapedSingleLineText(item.title)}
                    </p>
                    <p className="text-xs text-udemy-text-muted mt-1">
                      {item.topic_id
                        ? topicMapLabel(topics, item.topic_id)
                        : TRACK_LABELS[(item.track as keyof typeof TRACK_LABELS)] || "Interview"}{" "}
                      · priority {Math.round(item.priority)} · ~{item.estimated_minutes} min
                    </p>
                  </Link>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-[1340px] mx-auto px-4 sm:px-6 pt-6">
        <div className="udemy-card p-4">
          <div className="flex flex-col md:flex-row md:items-end gap-3">
            <div className="flex-1">
              <h2 className="text-sm font-bold mb-1">Create Custom Topic</h2>
              <p className="text-xs text-udemy-text-muted mb-2">
                Build a deep roadmap (100+ subtopics) for any topic.
              </p>
              <input
                type="text"
                value={customTopic}
                onChange={(e) => setCustomTopic(e.target.value)}
                placeholder="Enter custom topic (e.g. Java, Kafka, Spring Boot)"
                className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:border-udemy-purple"
              />
            </div>
            <button
              onClick={handleCreateCustomTopic}
              className="btn-primary w-full sm:w-auto sm:min-w-[220px] inline-flex items-center justify-center gap-2"
            >
              Analyze & Build Roadmap
            </button>
          </div>
          {customError && (
            <p className="text-xs text-amber-800 bg-amber-50 border border-amber-300 rounded px-2 py-1 mt-3">
              {customError}
            </p>
          )}
        </div>
      </div>

      {/* Course grid */}
      <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-6">
          <h2 className="text-xl font-bold">Interview Topics</h2>
          <span className="text-sm text-udemy-text-muted">
            {topics.length} topics &middot; {completedCount} completed
          </span>
        </div>

        {loading ? (
          <SkeletonCards />
        ) : (
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5"
            variants={containerVariants}
            initial="hidden"
            animate="show"
          >
            {topics.map((topic) => {
              const tp = getTopicProgress(topic.id);
              const isComplete = completedTopics.includes(topic.id);
              const title = normalizeEscapedSingleLineText(topic.title);
              const description = normalizeEscapedSingleLineText(topic.description);
              return (
                <motion.div key={topic.id} variants={cardVariants}>
                  <Link to={`/topics/${topic.id}`} className="block group">
                    <div className="udemy-card overflow-hidden hover:-translate-y-1 transition-transform duration-200">
                      {/* Color strip at top */}
                      <div className="h-1.5 bg-gradient-to-r from-udemy-purple to-udemy-purple-light" />

                      <div className="p-5">
                        {/* Icon + title */}
                        <div className="flex items-start gap-3 mb-3">
                          <span className="text-2xl">
                            {TOPIC_ICONS[topic.id] || "📄"}
                          </span>
                          <div className="flex-1 min-w-0">
                            <h3 className="font-bold text-[15px] text-udemy-text leading-tight group-hover:text-udemy-purple transition-colors line-clamp-2">
                              {title}
                            </h3>
                          </div>
                          {isComplete && (
                            <span className="flex-shrink-0 w-6 h-6 bg-udemy-success rounded-full flex items-center justify-center">
                              <span className="text-white text-xs">✓</span>
                            </span>
                          )}
                        </div>

                        {/* Description */}
                        <p className="text-sm text-udemy-text-muted line-clamp-2 mb-4 leading-relaxed">
                          {description ||
                            "Explore this topic through interview questions."}
                        </p>

                        <div className="mb-3">
                          <span className="inline-flex items-center rounded-full bg-udemy-purple/10 px-2 py-1 text-[11px] font-semibold text-udemy-purple">
                            {TRACK_LABELS[topic.track] ||
                              topic.track ||
                              "Topic"}
                          </span>
                        </div>

                        {/* Progress bar */}
                        <ProgressBar percent={tp} className="mb-3" />

                        {/* Footer */}
                        <div className="flex items-center justify-between text-xs text-udemy-text-muted">
                          <span>{topic.section_count} sections</span>
                          <span className="flex items-center gap-1 text-udemy-purple font-medium group-hover:gap-2 transition-all">
                            {tp > 0 ? "Continue" : "Start"}
                            <ChevronRight className="w-3.5 h-3.5" />
                          </span>
                        </div>
                      </div>
                    </div>
                  </Link>
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

function Stat({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-2 bg-white/10 rounded-lg px-4 py-2.5">
      <span className="text-udemy-purple-light">{icon}</span>
      <div>
        <div className="text-xs text-gray-400">{label}</div>
        <div className="text-sm font-bold">{value}</div>
      </div>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-udemy-border px-3 py-2 bg-white">
      <p className="text-xs text-udemy-text-muted">{label}</p>
      <p className="text-sm font-bold line-clamp-2">{value}</p>
    </div>
  );
}

function topicMapLabel(topics: TopicSummary[], topicId: string): string {
  const found = topics.find((topic) => topic.id === topicId);
  return found ? normalizeEscapedSingleLineText(found.title) : normalizeEscapedSingleLineText(topicId);
}
