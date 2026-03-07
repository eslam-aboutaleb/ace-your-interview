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
  Loader2,
} from "lucide-react";
import {
  containerVariants,
  cardVariants,
  pageVariants,
  pageTransition,
} from "@/utils/animations";
import {
  createCustomTopic,
  fetchInterviewStats,
  fetchTopicMastery,
  fetchTopics,
} from "@/services/api";
import { useProgressStore } from "@/store/progressStore";
import { useSettingsStore } from "@/store/settingsStore";
import ProgressBar from "@/components/common/ProgressBar";
import SkeletonCards from "@/components/common/SkeletonCards";
import type { TopicSummary } from "@/types";

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
  const settings = useSettingsStore();
  const [topics, setTopics] = useState<TopicSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [customTopic, setCustomTopic] = useState("");
  const [creatingCustom, setCreatingCustom] = useState(false);
  const [customError, setCustomError] = useState("");
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
  }, [loadTopics, setMastery]);

  const handleCreateCustomTopic = async () => {
    const topic = customTopic.trim();
    if (topic.length < 2) {
      setCustomError("Please enter at least 2 characters for the custom topic.");
      return;
    }

    setCreatingCustom(true);
    setCustomError("");
    try {
      const created = await createCustomTopic({
        topic,
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
      });
      await loadTopics();
      setCustomTopic("");
      navigate(`/topics/${created.id}`);
    } catch (err) {
      console.error(err);
      setCustomError("Could not generate the custom topic roadmap. Please try again.");
    } finally {
      setCreatingCustom(false);
    }
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
        </div>
      </div>

      {curriculumNoticePending && (
        <div className="max-w-[1340px] mx-auto px-4 sm:px-6 pt-4">
          <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 flex items-center justify-between gap-3">
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
              disabled={creatingCustom}
              className="btn-primary min-w-[220px] disabled:opacity-60 disabled:cursor-not-allowed inline-flex items-center justify-center gap-2"
            >
              {creatingCustom && <Loader2 className="w-4 h-4 animate-spin" />}
              {creatingCustom ? "Analyzing..." : "Analyze & Build Roadmap"}
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
        <div className="flex items-center justify-between mb-6">
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
                              {topic.title}
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
                          {topic.description ||
                            "Explore this topic through interview questions."}
                        </p>

                        <div className="mb-3">
                          <span className="inline-flex items-center rounded-full bg-udemy-purple/10 px-2 py-1 text-[11px] font-semibold text-udemy-purple">
                            {TRACK_LABELS[topic.track] || topic.track || "Topic"}
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
