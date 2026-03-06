import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  BookOpen,
  Clock,
  TrendingUp,
  Sparkles,
  ChevronRight,
} from "lucide-react";
import {
  containerVariants,
  cardVariants,
  pageVariants,
  pageTransition,
} from "@/utils/animations";
import { fetchTopicMastery, fetchTopics } from "@/services/api";
import { useProgressStore } from "@/store/progressStore";
import ProgressBar from "@/components/common/ProgressBar";
import SkeletonCards from "@/components/common/SkeletonCards";
import type { TopicSummary } from "@/types";

const TOPIC_ICONS: Record<string, string> = {
  "01-system-map": "🗺️",
  "02-tech-stack": "⚙️",
  "03-backend-architecture": "🏗️",
  "04-frontend-architecture": "🖥️",
  "05-ai-services-and-grpc": "🤖",
  "06-data-model-and-migrations": "🗄️",
  "07-auth-security-and-risk-controls": "🔐",
  "08-design-patterns-in-this-codebase": "🧩",
  "09-endpoint-and-service-navigation": "🧭",
  "10-debugging-testing-and-operations": "🐛",
  "11-change-playbooks": "📋",
  "12-30-day-ownership-plan": "📅",
  "13-glossary": "📖",
  README: "📘",
};

export default function Dashboard() {
  const [topics, setTopics] = useState<TopicSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const completedTopics = useProgressStore((s) => s.completedTopics);
  const totalProgress = useProgressStore((s) => s.totalProgress);
  const getTopicProgress = useProgressStore((s) => s.getTopicProgress);
  const setTopicCount = useProgressStore((s) => s.setTopicCount);
  const setMastery = useProgressStore((s) => s.setMastery);

  useEffect(() => {
    fetchTopics()
      .then((loaded) => {
        setTopics(loaded);
        setTopicCount(loaded.length);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
    fetchTopicMastery()
      .then((res) => {
        res.topics.forEach((t) => setMastery(t.topic_id, t.mastery_score));
      })
      .catch(() => {
        // ignore mastery bootstrap failures
      });
  }, [setTopicCount, setMastery]);

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
        <div className="max-w-[1340px] mx-auto px-6 py-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <h1 className="text-3xl md:text-4xl font-bold mb-3">
              Polymarket System Study
            </h1>
            <p className="text-gray-300 text-lg max-w-2xl mb-6">
              Master every aspect of the Polymarket trading platform through
              AI-powered interview questions and deep-dive study sessions.
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
              value="Live"
            />
          </motion.div>
        </div>
      </div>

      {/* Overall progress */}
      <div className="max-w-[1340px] mx-auto px-6 -mt-3">
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

      {/* Course grid */}
      <div className="max-w-[1340px] mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold">Study Topics</h2>
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
