import { useEffect, useState, useCallback, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { BookOpen, Search, ChevronRight, Filter } from "lucide-react";
import {
  pageVariants,
  pageTransition,
  containerVariants,
  cardVariants,
} from "@/utils/animations";
import { fetchTopics } from "@/services/api";
import { useProgressStore } from "@/store/progressStore";
import ProgressBar from "@/components/common/ProgressBar";
import SkeletonCards from "@/components/common/SkeletonCards";
import { normalizeEscapedSingleLineText } from "@/utils/textNormalization";
import type { InterviewLevel, LearningTrack, TopicSummary } from "@/types";

const TRACK_LABELS: Record<LearningTrack, string> = {
  backend: "Backend",
  frontend: "Frontend",
  system_design: "System Design",
  ai_stack: "AI Stack",
};

const LEVEL_LABELS: Record<InterviewLevel, string> = {
  junior: "Junior",
  mid: "Mid",
  senior: "Senior",
};

const MIN_STATIC_TOPIC_SECTIONS = 40;
const MIN_CUSTOM_TOPIC_SECTIONS = 100;

function topicHasCoverageGap(topic: TopicSummary): boolean {
  const minSections = topic.id.startsWith("custom-")
    ? MIN_CUSTOM_TOPIC_SECTIONS
    : MIN_STATIC_TOPIC_SECTIONS;
  return topic.section_count < minSections;
}

export default function TopicsList() {
  const navigate = useNavigate();
  const [topics, setTopics] = useState<TopicSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [track, setTrack] = useState<LearningTrack | "">("");
  const [level, setLevel] = useState<InterviewLevel | "">("");
  const [customTopic, setCustomTopic] = useState("");
  const [customError, setCustomError] = useState("");
  const completedTopics = useProgressStore((s) => s.completedTopics);
  const getTopicProgress = useProgressStore((s) => s.getTopicProgress);
  const gapFreeCount = useMemo(
    () => topics.filter((topic) => !topicHasCoverageGap(topic)).length,
    [topics],
  );
  const coverageGapCount = topics.length - gapFreeCount;

  const loadTopics = useCallback(async () => {
    setLoading(true);
    try {
      const loaded = await fetchTopics({
        track: track || undefined,
        level: level || undefined,
        q: search.trim() || undefined,
      });
      setTopics(loaded);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [search, track, level]);

  useEffect(() => {
    loadTopics();
  }, [loadTopics]);

  const handleCreateCustomTopic = () => {
    const topic = customTopic.trim();
    if (topic.length < 2) {
      setCustomError("Please enter at least 2 characters for the custom topic.");
      return;
    }
    setCustomError("");
    setCustomTopic("");
    navigate(`/topics/custom/build?topic=${encodeURIComponent(topic)}`);
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
            <BookOpen className="w-7 h-7 text-udemy-purple-light" />
            Interview Topics
          </h1>
          <p className="text-gray-400 mt-2">
            {topics.length} topics &middot; {gapFreeCount} gap-free &middot; {completedTopics.length} completed
          </p>
        </div>
      </div>

      <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-6">
        {/* Search */}
        <div className="relative mb-6">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-udemy-text-muted" />
          <input
            type="text"
            placeholder="Search topics..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full border border-udemy-border rounded-lg pl-10 pr-4 py-3 text-sm focus:outline-none focus:border-udemy-purple transition-colors"
          />
          <span className="hidden sm:flex absolute right-3 top-1/2 -translate-y-1/2 text-xs text-udemy-text-muted items-center gap-1">
            <Filter className="w-3.5 h-3.5" />
            {topics.length} results
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
          <select
            value={track}
            onChange={(e) => setTrack(e.target.value as LearningTrack | "")}
            className="border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
          >
            <option value="">All tracks</option>
            <option value="backend">Backend</option>
            <option value="frontend">Frontend</option>
            <option value="system_design">System Design</option>
            <option value="ai_stack">AI Stack</option>
          </select>
          <select
            value={level}
            onChange={(e) => setLevel(e.target.value as InterviewLevel | "")}
            className="border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
          >
            <option value="">All levels</option>
            <option value="junior">Junior</option>
            <option value="mid">Mid</option>
            <option value="senior">Senior</option>
          </select>
        </div>

        <div className="udemy-card p-4 mb-6">
          <h2 className="text-sm font-bold mb-2">Create Custom Topic</h2>
          <p className="text-xs text-udemy-text-muted mb-3">
            Enter any topic to generate a deep roadmap with 100+ subtopics and full learning-path coverage.
          </p>
          <div className="flex flex-col md:flex-row gap-3">
            <input
              type="text"
              value={customTopic}
              onChange={(e) => setCustomTopic(e.target.value)}
              placeholder="Enter custom topic (e.g. Java, Kafka, Spring Boot)"
              className="flex-1 border border-udemy-border rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:border-udemy-purple"
            />
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

        {coverageGapCount > 0 && (
          <div className="mb-5 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
            {coverageGapCount} topic(s) are currently flagged with possible coverage gaps.
          </div>
        )}

        {loading ? (
          <SkeletonCards count={9} />
        ) : (
          <motion.div
            className="space-y-3"
            variants={containerVariants}
            initial="hidden"
            animate="show"
          >
            {topics.map((topic, idx) => {
              const progress = getTopicProgress(topic.id);
              const isComplete = completedTopics.includes(topic.id);
              const hasCoverageGap = topicHasCoverageGap(topic);
              const title = normalizeEscapedSingleLineText(topic.title);
              const description = normalizeEscapedSingleLineText(topic.description);
              return (
                <motion.div key={topic.id} variants={cardVariants}>
                  <Link
                    to={`/topics/${topic.id}`}
                    className="block group udemy-card hover:-translate-y-0.5 transition-transform duration-200"
                  >
                    <div className="flex items-center gap-3 sm:gap-5 p-4 sm:p-5">
                      {/* Number */}
                      <div
                        className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0 ${
                          isComplete
                            ? "bg-udemy-success text-white"
                            : "bg-udemy-purple/10 text-udemy-purple"
                        }`}
                      >
                        {isComplete ? "✓" : idx + 1}
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <h3 className="font-bold text-[15px] group-hover:text-udemy-purple transition-colors">
                          {title}
                        </h3>
                        <p className="text-sm text-udemy-text-muted line-clamp-1 mt-0.5">
                          {description}
                        </p>
                        <div className="flex items-center gap-2 mt-2">
                          <span className="text-[11px] font-semibold rounded-full bg-udemy-purple/10 text-udemy-purple px-2 py-0.5">
                            {TRACK_LABELS[topic.track] || topic.track || "Topic"}
                          </span>
                          <span
                            className={`text-[11px] font-semibold rounded-full px-2 py-0.5 ${
                              hasCoverageGap
                                ? "bg-amber-100 text-amber-800"
                                : "bg-green-100 text-green-700"
                            }`}
                          >
                            {hasCoverageGap ? "Coverage gap" : "Gap-free path"}
                          </span>
                          <span className="text-[11px] text-udemy-text-muted">
                            {(topic.levels || [])
                              .map((lv) => LEVEL_LABELS[lv] || lv)
                              .join(" / ")}
                          </span>
                        </div>
                        <div className="flex flex-wrap items-center gap-2 sm:gap-4 mt-2">
                          <span className="text-xs text-udemy-text-muted">
                            {topic.section_count} sections
                          </span>
                          <span className="text-xs text-udemy-text-muted">
                            ~{topic.estimated_questions} questions
                          </span>
                          <div className="w-full sm:w-auto flex-1 sm:max-w-[200px]">
                            <ProgressBar percent={progress} />
                          </div>
                        </div>
                      </div>

                      {/* Arrow */}
                      <ChevronRight className="w-5 h-5 text-udemy-text-muted group-hover:text-udemy-purple group-hover:translate-x-1 transition-all flex-shrink-0" />
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
