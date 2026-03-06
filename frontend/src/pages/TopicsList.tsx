import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
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
import type { TopicSummary } from "@/types";

export default function TopicsList() {
  const [topics, setTopics] = useState<TopicSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const completedTopics = useProgressStore((s) => s.completedTopics);
  const getTopicProgress = useProgressStore((s) => s.getTopicProgress);

  useEffect(() => {
    fetchTopics()
      .then(setTopics)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const filtered = topics.filter(
    (t) =>
      t.title.toLowerCase().includes(search.toLowerCase()) ||
      t.description.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
    >
      <div className="bg-udemy-dark text-white">
        <div className="max-w-[1340px] mx-auto px-6 py-8">
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
            <BookOpen className="w-7 h-7 text-udemy-purple-light" />
            All Topics
          </h1>
          <p className="text-gray-400 mt-2">
            {topics.length} topics &middot; {completedTopics.length} completed
          </p>
        </div>
      </div>

      <div className="max-w-[1340px] mx-auto px-6 py-6">
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
          {search && (
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-udemy-text-muted flex items-center gap-1">
              <Filter className="w-3.5 h-3.5" />
              {filtered.length} results
            </span>
          )}
        </div>

        {loading ? (
          <SkeletonCards count={9} />
        ) : (
          <motion.div
            className="space-y-3"
            variants={containerVariants}
            initial="hidden"
            animate="show"
          >
            {filtered.map((topic, idx) => {
              const progress = getTopicProgress(topic.id);
              const isComplete = completedTopics.includes(topic.id);
              return (
                <motion.div key={topic.id} variants={cardVariants}>
                  <Link
                    to={`/topics/${topic.id}`}
                    className="block group udemy-card hover:-translate-y-0.5 transition-transform duration-200"
                  >
                    <div className="flex items-center gap-5 p-5">
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
                          {topic.title}
                        </h3>
                        <p className="text-sm text-udemy-text-muted line-clamp-1 mt-0.5">
                          {topic.description}
                        </p>
                        <div className="flex items-center gap-4 mt-2">
                          <span className="text-xs text-udemy-text-muted">
                            {topic.section_count} sections
                          </span>
                          <span className="text-xs text-udemy-text-muted">
                            ~{topic.estimated_questions} questions
                          </span>
                          <div className="flex-1 max-w-[200px]">
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
