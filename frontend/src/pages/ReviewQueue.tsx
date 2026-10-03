import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { AlertTriangle, Clock3, Loader2, RefreshCw } from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import { fetchReviewQueue, fetchTopics, submitCardReview } from "@/services/api";
import { normalizeEscapedSingleLineText } from "@/utils/textNormalization";
import type {
  LearningReviewRating,
  ReviewQueueItem,
  TopicSummary,
} from "@/types";

function formatDateTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

const RATING_BUTTONS: {
  rating: LearningReviewRating;
  label: string;
  className: string;
}[] = [
  {
    rating: "again",
    label: "Again",
    className: "btn-secondary rating-again",
  },
  {
    rating: "hard",
    label: "Hard",
    className: "btn-secondary rating-hard",
  },
  {
    rating: "good",
    label: "Good",
    className: "btn-primary rating-good",
  },
  {
    rating: "easy",
    label: "Easy",
    className: "btn-secondary rating-easy",
  },
];

export default function ReviewQueue() {
  const [items, setItems] = useState<ReviewQueueItem[]>([]);
  const [topics, setTopics] = useState<TopicSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");
  const [submitting, setSubmitting] = useState<Record<string, boolean>>(
    {},
  );
  const [gradeError, setGradeError] = useState<
    Record<string, string>
  >({});
  const [loadedAt, setLoadedAt] = useState<Record<string, number>>({});

  const load = async () => {
    setLoading(true);
    setErrorMsg("");
    try {
      const [queue, topicList] = await Promise.all([
        fetchReviewQueue(200),
        fetchTopics(),
      ]);
      setItems(queue.items || []);
      setTopics(topicList || []);
      const now = Date.now();
      const stamps: Record<string, number> = {};
      for (const item of queue.items || []) {
        stamps[item.question_id] = now;
      }
      setLoadedAt(stamps);
      setSubmitting({});
      setGradeError({});
    } catch (err) {
      console.error(err);
      setErrorMsg("Could not load review queue. Please retry.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const topicMap = useMemo(() => {
    return new Map(
      topics.map((topic) => [topic.id, normalizeEscapedSingleLineText(topic.title)]),
    );
  }, [topics]);

  const overdueNow = useMemo(() => {
    const now = Date.now();
    return items.filter((item) => new Date(item.due_at).getTime() <= now).length;
  }, [items]);

  const avgMasteryPct = useMemo(() => {
    if (!items.length) return 0;
    const total = items.reduce((sum, item) => sum + item.mastery_score, 0);
    return Math.round((total / items.length) * 100);
  }, [items]);

  const leechCount = useMemo(
    () => items.filter((item) => item.leech).length,
    [items],
  );

  const handleGrade = async (
    item: ReviewQueueItem,
    rating: LearningReviewRating,
  ) => {
    const key = item.question_id;
    if (submitting[key] || item.suspended) return;
    setSubmitting((prev) => ({ ...prev, [key]: true }));
    setGradeError((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
    const responseTimeMs = loadedAt[key]
      ? Math.max(0, Date.now() - loadedAt[key])
      : 0;
    try {
      await submitCardReview({
        card_id: item.question_id,
        topic_id: item.topic_id,
        rating,
        response_time_ms: responseTimeMs,
        source_type: "question",
      });
      await load();
    } catch (err) {
      console.error(err);
      setGradeError((prev) => ({
        ...prev,
        [key]: "Could not submit review. Please retry.",
      }));
      setSubmitting((prev) => ({ ...prev, [key]: false }));
    }
  };

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
      className="max-w-[1200px] mx-auto px-4 sm:px-6 py-8"
    >
      <div className="udemy-card p-6">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold">Review Queue</h1>
            <p className="text-sm text-udemy-text-muted mt-1">
              Due items prioritized by mastery and schedule.
            </p>
          </div>
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
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 mt-5">
          <SummaryChip label="Total due" value={`${items.length}`} />
          <SummaryChip label="Overdue now" value={`${overdueNow}`} />
          <SummaryChip label="Average mastery" value={`${avgMasteryPct}%`} />
          <SummaryChip label="Leeches" value={`${leechCount}`} />
        </div>

        {errorMsg && (
          <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>

      <div className="mt-6">
        {loading ? (
          <div className="udemy-card p-8 flex items-center justify-center gap-2 text-sm text-udemy-text-muted">
            <Loader2 className="w-5 h-5 animate-spin text-udemy-purple" />
            Loading review queue...
          </div>
        ) : items.length === 0 ? (
          <div className="udemy-card p-8 text-center">
            <Clock3 className="w-8 h-8 mx-auto text-udemy-text-muted mb-3" />
            <p className="font-semibold">No due items right now.</p>
            <p className="text-sm text-udemy-text-muted mt-1">
              Keep practicing topics and quizzes to build your adaptive queue.
            </p>
            <Link to="/topics" className="btn-primary mt-4 inline-flex">
              Explore Topics
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {items.map((item) => {
              const topicId = normalizeEscapedSingleLineText(item.topic_id);
              const questionId = normalizeEscapedSingleLineText(item.question_id);
              const isSubmitting = Boolean(submitting[item.question_id]);
              const itemError = gradeError[item.question_id];
              const isSuspended = Boolean(item.suspended);
              return (
                <div
                  key={`${item.topic_id}:${item.question_id}`}
                  className="udemy-card p-4"
                >
                  <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                    <div className="min-w-0">
                      <p className="text-base font-semibold truncate">
                        {topicMap.get(item.topic_id) || topicId}
                      </p>
                      <p className="text-xs text-udemy-text-muted mt-1 break-all">
                        {topicId} · {questionId}
                      </p>
                      <div className="flex flex-wrap gap-2 mt-2 text-xs">
                        <Badge>{`Due: ${formatDateTime(item.due_at)}`}</Badge>
                        <Badge>{`Mastery: ${Math.round(item.mastery_score * 100)}%`}</Badge>
                        <Badge>{`State: ${item.state || "new"}`}</Badge>
                        <Badge>{`Reps: ${item.attempts}`}</Badge>
                        <Badge>{`Lapses: ${item.lapses ?? 0}`}</Badge>
                        {item.leech && (
                          <Badge>
                            <span className="inline-flex items-center gap-1 text-red-700">
                              <AlertTriangle className="w-3 h-3" />
                              Leech
                            </span>
                          </Badge>
                        )}
                        {isSuspended && <Badge>Suspended</Badge>}
                      </div>
                      {itemError && (
                        <p className="text-xs text-red-700 mt-2">
                          {itemError}
                        </p>
                      )}
                    </div>
                    <div className="flex flex-col gap-2">
                      <div className="flex flex-wrap gap-2">
                        {RATING_BUTTONS.map(({ rating, label, className }) => (
                          <button
                            key={rating}
                            onClick={() => handleGrade(item, rating)}
                            disabled={isSubmitting || isSuspended}
                            className={`${className} disabled:opacity-50 min-w-[72px]`}
                          >
                            {isSubmitting ? (
                              <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                              label
                            )}
                          </button>
                        ))}
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <Link
                          to={`/topics/${item.topic_id}`}
                          className="btn-secondary"
                        >
                          Study now
                        </Link>
                        <Link
                          to={`/quiz/${item.topic_id}`}
                          className="btn-primary"
                        >
                          Quiz this topic
                        </Link>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </motion.div>
  );
}

function SummaryChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-udemy-border px-3 py-2 bg-white">
      <p className="text-xs text-udemy-text-muted">{label}</p>
      <p className="text-lg font-bold">{value}</p>
    </div>
  );
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full bg-udemy-bg text-udemy-text-muted px-2 py-0.5">
      {children}
    </span>
  );
}
