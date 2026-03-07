import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { AlertTriangle, CalendarDays, Loader2, RefreshCw } from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import { fetchStudyPlan, fetchTopics } from "@/services/api";
import type { StudyPlanResponse, TopicSummary } from "@/types";

export default function StudyPlanPage() {
  const [plan, setPlan] = useState<StudyPlanResponse | null>(null);
  const [topics, setTopics] = useState<TopicSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");

  const load = async () => {
    setLoading(true);
    setErrorMsg("");
    try {
      const [studyPlan, topicList] = await Promise.all([
        fetchStudyPlan(7, 3),
        fetchTopics(),
      ]);
      setPlan(studyPlan);
      setTopics(topicList);
    } catch (err) {
      console.error(err);
      setErrorMsg("Could not load study plan. Please retry.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const topicMap = useMemo(
    () => new Map(topics.map((topic) => [topic.id, topic.title])),
    [topics],
  );

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
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <CalendarDays className="w-6 h-6 text-udemy-purple" />
              7-Day Study Plan
            </h1>
            <p className="text-sm text-udemy-text-muted mt-1">
              Rule-based plan generated from weak areas and due review pressure.
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
            Regenerate
          </button>
        </div>

        {plan && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5">
            <Summary label="Days" value={`${plan.days}`} />
            <Summary label="Items / day" value={`${plan.daily_items}`} />
            <Summary label="Total tasks" value={`${plan.total_tasks}`} />
            <Summary
              label="Generated"
              value={new Date(plan.generated_at).toLocaleDateString()}
            />
          </div>
        )}

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
            Building your study plan...
          </div>
        ) : !plan || plan.days_plan.length === 0 ? (
          <div className="udemy-card p-8 text-center">
            <p className="font-semibold">No plan available yet.</p>
            <p className="text-sm text-udemy-text-muted mt-1">
              Complete more topic practice or quizzes to generate an actionable plan.
            </p>
            <Link to="/topics" className="btn-primary mt-4 inline-flex">
              Start studying
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {plan.days_plan.map((day) => (
              <div key={`${day.day_index}:${day.date}`} className="udemy-card p-4">
                <div className="flex items-center justify-between gap-2 mb-3">
                  <h2 className="text-lg font-bold">{day.label}</h2>
                  <span className="text-xs text-udemy-text-muted">{day.date}</span>
                </div>

                {day.tasks.length === 0 ? (
                  <p className="text-sm text-udemy-text-muted">
                    No tasks scheduled for this day.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {day.tasks.map((task, idx) => (
                      <div
                        key={`${day.day_index}:${idx}:${task.topic_id}:${task.task_type}`}
                        className="rounded-lg border border-udemy-border bg-white p-3"
                      >
                        <p className="text-sm font-semibold">{task.title}</p>
                        <p className="text-xs text-udemy-text-muted mt-1">
                          {topicMap.get(task.topic_id) || task.topic_id} · {task.task_type.replace("_", " ")} · ~
                          {task.estimated_minutes} min
                        </p>
                        <p className="text-xs text-udemy-text-muted mt-1">
                          {task.reason}
                        </p>
                        <Link
                          to={task.cta_route}
                          className="btn-secondary mt-3 inline-flex text-sm"
                        >
                          Open task
                        </Link>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-udemy-border px-3 py-2 bg-white">
      <p className="text-xs text-udemy-text-muted">{label}</p>
      <p className="text-lg font-bold">{value}</p>
    </div>
  );
}
