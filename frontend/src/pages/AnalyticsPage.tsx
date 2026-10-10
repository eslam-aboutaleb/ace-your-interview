import { useMemo } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  BarChart3,
  CalendarDays,
  Flame,
  Loader2,
  RefreshCw,
  Target,
  TrendingUp,
} from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import {
  fetchHeatmap,
  fetchMasteryCurve,
  fetchRetention,
} from "@/services/api";
import { useApi } from "@/hooks/useApi";
import type {
  HeatmapResponse,
  MasteryCurveResponse,
  RetentionResponse,
} from "@/types";

export default function AnalyticsPage() {
  const heatmap = useApi<HeatmapResponse>(
    async () => fetchHeatmap(26),
    { errorMessage: "Could not load the activity heatmap.", fixedErrorMessage: true },
  );
  const retention = useApi<RetentionResponse>(
    async () => fetchRetention(),
    { errorMessage: "Could not load retention buckets.", fixedErrorMessage: true },
  );
  const mastery = useApi<MasteryCurveResponse>(
    async () => fetchMasteryCurve(),
    { errorMessage: "Could not load the mastery curve.", fixedErrorMessage: true },
  );

  const loading = heatmap.loading || retention.loading || mastery.loading;
  const error = heatmap.error || retention.error || mastery.error;

  const heatmapWeeks = useMemo(() => {
    const days = heatmap.data?.days ?? [];
    if (days.length === 0) return [];
    const weeks: Array<Array<(typeof days)[number] | null>> = [];
    let current: Array<(typeof days)[number] | null> = [];
    for (const day of days) {
      current.push(day);
      if (current.length === 7) {
        weeks.push(current);
        current = [];
      }
    }
    if (current.length > 0) {
      while (current.length < 7) current.push(null);
      weeks.push(current);
    }
    return weeks;
  }, [heatmap.data]);

  const maxCount = useMemo(() => {
    const days = heatmap.data?.days ?? [];
    return Math.max(1, ...days.map((d) => d.count));
  }, [heatmap.data]);

  const masteryPoints = useMemo(() => mastery.data?.points ?? [], [mastery.data]);
  const masteryPath = useMemo(() => {
    if (masteryPoints.length < 2) return "";
    const width = 100;
    const height = 40;
    const step = width / Math.max(1, masteryPoints.length - 1);
    return masteryPoints
      .map(
        (point, index) =>
          `${index === 0 ? "M" : "L"}${(index * step).toFixed(2)},${(
            height -
            point.mastery * height
          ).toFixed(2)}`,
      )
      .join(" ");
  }, [masteryPoints]);

  const totalActivities = useMemo(() => {
    const days = heatmap.data?.days ?? [];
    return days.reduce((sum, day) => sum + day.count, 0);
  }, [heatmap.data]);

  const activeDays = useMemo(() => {
    const days = heatmap.data?.days ?? [];
    return days.filter((day) => day.count > 0).length;
  }, [heatmap.data]);

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
              <BarChart3 className="w-6 h-6 text-udemy-purple" />
              Study Analytics
            </h1>
            <p className="text-sm text-udemy-text-muted mt-1">
              Activity heatmap, retention buckets, and mastery curve.
            </p>
          </div>
          <button
            onClick={() => {
              void heatmap.retry();
              void retention.retry();
              void mastery.retry();
            }}
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

        {error && (
          <div
            className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2"
            role="alert"
          >
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {heatmap.data && (
          <section className="mt-6">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <CalendarDays className="w-5 h-5 text-udemy-purple" />
              Activity
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
              <Metric label="Total events" value={`${totalActivities}`} />
              <Metric label="Active days" value={`${activeDays}`} />
              <Metric
                label="Last 26 weeks"
                value={`${heatmap.data.weeks}`}
              />
            </div>
            <div className="mt-4 overflow-x-auto">
              <div className="inline-flex gap-[3px]">
                {heatmapWeeks.map((week, weekIndex) => (
                  <div key={weekIndex} className="flex flex-col gap-[3px]">
                    {week.map((day, dayIndex) => (
                      <div
                        key={dayIndex}
                        title={
                          day
                            ? `${day.date}: ${day.count} event${day.count === 1 ? "" : "s"}`
                            : undefined
                        }
                        className="w-[12px] h-[12px] rounded-[2px]"
                        style={{
                          backgroundColor: day
                            ? day.count === 0
                              ? "#ebedf0"
                              : `rgba(22, 163, 74, ${
                                  0.25 +
                                  0.75 * (day.count / maxCount)
                                })`
                            : "transparent",
                        }}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-2 mt-3 text-xs text-udemy-text-muted">
              <span>Less</span>
              {[0.1, 0.35, 0.6, 0.85, 1].map((opacity) => (
                <div
                  key={opacity}
                  className="w-[12px] h-[12px] rounded-[2px]"
                  style={{
                    backgroundColor: `rgba(22, 163, 74, ${opacity})`,
                  }}
                />
              ))}
              <span>More</span>
            </div>
          </section>
        )}

        {retention.data && (
          <section className="mt-8">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Target className="w-5 h-5 text-udemy-purple" />
              Retention
            </h2>
            <p className="text-sm text-udemy-text-muted mt-1">
              Recall rate at 1-day, 7-day, and 30-day intervals.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-3">
              {retention.data.buckets.map((bucket) => (
                <div
                  key={bucket.bucket}
                  className="rounded-lg border border-udemy-border p-4"
                >
                  <div className="text-sm font-medium text-udemy-text-muted">
                    {bucket.bucket} recall
                  </div>
                  <div className="mt-1 text-2xl font-bold">
                    {Math.round(bucket.recall_rate * 100)}%
                  </div>
                  <div className="text-xs text-udemy-text-muted mt-1">
                    {bucket.successful}/{bucket.reviews} reviews
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {mastery.data && (
          <section className="mt-8">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-udemy-purple" />
              Mastery curve
            </h2>
            <p className="text-sm text-udemy-text-muted mt-1">
              Exponential moving average of correctness over time.
            </p>
            {masteryPath ? (
              <div className="mt-4">
                <svg
                  viewBox="0 0 100 40"
                  preserveAspectRatio="none"
                  className="w-full h-40"
                  role="img"
                  aria-label="Mastery curve"
                >
                  <path
                    d={masteryPath}
                    fill="none"
                    stroke="#7c3aed"
                    strokeWidth="1.5"
                    vectorEffect="non-scaling-stroke"
                  />
                </svg>
                <div className="flex justify-between text-xs text-udemy-text-muted mt-1">
                  <span>{masteryPoints[0]?.date}</span>
                  <span>{masteryPoints[masteryPoints.length - 1]?.date}</span>
                </div>
              </div>
            ) : (
              <div className="mt-4 rounded-lg border border-udemy-border p-6 text-center text-sm text-udemy-text-muted">
                <Flame className="w-5 h-5 mx-auto mb-2 opacity-50" />
                Complete some study sessions to build the mastery curve.
              </div>
            )}
          </section>
        )}
      </div>
    </motion.div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-udemy-border p-3">
      <div className="text-xs text-udemy-text-muted">{label}</div>
      <div className="mt-1 text-xl font-bold">{value}</div>
    </div>
  );
}
