import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  BarChart3,
  CalendarDays,
  Flame,
  Gauge,
  Loader2,
  RefreshCw,
  Target,
  TrendingUp,
} from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import {
  fetchCalibration,
  fetchHeatmap,
  fetchMasteryCurve,
  fetchRetention,
} from "@/services/api";
import { useApi } from "@/hooks/useApi";
import type {
  CalibrationResponse,
  HeatmapDay,
  HeatmapResponse,
  MasteryCurveResponse,
  RetentionResponse,
} from "@/types";

/** Event types the backend heatmap payload breaks each day into. */
const EVENT_TYPES = [
  { key: "attempt", label: "Attempts", rgb: "22, 163, 74" },
  { key: "review", label: "Reviews", rgb: "124, 58, 237" },
  { key: "exam", label: "Exams", rgb: "245, 158, 11" },
] as const;

type EventTypeKey = (typeof EVENT_TYPES)[number]["key"];
type EventFilter = "all" | EventTypeKey;

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
  const calibration = useApi<CalibrationResponse>(
    async () => fetchCalibration(),
    {
      errorMessage: "Could not load calibration data.",
      fixedErrorMessage: true,
    },
  );
  const [eventFilter, setEventFilter] = useState<EventFilter>("all");

  const loading =
    heatmap.loading ||
    retention.loading ||
    mastery.loading ||
    calibration.loading;
  const error =
    heatmap.error || retention.error || mastery.error || calibration.error;

  const heatmapDays = useMemo(
    () => heatmap.data?.days ?? [],
    [heatmap.data],
  );

  const heatmapWeeks = useMemo(() => {
    if (heatmapDays.length === 0) return [];
    const weeks: Array<Array<HeatmapDay | null>> = [];
    let current: Array<HeatmapDay | null> = [];
    for (const day of heatmapDays) {
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
  }, [heatmapDays]);

  const dayCount = (day: HeatmapDay) =>
    eventFilter === "all" ? day.count : (day.types[eventFilter] ?? 0);

  const maxCount = useMemo(
    () => Math.max(1, ...heatmapDays.map(dayCount)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [heatmapDays, eventFilter],
  );

  const eventTotals = useMemo(() => {
    const totals: Record<EventTypeKey, number> = {
      attempt: 0,
      review: 0,
      exam: 0,
    };
    for (const day of heatmapDays) {
      for (const type of EVENT_TYPES) {
        totals[type.key] += day.types[type.key] ?? 0;
      }
    }
    return totals;
  }, [heatmapDays]);

  const filteredTotal = useMemo(
    () => (eventFilter === "all" ? 0 : eventTotals[eventFilter]),
    [eventFilter, eventTotals],
  );

  const activeColor =
    EVENT_TYPES.find((type) => type.key === eventFilter)?.rgb ?? "22, 163, 74";

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

  const totalActivities = useMemo(
    () => heatmapDays.reduce((sum, day) => sum + day.count, 0),
    [heatmapDays],
  );

  const activeDays = useMemo(
    () => heatmapDays.filter((day) => day.count > 0).length,
    [heatmapDays],
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
              <BarChart3 className="w-6 h-6 text-udemy-purple" />
              Study Analytics
            </h1>
            <p className="text-sm text-udemy-text-muted mt-1">
              Activity heatmap, retention buckets, mastery curve, and
              confidence calibration.
            </p>
          </div>
          <button
            onClick={() => {
              void heatmap.retry();
              void retention.retry();
              void mastery.retry();
              void calibration.retry();
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
              <Metric
                label={eventFilter === "all" ? "Event types" : "Filtered events"}
                value={
                  eventFilter === "all"
                    ? `${EVENT_TYPES.length}`
                    : `${filteredTotal}`
                }
              />
            </div>
            {/* Legend + toggle: each chip filters the heatmap below. */}
            <div className="flex flex-wrap items-center gap-2 mt-4">
              {EVENT_TYPES.map((type) => {
                const active = eventFilter === type.key;
                return (
                  <button
                    key={type.key}
                    type="button"
                    aria-pressed={active}
                    onClick={() =>
                      setEventFilter(active ? "all" : type.key)
                    }
                    title={
                      active
                        ? `Show all event types`
                        : `Show only ${type.label.toLowerCase()}`
                    }
                    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors ${
                      active
                        ? "border-udemy-purple bg-udemy-purple/10 text-udemy-purple"
                        : "border-udemy-border bg-white text-udemy-text-muted hover:bg-gray-50"
                    }`}
                  >
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: `rgb(${type.rgb})` }}
                    />
                    {type.label}
                    <span className="font-bold">{eventTotals[type.key]}</span>
                  </button>
                );
              })}
              {eventFilter !== "all" && (
                <button
                  type="button"
                  onClick={() => setEventFilter("all")}
                  className="inline-flex items-center gap-1 rounded-full border border-udemy-border bg-white px-2.5 py-1 text-xs font-medium text-udemy-text-muted hover:bg-gray-50 transition-colors"
                >
                  Clear filter
                </button>
              )}
            </div>
            <div className="mt-4 overflow-x-auto">
              <div className="inline-flex gap-[3px]">
                {heatmapWeeks.map((week, weekIndex) => (
                  <div key={weekIndex} className="flex flex-col gap-[3px]">
                    {week.map((day, dayIndex) => {
                      const count = day ? dayCount(day) : 0;
                      return (
                        <div
                          key={dayIndex}
                          title={
                            day
                              ? eventFilter === "all"
                                ? `${day.date}: ${day.count} event${day.count === 1 ? "" : "s"} (${EVENT_TYPES.map(
                                    (type) =>
                                      `${day.types[type.key] ?? 0} ${type.label.toLowerCase()}`,
                                  ).join(", ")})`
                                : `${day.date}: ${count} ${EVENT_TYPES.find(
                                    (type) => type.key === eventFilter,
                                  )?.label.toLowerCase() ?? "events"}`
                              : undefined
                          }
                          className="w-[12px] h-[12px] rounded-[2px]"
                          style={{
                            backgroundColor: day
                              ? count === 0
                                ? "#ebedf0"
                                : `rgba(${activeColor}, ${
                                    0.25 + 0.75 * (count / maxCount)
                                  })`
                              : "transparent",
                          }}
                        />
                      );
                    })}
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
                    backgroundColor: `rgba(${activeColor}, ${opacity})`,
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

        {calibration.data && calibration.data.eligible_reviews > 0 && (
          <section className="mt-8">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Gauge className="w-5 h-5 text-udemy-purple" />
              Confidence calibration
            </h2>
            <p className="text-sm text-udemy-text-muted mt-1">
              True retention vs. the FSRS target band, from spaced
              reviews (at least one day apart).
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
              <Metric
                label="True retention"
                value={`${Math.round(calibration.data.true_retention * 100)}%`}
              />
              <Metric
                label="Target band"
                value={`${Math.round(
                  calibration.data.target_band_low * 100,
                )}–${Math.round(calibration.data.target_band_high * 100)}%`}
              />
              <Metric
                label="Eligible reviews"
                value={`${calibration.data.eligible_reviews}`}
                hint={`${calibration.data.successful_reviews} successful`}
              />
              <Metric
                label="Good rating share"
                value={`${Math.round(calibration.data.good_rating_pct * 100)}%`}
              />
            </div>
            <div className="mt-4">
              <div className="relative h-2 rounded-full bg-gray-200">
                <div
                  className="absolute h-2 rounded-full bg-udemy-purple/30"
                  style={{
                    left: `${calibration.data.target_band_low * 100}%`,
                    width: `${
                      (calibration.data.target_band_high -
                        calibration.data.target_band_low) *
                      100
                    }%`,
                  }}
                />
                <div
                  className="absolute top-1/2 h-3 w-3 -translate-y-1/2 rounded-full border-2 border-white bg-udemy-purple"
                  style={{
                    left: `calc(${calibration.data.true_retention * 100}% - 6px)`,
                  }}
                  title={`True retention: ${Math.round(
                    calibration.data.true_retention * 100,
                  )}%`}
                />
              </div>
              <div className="flex justify-between text-xs text-udemy-text-muted mt-1">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>
            <p className="mt-3 text-sm">
              {calibration.data.within_band
                ? "Within the target band — your review schedule is dialed in."
                : calibration.data.true_retention <
                    calibration.data.target_band_low
                  ? "Below the target band: cards may feel too easy, or ratings may be generous."
                  : "Above the target band: cards may feel too hard, so reviews pile up."}
            </p>
            {calibration.data.low_signal && (
              <div
                className="mt-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2"
                role="note"
              >
                <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>
                  Low signal: fewer than 50 eligible reviews or a high
                  &ldquo;Good&rdquo; share. Treat this estimate as
                  provisional.
                </span>
              </div>
            )}
          </section>
        )}

        {calibration.data && calibration.data.eligible_reviews === 0 && (
          <section className="mt-8">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Gauge className="w-5 h-5 text-udemy-purple" />
              Confidence calibration
            </h2>
            <div className="mt-4 rounded-lg border border-udemy-border p-6 text-center text-sm text-udemy-text-muted">
              <Flame className="w-5 h-5 mx-auto mb-2 opacity-50" />
              Complete some spaced reviews to build the calibration
              estimate.
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

function Metric({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-udemy-border p-3">
      <div className="text-xs text-udemy-text-muted">{label}</div>
      <div className="mt-1 text-xl font-bold">{value}</div>
      {hint && (
        <div className="text-xs text-udemy-text-muted mt-0.5">{hint}</div>
      )}
    </div>
  );
}
