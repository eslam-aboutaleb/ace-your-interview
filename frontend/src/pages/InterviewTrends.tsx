import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { AlertTriangle, LineChart, Loader2, RefreshCw } from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import { fetchInterviewTrends } from "@/services/api";
import type { InterviewTrendPoint, InterviewTrendsResponse } from "@/types";

type RubricKey =
  | "technical_accuracy"
  | "reasoning_depth"
  | "communication_clarity"
  | "completeness"
  | "confidence_signal";

const DIMENSIONS: Array<{ key: RubricKey; label: string; color: string }> = [
  { key: "technical_accuracy", label: "Technical", color: "#16a34a" },
  { key: "reasoning_depth", label: "Reasoning", color: "#2563eb" },
  { key: "communication_clarity", label: "Communication", color: "#d97706" },
  { key: "completeness", label: "Completeness", color: "#dc2626" },
  { key: "confidence_signal", label: "Confidence", color: "#0ea5e9" },
];

export default function InterviewTrendsPage() {
  const [data, setData] = useState<InterviewTrendsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");
  const [enabledDims, setEnabledDims] = useState<Record<RubricKey, boolean>>({
    technical_accuracy: true,
    reasoning_depth: true,
    communication_clarity: false,
    completeness: false,
    confidence_signal: false,
  });

  const load = async () => {
    setLoading(true);
    setErrorMsg("");
    try {
      const trends = await fetchInterviewTrends(50);
      setData(trends);
    } catch (err) {
      console.error(err);
      setErrorMsg("Could not load interview trends.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const points = useMemo<InterviewTrendPoint[]>(
    () => data?.points || [],
    [data],
  );

  const toggleDim = (key: RubricKey) => {
    setEnabledDims((prev) => ({ ...prev, [key]: !prev[key] }));
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
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <LineChart className="w-6 h-6 text-udemy-purple" />
              Interview Trends
            </h1>
            <p className="text-sm text-udemy-text-muted mt-1">
              Session-by-session trend for overall readiness and rubric dimensions.
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

        {data && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5">
            <Metric label="Sessions" value={`${data.summary.session_count}`} />
            <Metric label="Latest" value={`${Math.round(data.summary.latest_score)}%`} />
            <Metric label="Previous" value={`${Math.round(data.summary.previous_score)}%`} />
            <Metric
              label="Delta"
              value={`${data.summary.delta >= 0 ? "+" : ""}${data.summary.delta.toFixed(1)}`}
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

      <div className="mt-6 space-y-4">
        {loading ? (
          <div className="udemy-card p-8 flex items-center justify-center gap-2 text-sm text-udemy-text-muted">
            <Loader2 className="w-5 h-5 animate-spin text-udemy-purple" />
            Loading trend data...
          </div>
        ) : points.length === 0 ? (
          <div className="udemy-card p-8 text-center">
            <p className="font-semibold">No completed interview sessions yet.</p>
            <p className="text-sm text-udemy-text-muted mt-1">
              Complete at least one session to unlock trend insights.
            </p>
            <Link to="/interview" className="btn-primary mt-4 inline-flex">
              Start mock interview
            </Link>
          </div>
        ) : (
          <>
            <div className="udemy-card p-4">
              <h2 className="text-sm font-bold mb-3">Overall Score Trend</h2>
              <TrendChart
                points={points}
                lines={[
                  {
                    key: "overall",
                    label: "Overall",
                    color: "#2563eb",
                    values: points.map((p) => p.overall_score),
                  },
                ]}
                maxValue={100}
              />
            </div>

            <div className="udemy-card p-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-3">
                <h2 className="text-sm font-bold">Rubric Dimension Trends</h2>
                <div className="flex flex-wrap gap-2">
                  {DIMENSIONS.map((dim) => (
                    <button
                      key={dim.key}
                      onClick={() => toggleDim(dim.key)}
                      className={`px-2 py-1 rounded-full text-xs border ${
                        enabledDims[dim.key]
                          ? "border-transparent text-white"
                          : "border-udemy-border text-udemy-text-muted bg-white"
                      }`}
                      style={enabledDims[dim.key] ? { backgroundColor: dim.color } : {}}
                    >
                      {dim.label}
                    </button>
                  ))}
                </div>
              </div>
              <TrendChart
                points={points}
                lines={DIMENSIONS.filter((dim) => enabledDims[dim.key]).map((dim) => ({
                  key: dim.key,
                  label: dim.label,
                  color: dim.color,
                  values: points.map((p) => (p.rubric_averages[dim.key] || 0) * 20),
                }))}
                maxValue={100}
              />
              <p className="text-xs text-udemy-text-muted mt-2">
                Dimension values are normalized to percentage (0-5 scaled to 0-100).
              </p>
            </div>
          </>
        )}
      </div>
    </motion.div>
  );
}

type ChartLine = {
  key: string;
  label: string;
  color: string;
  values: number[];
};

function TrendChart({
  points,
  lines,
  maxValue,
}: {
  points: InterviewTrendPoint[];
  lines: ChartLine[];
  maxValue: number;
}) {
  const width = 900;
  const height = 260;
  const pad = 24;
  const innerWidth = width - pad * 2;
  const innerHeight = height - pad * 2;
  const safeLines = lines.filter((line) => line.values.length === points.length);

  const xFor = (index: number) => {
    if (points.length <= 1) return pad + innerWidth / 2;
    return pad + (index * innerWidth) / (points.length - 1);
  };
  const yFor = (value: number) => {
    const clamped = Math.max(0, Math.min(maxValue, value));
    return pad + innerHeight - (clamped / maxValue) * innerHeight;
  };
  const buildPath = (values: number[]) =>
    values
      .map((value, idx) => `${idx === 0 ? "M" : "L"} ${xFor(idx)} ${yFor(value)}`)
      .join(" ");

  if (points.length < 2) {
    return (
      <div className="rounded border border-udemy-border p-4 text-sm text-udemy-text-muted">
        Need at least 2 completed sessions to draw a line chart.
      </div>
    );
  }

  if (safeLines.length === 0) {
    return (
      <div className="rounded border border-udemy-border p-4 text-sm text-udemy-text-muted">
        No dimensions selected.
      </div>
    );
  }

  return (
    <div className="rounded border border-udemy-border bg-white p-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-[260px]">
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#d1d5db" strokeWidth="1" />
        <line
          x1={pad}
          y1={height - pad}
          x2={width - pad}
          y2={height - pad}
          stroke="#d1d5db"
          strokeWidth="1"
        />
        {[0, 25, 50, 75, 100].map((tick) => (
          <g key={tick}>
            <line
              x1={pad}
              y1={yFor(tick)}
              x2={width - pad}
              y2={yFor(tick)}
              stroke="#f3f4f6"
              strokeWidth="1"
            />
            <text x={4} y={yFor(tick) + 4} fontSize="10" fill="#6b7280">
              {tick}
            </text>
          </g>
        ))}
        {safeLines.map((line) => (
          <path
            key={line.key}
            d={buildPath(line.values)}
            fill="none"
            stroke={line.color}
            strokeWidth="3"
            strokeLinecap="round"
          />
        ))}
        {safeLines.map((line) =>
          line.values.map((value, idx) => (
            <circle
              key={`${line.key}:${idx}`}
              cx={xFor(idx)}
              cy={yFor(value)}
              r={3}
              fill={line.color}
            />
          )),
        )}
      </svg>
      <div className="flex flex-wrap gap-3 mt-2">
        {safeLines.map((line) => (
          <span key={line.key} className="text-xs inline-flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: line.color }} />
            {line.label}
          </span>
        ))}
      </div>
    </div>
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
