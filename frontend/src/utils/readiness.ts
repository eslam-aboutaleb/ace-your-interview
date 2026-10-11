/**
 * Readiness data-shaping helpers.
 *
 * `GET /api/learning/readiness` returns weighted rows, gap
 * dimensions, ranked lists and banded labels that mirror the
 * interview report's bands (Strong / On Track / Developing /
 * Needs Work / Not Started). These helpers turn those payloads
 * into the labels, colours and orderings the readiness page
 * renders, so the page stays presentational and the rules stay
 * unit-testable.
 */

import type {
  BestFitRole,
  CompanyReadinessEntry,
  ReadinessGapDimension,
  ReadinessGapEvidence,
  ReadinessWeightRow,
  RoleReadinessEvidence,
} from "@/types";

/** Interview-report band labels, strongest first. */
export const READINESS_BANDS = [
  "Strong",
  "On Track",
  "Developing",
  "Needs Work",
  "Not Started",
] as const;

export type ReadinessBand = (typeof READINESS_BANDS)[number];

/** Badge classes per band, matching the interview report's vocabulary. */
const BAND_BADGE_CLASS: Record<ReadinessBand, string> = {
  Strong: "border-emerald-300 bg-emerald-50 text-emerald-900",
  "On Track": "border-sky-300 bg-sky-50 text-sky-900",
  Developing: "border-amber-300 bg-amber-50 text-amber-900",
  "Needs Work": "border-orange-300 bg-orange-50 text-orange-900",
  "Not Started": "border-udemy-border bg-white text-udemy-text-muted",
};

/** Bar fill colours per band. */
const BAND_BAR_COLOR: Record<ReadinessBand, string> = {
  Strong: "#059669",
  "On Track": "#0284c7",
  Developing: "#d97706",
  "Needs Work": "#ea580c",
  "Not Started": "#9ca3af",
};

const EVIDENCE_KIND_LABELS: Record<string, string> = {
  topic: "Topic",
  interview_competency: "Interview competency",
  error_pattern: "Error pattern",
};

const ROLE_MATCH_SOURCE_LABELS: Record<string, string> = {
  catalog: "Built-in role model",
  primary_track_default: "Primary-track default",
};

const COMPANY_MATCH_SOURCE_LABELS: Record<string, string> = {
  catalog: "Built-in company profile",
  default: "Role-inherited default",
};

export function isReadinessBand(label: string): label is ReadinessBand {
  return (READINESS_BANDS as readonly string[]).includes(label);
}

/** Badge classes for a band label, with a calm fallback. */
export function readinessBandBadgeClass(label: string): string {
  return isReadinessBand(label)
    ? BAND_BADGE_CLASS[label]
    : BAND_BADGE_CLASS["Not Started"];
}

/** Bar fill colour for a band label. */
export function readinessBandBarColor(label: string): string {
  return isReadinessBand(label)
    ? BAND_BAR_COLOR[label]
    : BAND_BAR_COLOR["Not Started"];
}

/** Round a 0-100 score for display. */
export function formatScore(value: number): string {
  return `${Math.round(Number.isFinite(value) ? value : 0)}`;
}

/** Render a 0-1 weight as the share of the model it carries. */
export function formatWeight(value: number): string {
  return `${Math.round((Number.isFinite(value) ? value : 0) * 100)}%`;
}

/** Render the readiness points a dimension already contributes. */
export function formatPoints(value: number): string {
  const points = Number.isFinite(value) ? value : 0;
  return `${points.toFixed(1)} pts`;
}

/** Render the readiness points a dimension can still return. */
export function formatGapPoints(value: number): string {
  const points = Number.isFinite(value) ? value : 0;
  return `${points > 0 ? "+" : ""}${points.toFixed(1)} pts`;
}

/** Human label for a gap-evidence kind. */
export function evidenceKindLabel(kind: string): string {
  return EVIDENCE_KIND_LABELS[kind] ?? kind.replace(/_/g, " ");
}

/**
 * Value to show for one piece of gap evidence: scored evidence
 * shows its percentage, count-only evidence shows how often it
 * recurred.
 */
export function evidenceValueLabel(evidence: ReadinessGapEvidence): string {
  if (typeof evidence.score === "number") {
    return `${formatScore(evidence.score)}%`;
  }
  if (evidence.count > 0) {
    return `${evidence.count}×`;
  }
  return "";
}

/** One-line "2 topics · 1 error pattern" summary of a gap's evidence. */
export function summarizeGapEvidence(
  evidence: ReadinessGapEvidence[],
): string {
  const counts = new Map<string, number>();
  for (const item of evidence) {
    counts.set(item.kind, (counts.get(item.kind) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([kind, count]) => {
      const label = evidenceKindLabel(kind).toLowerCase();
      return `${count} ${label}${count === 1 ? "" : "s"}`;
    })
    .join(" · ");
}

/** Where a role or company weight model came from. */
export function matchSourceLabel(
  source: string,
  entity: "role" | "company",
): string {
  const labels =
    entity === "role" ? ROLE_MATCH_SOURCE_LABELS : COMPANY_MATCH_SOURCE_LABELS;
  return labels[source] ?? "Estimated from your profile";
}

/** Break down a readiness score, biggest contribution first. */
export function sortBreakdown(
  rows: ReadinessWeightRow[],
): ReadinessWeightRow[] {
  return [...rows].sort(
    (a, b) =>
      b.contribution - a.contribution ||
      b.gap_points - a.gap_points ||
      a.label.localeCompare(b.label),
  );
}

/** Ranked gaps, most headroom first. */
export function sortGapDimensions(
  dimensions: ReadinessGapDimension[],
): ReadinessGapDimension[] {
  return [...dimensions].sort(
    (a, b) =>
      b.gap_points - a.gap_points ||
      b.weight - a.weight ||
      a.label.localeCompare(b.label),
  );
}

/**
 * The `limit` gaps that still hold the most readiness points back,
 * for surfaces that show a short list instead of the full ranked set.
 */
export function topGapDimensions(
  dimensions: ReadinessGapDimension[],
  limit = 3,
): ReadinessGapDimension[] {
  if (limit <= 0) return [];
  return sortGapDimensions(dimensions).slice(0, limit);
}

/** Best-fit roles ranked by readiness, then by name. */
export function rankedBestFitRoles(
  roles: BestFitRole[],
): BestFitRole[] {
  return [...roles].sort(
    (a, b) =>
      b.readiness - a.readiness || a.role_label.localeCompare(b.role_label),
  );
}

/** Target companies ranked by fit, then by name. */
export function rankedCompanyFits(
  companies: CompanyReadinessEntry[],
): CompanyReadinessEntry[] {
  return [...companies].sort(
    (a, b) => b.fit - a.fit || a.company.localeCompare(b.company),
  );
}

/** The evidence counters that explain the readiness blend. */
export function readinessEvidenceRows(
  evidence: RoleReadinessEvidence,
): Array<{ label: string; value: number }> {
  return [
    { label: "Topic reviews", value: evidence.topic_reviews },
    { label: "Interview reports", value: evidence.interview_reports },
    {
      label: "Competency reviews",
      value: evidence.interview_competency_reviews,
    },
    { label: "Error patterns", value: evidence.error_patterns },
  ];
}

/** Whether the learner has at least one non-blank target company. */
export function hasTargetCompanies(
  profile: { target_companies?: string[] | null } | null | undefined,
): boolean {
  return (profile?.target_companies ?? []).some(
    (company) => typeof company === "string" && company.trim().length > 0,
  );
}

/**
 * The backend gates the endpoint behind the adaptive-learning flag
 * with a 404, so a disabled deployment surfaces as an error whose
 * detail reads "Adaptive learning disabled".
 */
export function isAdaptiveLearningDisabledError(message: string): boolean {
  return /adaptive learning/i.test(message);
}
