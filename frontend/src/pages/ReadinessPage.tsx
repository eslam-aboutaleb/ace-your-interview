import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  Building2,
  CheckCircle2,
  Compass,
  Gauge,
  Loader2,
  RefreshCw,
  Target,
  XCircle,
} from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import { fetchReadiness } from "@/services/api";
import { useApi } from "@/hooks/useApi";
import { useAuthStore } from "@/store/authStore";
import {
  evidenceKindLabel,
  evidenceValueLabel,
  formatGapPoints,
  formatPoints,
  formatScore,
  formatWeight,
  hasTargetCompanies,
  isAdaptiveLearningDisabledError,
  matchSourceLabel,
  rankedBestFitRoles,
  rankedCompanyFits,
  readinessBandBadgeClass,
  readinessBandBarColor,
  readinessEvidenceRows,
  sortBreakdown,
  sortGapDimensions,
  summarizeGapEvidence,
} from "@/utils/readiness";
import type {
  CompanyReadinessEntry,
  ReadinessGapDimension,
  ReadinessWeightRow,
  RoleReadinessResponse,
} from "@/types";
import SkeletonCards from "@/components/common/SkeletonCards";

export default function ReadinessPage() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const authLoading = useAuthStore((s) => s.isLoading);

  const {
    data,
    loading,
    error,
    retry,
  } = useApi<RoleReadinessResponse>(async () => fetchReadiness(), {
    // Wait for the auth check so an unauthenticated visitor sees the
    // sign-in card instead of a doomed request.
    deps: [isAuthenticated, authLoading],
    immediate: isAuthenticated && !authLoading,
    errorMessage: "Could not load your role readiness.",
  });

  if (!authLoading && !isAuthenticated) {
    return (
      <div className="min-h-screen bg-udemy-bg flex items-center justify-center px-4">
        <div className="udemy-card p-8 max-w-md w-full text-center">
          <h1 className="text-xl font-bold mb-2">Sign in to continue</h1>
          <p className="text-sm text-udemy-text-muted mb-6">
            Readiness is computed from your saved learning evidence, so it
            needs an account.
          </p>
          <Link to="/login" className="btn-primary inline-flex">
            Go to sign in
          </Link>
        </div>
      </div>
    );
  }

  const disabled = isAdaptiveLearningDisabledError(error);

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
              <Target className="w-6 h-6 text-udemy-purple" />
              Role Readiness
            </h1>
            <p className="text-sm text-udemy-text-muted mt-1">
              Readiness for your target role, the ranked roles your evidence
              fits today, and how each target company scores against its bar.
            </p>
          </div>
          <button
            onClick={() => void retry()}
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

        {error && !disabled && (
          <div
            className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2"
            role="alert"
          >
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {loading && !data && (
          <div className="mt-6">
            <SkeletonCards count={3} />
          </div>
        )}

        {disabled && (
          <div className="mt-8 rounded-lg border border-udemy-border p-6 text-center text-sm text-udemy-text-muted">
            <Gauge className="w-6 h-6 mx-auto mb-2 opacity-50" />
            Adaptive learning is turned off for this deployment, so role and
            company readiness are unavailable.
          </div>
        )}

        {data && !disabled && (
          <>
            <TargetRoleSection data={data} />
            <BestFitRolesSection data={data} />
            <CompaniesSection data={data} />
          </>
        )}
      </div>
    </motion.div>
  );
}

function TargetRoleSection({ data }: { data: RoleReadinessResponse }) {
  const role = data.role;
  const trackRows = sortBreakdown(role.track_breakdown);
  const competencyRows = sortBreakdown(role.competency_breakdown);
  const gaps = sortGapDimensions(role.gap_dimensions);

  return (
    <section className="mt-6">
      <div
        className="rounded-lg border border-udemy-border bg-white p-4"
        data-testid="target-role"
      >
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Compass className="w-5 h-5 text-udemy-purple" />
              {role.role_label || "Target role"}
            </h2>
            <p className="text-sm text-udemy-text-muted mt-1">
              {role.target_role
                ? `Your target role: ${role.target_role}`
                : "No target role set yet."}{" "}
              · {matchSourceLabel(role.match_source, "role")}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-3xl font-bold text-udemy-purple">
              {formatScore(role.readiness)}%
            </span>
            <BandBadge label={role.readiness_label} />
          </div>
        </div>

        <div className="mt-4 h-2 rounded-full bg-udemy-border overflow-hidden">
          <div
            className="h-full rounded-full"
            style={{
              width: `${Math.max(0, Math.min(100, role.readiness))}%`,
              backgroundColor: readinessBandBarColor(role.readiness_label),
            }}
          />
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          {readinessEvidenceRows(role.evidence).map((row) => (
            <Metric key={row.label} label={row.label} value={`${row.value}`} />
          ))}
        </div>
      </div>

      <h3 className="text-base font-semibold mt-6">
        What moves the score
      </h3>
      <p className="text-sm text-udemy-text-muted mt-1">
        Each row carries a share of the role&rsquo;s weight model; the weighted
        contributions add up to the readiness score.
      </p>

      <BreakdownTable
        caption="Tracks"
        rows={trackRows}
        testId="track-breakdown"
        emptyMessage="No track evidence yet — study a topic to seed your track scores."
      />
      <BreakdownTable
        caption="Interview competencies"
        rows={competencyRows}
        testId="competency-breakdown"
        emptyMessage="No interview competency evidence yet — run a mock interview to seed these scores."
      />

      <h3 className="text-base font-semibold mt-6">What holds it back</h3>
      {gaps.length === 0 ? (
        <div className="mt-3 rounded-lg border border-udemy-border p-6 text-center text-sm text-udemy-text-muted">
          <CheckCircle2 className="w-5 h-5 mx-auto mb-2 opacity-50" />
          No weighted gaps left in this role model.
        </div>
      ) : (
        <div className="space-y-2 mt-3" data-testid="gap-dimensions">
          {gaps.map((gap) => (
            <GapDimensionCard key={`${gap.kind}:${gap.id}`} gap={gap} />
          ))}
        </div>
      )}
    </section>
  );
}

function BestFitRolesSection({ data }: { data: RoleReadinessResponse }) {
  const roles = rankedBestFitRoles(data.best_fit_roles);

  return (
    <section className="mt-8">
      <h2 className="text-lg font-semibold">Best-fit roles</h2>
      <p className="text-sm text-udemy-text-muted mt-1">
        Every role in the catalog scored against the same evidence, best
        first.
      </p>
      {roles.length === 0 ? (
        <div className="mt-3 rounded-lg border border-udemy-border p-6 text-center text-sm text-udemy-text-muted">
          No roles to rank yet.
        </div>
      ) : (
        <ul className="mt-3 space-y-2" data-testid="best-fit-roles">
          {roles.map((role) => (
            <li
              key={role.role_id}
              className="rounded-lg border border-udemy-border bg-white p-3 flex items-center justify-between gap-3"
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-sm font-semibold truncate">
                  {role.role_label}
                </span>
                {role.is_target && (
                  <span className="inline-flex items-center gap-1 rounded-full border border-udemy-purple bg-udemy-purple/10 px-2 py-0.5 text-xs font-medium text-udemy-purple">
                    <Target className="w-3 h-3" />
                    Target role
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3 flex-shrink-0">
                <span className="text-sm font-bold text-udemy-purple">
                  {formatScore(role.readiness)}%
                </span>
                <BandBadge label={role.readiness_label} />
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function CompaniesSection({ data }: { data: RoleReadinessResponse }) {
  const companies = rankedCompanyFits(data.companies);
  const hasTargets = hasTargetCompanies(data.profile);

  return (
    <section className="mt-8">
      <h2 className="text-lg font-semibold flex items-center gap-2">
        <Building2 className="w-5 h-5 text-udemy-purple" />
        Target companies
      </h2>
      <p className="text-sm text-udemy-text-muted mt-1">
        Each company&rsquo;s hiring bar applied to your current evidence.
      </p>

      {!hasTargets ? (
        <div className="mt-3 rounded-lg border border-udemy-border p-6 text-center text-sm text-udemy-text-muted">
          <Building2 className="w-5 h-5 mx-auto mb-2 opacity-50" />
          Add target companies to your learner profile to see how each
          company&rsquo;s bar scores your evidence.
          <div className="mt-3">
            <Link to="/study-plan" className="btn-secondary text-sm inline-flex">
              Update target companies
            </Link>
          </div>
        </div>
      ) : companies.length === 0 ? (
        <div className="mt-3 rounded-lg border border-udemy-border p-6 text-center text-sm text-udemy-text-muted">
          <Building2 className="w-5 h-5 mx-auto mb-2 opacity-50" />
          No target companies to score yet.
        </div>
      ) : (
        <div className="mt-3 space-y-3" data-testid="company-fits">
          {companies.map((company) => (
            <CompanyFitCard key={company.company_key} company={company} />
          ))}
        </div>
      )}
    </section>
  );
}

function CompanyFitCard({ company }: { company: CompanyReadinessEntry }) {
  const gaps = sortGapDimensions(company.gap_dimensions);

  return (
    <article className="rounded-lg border border-udemy-border bg-white p-4">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold">
            {company.company_label || company.company}
          </h3>
          <p className="text-xs text-udemy-text-muted mt-1">
            {matchSourceLabel(company.match_source, "company")} · bar{" "}
            {formatScore(company.bar)}%
          </p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="text-2xl font-bold text-udemy-purple">
            {formatScore(company.fit)}%
          </span>
          <BandBadge label={company.fit_label} />
          <span
            className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${
              company.clears_bar
                ? "border-emerald-300 bg-emerald-50 text-emerald-900"
                : "border-orange-300 bg-orange-50 text-orange-900"
            }`}
          >
            {company.clears_bar ? (
              <CheckCircle2 className="w-3 h-3" />
            ) : (
              <XCircle className="w-3 h-3" />
            )}
            {company.clears_bar ? "Clears the bar" : "Below the bar"}
          </span>
        </div>
      </div>

      <div className="mt-3 h-2 rounded-full bg-udemy-border overflow-hidden relative">
        <div
          className="h-full rounded-full"
          style={{
            width: `${Math.max(0, Math.min(100, company.fit))}%`,
            backgroundColor: readinessBandBarColor(company.fit_label),
          }}
        />
        <div
          className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-udemy-dark"
          style={{ left: `${Math.max(0, Math.min(100, company.bar))}%` }}
          title={`Hiring bar: ${formatScore(company.bar)}%`}
        />
      </div>

      {gaps.length > 0 && (
        <div className="space-y-2 mt-3">
          {gaps.map((gap) => (
            <GapDimensionCard key={`${gap.kind}:${gap.id}`} gap={gap} />
          ))}
        </div>
      )}
    </article>
  );
}

function BreakdownTable({
  caption,
  rows,
  emptyMessage,
  testId,
}: {
  caption: string;
  rows: ReadinessWeightRow[];
  emptyMessage: string;
  testId: string;
}) {
  return (
    <div className="mt-3" data-testid={testId}>
      <h4 className="text-sm font-semibold text-udemy-text-muted">
        {caption}
      </h4>
      {rows.length === 0 ? (
        <div className="mt-2 rounded-lg border border-udemy-border p-4 text-center text-sm text-udemy-text-muted">
          {emptyMessage}
        </div>
      ) : (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-udemy-text-muted">
                <th className="py-1 pr-3 font-medium">Dimension</th>
                <th className="py-1 pr-3 font-medium">Weight</th>
                <th className="py-1 pr-3 font-medium">Score</th>
                <th className="py-1 pr-3 font-medium">Contribution</th>
                <th className="py-1 font-medium">Headroom</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${row.kind}:${row.id}`} className="border-t border-udemy-border">
                  <td className="py-2 pr-3">
                    <span className="font-medium">{row.label}</span>
                    <span className="text-xs text-udemy-text-muted ml-2">
                      evidence {row.evidence_count}
                    </span>
                  </td>
                  <td className="py-2 pr-3">{formatWeight(row.weight)}</td>
                  <td className="py-2 pr-3">{formatScore(row.score)}%</td>
                  <td className="py-2 pr-3 font-semibold text-udemy-purple">
                    {formatPoints(row.contribution)}
                  </td>
                  <td className="py-2">{formatGapPoints(row.gap_points)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function GapDimensionCard({ gap }: { gap: ReadinessGapDimension }) {
  return (
    <div className="rounded-lg border border-udemy-border bg-white p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold">{gap.label}</p>
          <p className="text-xs text-udemy-text-muted mt-0.5">
            {gap.kind === "track"
              ? "Track"
              : evidenceKindLabel(gap.kind)}{" "}
            · {formatWeight(gap.weight)} of the model · score{" "}
            {formatScore(gap.score)}%
            {summarizeGapEvidence(gap.evidence)
              ? ` · ${summarizeGapEvidence(gap.evidence)}`
              : ""}
          </p>
        </div>
        <span className="text-sm font-bold text-orange-600 flex-shrink-0">
          {formatGapPoints(gap.gap_points)}
        </span>
      </div>

      {gap.evidence.length > 0 && (
        <ul className="mt-2 space-y-1">
          {gap.evidence.map((item, index) => (
            <li
              key={`${item.kind}:${item.ref}:${index}`}
              className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-udemy-text-muted"
            >
              <span className="inline-flex items-center rounded-full border border-udemy-border bg-white px-2 py-0.5 font-medium text-udemy-text">
                {evidenceKindLabel(item.kind)}
              </span>
              <span className="text-udemy-text">{item.label}</span>
              {evidenceValueLabel(item) && (
                <span className="font-semibold">{evidenceValueLabel(item)}</span>
              )}
              {item.detail && <span>{item.detail}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function BandBadge({ label }: { label: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${readinessBandBadgeClass(
        label,
      )}`}
    >
      {label || "Not Started"}
    </span>
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
