import { describe, expect, it } from "vitest";
import {
  evidenceKindLabel,
  evidenceValueLabel,
  formatGapPoints,
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
  topGapDimensions,
} from "./readiness";
import type {
  BestFitRole,
  CompanyReadinessEntry,
  ReadinessGapDimension,
  ReadinessGapEvidence,
  ReadinessWeightRow,
  RoleReadinessEvidence,
} from "@/types";

function weightRow(
  id: string,
  overrides: Partial<ReadinessWeightRow> = {},
): ReadinessWeightRow {
  return {
    kind: "track",
    id,
    label: id,
    weight: 0.5,
    score: 40,
    evidence_count: 3,
    contribution: 20,
    gap_points: 30,
    ...overrides,
  };
}

function gap(
  id: string,
  gapPoints: number,
  overrides: Partial<ReadinessGapDimension> = {},
): ReadinessGapDimension {
  return {
    kind: "track",
    id,
    label: id,
    weight: 0.5,
    score: 30,
    evidence_count: 2,
    gap_points: gapPoints,
    evidence: [],
    ...overrides,
  };
}

describe("readiness formatting helpers", () => {
  it("rounds scores and weights for display", () => {
    expect(formatScore(72.4)).toBe("72");
    expect(formatScore(Number.NaN)).toBe("0");
    expect(formatWeight(0.345)).toBe("35%");
    expect(formatGapPoints(22.8)).toBe("+22.8 pts");
    expect(formatGapPoints(0)).toBe("0.0 pts");
  });

  it("labels evidence kinds and values", () => {
    expect(evidenceKindLabel("topic")).toBe("Topic");
    expect(evidenceKindLabel("interview_competency")).toBe(
      "Interview competency",
    );
    expect(evidenceKindLabel("something_new")).toBe("something new");

    const scored: ReadinessGapEvidence = {
      kind: "topic",
      ref: "t",
      label: "Topic",
      score: 41,
      count: 0,
      detail: "",
    };
    const counted: ReadinessGapEvidence = {
      kind: "error_pattern",
      ref: "off_by_one",
      label: "Off By One",
      count: 3,
      detail: "",
    };
    const bare: ReadinessGapEvidence = {
      kind: "topic",
      ref: "t",
      label: "Topic",
      count: 0,
      detail: "",
    };
    expect(evidenceValueLabel(scored)).toBe("41%");
    expect(evidenceValueLabel(counted)).toBe("3×");
    expect(evidenceValueLabel(bare)).toBe("");
  });

  it("summarises gap evidence by kind", () => {
    expect(
      summarizeGapEvidence([
        { kind: "topic", ref: "a", label: "A", count: 0, detail: "" },
        { kind: "topic", ref: "b", label: "B", count: 0, detail: "" },
        {
          kind: "error_pattern",
          ref: "c",
          label: "C",
          count: 1,
          detail: "",
        },
      ]),
    ).toBe("2 topics · 1 error pattern");
  });

  it("labels where a role or company weight model came from", () => {
    expect(matchSourceLabel("catalog", "role")).toBe("Built-in role model");
    expect(matchSourceLabel("primary_track_default", "role")).toBe(
      "Primary-track default",
    );
    expect(matchSourceLabel("default", "company")).toBe(
      "Role-inherited default",
    );
    expect(matchSourceLabel("mystery", "company")).toBe(
      "Estimated from your profile",
    );
  });

  it("marks every band with its badge class and bar colour", () => {
    expect(readinessBandBadgeClass("Strong")).toContain("emerald");
    expect(readinessBandBarColor("On Track")).toBe("#0284c7");
    expect(readinessBandBadgeClass("Unknown Band")).toContain("udemy-border");
  });

  it("recognises the adaptive-learning flag failure", () => {
    expect(isAdaptiveLearningDisabledError("Adaptive learning disabled")).toBe(
      true,
    );
    expect(isAdaptiveLearningDisabledError("Learning store down")).toBe(false);
  });
});

describe("readiness ordering helpers", () => {
  it("orders breakdown rows by contribution, then headroom", () => {
    const rows = [
      weightRow("a", { contribution: 10, gap_points: 5 }),
      weightRow("b", { contribution: 30, gap_points: 1 }),
      weightRow("c", { contribution: 10, gap_points: 9 }),
    ];
    expect(sortBreakdown(rows).map((row) => row.id)).toEqual(["b", "c", "a"]);
    expect(rows.map((row) => row.id)).toEqual(["a", "b", "c"]);
  });

  it("orders gap dimensions by the points they can still return", () => {
    const gaps = [
      gap("a", 4.8),
      gap("b", 22.8),
      gap("c", 9.6),
    ];
    expect(sortGapDimensions(gaps).map((item) => item.id)).toEqual([
      "b",
      "c",
      "a",
    ]);
  });

  it("keeps only the requested number of top gaps", () => {
    const gaps = [gap("a", 4.8), gap("b", 22.8), gap("c", 9.6), gap("d", 1)];
    expect(topGapDimensions(gaps, 3).map((item) => item.id)).toEqual([
      "b",
      "c",
      "a",
    ]);
    expect(topGapDimensions(gaps)).toHaveLength(3);
    expect(topGapDimensions(gaps, 0)).toEqual([]);
  });

  it("ranks best-fit roles and target companies by score", () => {
    const roles: BestFitRole[] = [
      { role_id: "a", role_label: "A", readiness: 40, readiness_label: "", is_target: false },
      { role_id: "b", role_label: "B", readiness: 80, readiness_label: "", is_target: true },
    ];
    expect(rankedBestFitRoles(roles).map((role) => role.role_id)).toEqual([
      "b",
      "a",
    ]);

    const companies: CompanyReadinessEntry[] = [
      {
        company: "Stripe",
        company_key: "stripe",
        company_label: "Stripe",
        match_source: "catalog",
        fit: 61.3,
        fit_label: "",
        bar: 70,
        clears_bar: false,
        track_weights: {},
        competency_weights: {},
        gap_dimensions: [],
      },
      {
        company: "Datadog",
        company_key: "datadog",
        company_label: "Datadog",
        match_source: "default",
        fit: 83.1,
        fit_label: "",
        bar: 74,
        clears_bar: true,
        track_weights: {},
        competency_weights: {},
        gap_dimensions: [],
      },
    ];
    expect(rankedCompanyFits(companies).map((item) => item.company_key)).toEqual([
      "datadog",
      "stripe",
    ]);
  });
});

describe("readiness evidence rows", () => {
  it("reports the counters the backend actually sends", () => {
    const evidence: RoleReadinessEvidence = {
      topic_reviews: 12,
      interview_reports: 2,
      interview_competency_reviews: 18,
      error_patterns: 5,
    };
    expect(readinessEvidenceRows(evidence)).toEqual([
      { label: "Topic reviews", value: 12 },
      { label: "Interview reports", value: 2 },
      { label: "Competency reviews", value: 18 },
      { label: "Error patterns", value: 5 },
    ]);
  });

  it("detects at least one non-blank target company", () => {
    expect(hasTargetCompanies({ target_companies: [" ", ""] })).toBe(false);
    expect(hasTargetCompanies({ target_companies: ["Stripe"] })).toBe(true);
    expect(hasTargetCompanies(null)).toBe(false);
    expect(hasTargetCompanies(undefined)).toBe(false);
  });
});
