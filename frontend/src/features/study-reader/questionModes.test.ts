import { describe, expect, it } from "vitest";
import {
  canReveal,
  MODE_POLICIES,
  OUTCOME_GRADES,
  parseReaderMode,
  READER_MODES,
  REVIEW_GRADES,
} from "./questionModes";

describe("questionModes", () => {
  it("parses the mode query param with a study default", () => {
    expect(parseReaderMode("practice")).toBe("practice");
    expect(parseReaderMode("review")).toBe("review");
    expect(parseReaderMode("interview")).toBe("interview");
    expect(parseReaderMode(null)).toBe("study");
    expect(parseReaderMode("bogus")).toBe("study");
    expect(READER_MODES).toHaveLength(4);
  });

  it("study mode shows content immediately and offers completion", () => {
    const policy = MODE_POLICIES.study;
    expect(policy.showContentImmediately).toBe(true);
    expect(policy.completionToggle).toBe(true);
    expect(canReveal(policy, false, false)).toBe(true);
  });

  it("practice and review require an attempt before reveal", () => {
    for (const mode of ["practice", "review"] as const) {
      const policy = MODE_POLICIES[mode];
      expect(canReveal(policy, false, false)).toBe(false);
      expect(canReveal(policy, true, false)).toBe(true);
    }
  });

  it("interview requires a submitted response before reveal", () => {
    const policy = MODE_POLICIES.interview;
    expect(canReveal(policy, true, false)).toBe(false);
    expect(canReveal(policy, true, true)).toBe(true);
    expect(canReveal(policy, false, false)).toBe(false);
  });

  it("maps FSRS-style review grades onto the attempts contract", () => {
    expect(REVIEW_GRADES.map((g) => g.label)).toEqual(["Again", "Hard", "Good", "Easy"]);
    const again = REVIEW_GRADES[0];
    expect(again.outcome).toBe("incorrect");
    const easy = REVIEW_GRADES[3];
    expect(easy.outcome).toBe("correct");
    expect(easy.confidence).toBe(5);
    for (const grade of REVIEW_GRADES) {
      expect(grade.confidence).toBeGreaterThanOrEqual(0);
      expect(grade.confidence).toBeLessThanOrEqual(5);
    }
  });

  it("outcome grades fit the attempts contract", () => {
    expect(OUTCOME_GRADES.map((g) => g.outcome)).toEqual(["correct", "incorrect", "skipped"]);
  });
});
