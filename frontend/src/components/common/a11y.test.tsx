import { render } from "@testing-library/react";
import axe from "axe-core";
import { describe, expect, it } from "vitest";
import MarkdownRenderer from "./MarkdownRenderer";

// Accessibility release gate for the shared reader components (issue #17):
// rendered output must have zero critical or serious axe violations. This
// runs in the deterministic CI tier; broader Playwright browser journeys
// are tracked as follow-up work.

type AxeResults = {
  violations: Array<{ id: string; impact: string | null; nodes: unknown[] }>;
};

async function runAxe(container: HTMLElement): Promise<AxeResults> {
  const results = await axe.run(container, {
    resultTypes: ["violations"],
  });
  return results as AxeResults;
}

function blockingViolations(results: AxeResults) {
  return results.violations.filter(
    (v) => v.impact === "critical" || v.impact === "serious",
  );
}

describe("accessibility gate", () => {
  it("MarkdownRenderer prose has no critical or serious violations", async () => {
    const { container } = render(
      <MarkdownRenderer
        content={
          "# Heading\n\nSome **bold** text with a [link](https://example.com).\n\n- one\n- two\n"
        }
      />,
    );
    const violations = blockingViolations(await runAxe(container));
    expect(
      violations.map((v) => `${v.id} (${v.impact})`),
    ).toEqual([]);
  });

  it("MarkdownRenderer tables and code keep zero blocking violations", async () => {
    const { container } = render(
      <MarkdownRenderer
        content={
          "| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nprint('hi')\n```\n"
        }
      />,
    );
    const violations = blockingViolations(await runAxe(container));
    expect(
      violations.map((v) => `${v.id} (${v.impact})`),
    ).toEqual([]);
  });
});
