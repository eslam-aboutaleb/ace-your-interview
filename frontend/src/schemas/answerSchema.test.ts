import { describe, expect, it } from "vitest";

import {
  ANSWER_SCHEMA_VERSION,
  hasHeadingLevelSkip,
  hasUnsafeHtml,
  richAnswer,
  validateRichAnswer,
} from "./answerSchema";

const fullPayload = {
  answer_schema_version: ANSWER_SCHEMA_VERSION,
  sections: [
    { type: "summary", id: "summary-0", text: "HTTP is a request/response protocol." },
    {
      type: "explanation",
      id: "lifecycle-1",
      title: "Request lifecycle",
      text: "DNS, TCP, TLS, then the app server.",
    },
    {
      type: "code",
      id: "example-2",
      language: "Python",
      code: "print('hello')\n",
      filename: "main.py",
      line_highlights: [1],
    },
    {
      type: "table",
      id: "verbs-3",
      columns: ["Verb", "Idempotent"],
      rows: [
        ["GET", "yes"],
        ["POST", "no"],
      ],
    },
    {
      type: "diagram",
      id: "flow-4",
      format: "mermaid",
      source: "sequenceDiagram\nClient->>Server: GET /",
      alt: "Request flow",
    },
    {
      type: "tradeoffs",
      id: "tradeoffs-5",
      items: [{ option: "HTTP/1.1", pros: ["simple"], cons: ["HOL blocking"] }],
    },
    {
      type: "common_mistakes",
      id: "mistakes-6",
      items: [{ mistake: "Using POST for reads", correction: "Use GET" }],
    },
    {
      type: "references",
      id: "refs-7",
      items: [{ label: "RFC 9110", source_ref: "https://www.rfc-editor.org/rfc/rfc9110" }],
    },
    { type: "follow_up", id: "follow-8", prompts: ["How does HTTP/2 help?"] },
  ],
};

describe("richAnswer schema", () => {
  it("accepts a full payload and round-trips", () => {
    const parsed = richAnswer.parse(fullPayload);
    expect(parsed.sections).toHaveLength(9);
    const again = richAnswer.parse(JSON.parse(JSON.stringify(parsed)));
    expect(again).toEqual(parsed);
  });

  it("normalises the code language to lowercase", () => {
    const parsed = richAnswer.parse(fullPayload);
    const code = parsed.sections.find((s) => s.type === "code");
    expect(code && code.type === "code" && code.language).toBe("python");
  });

  it("rejects an unknown section type", () => {
    const result = validateRichAnswer({
      sections: [{ type: "interactive_widget", text: "hi" }],
    });
    expect(result.ok).toBe(false);
  });

  it("rejects duplicate section ids", () => {
    const result = validateRichAnswer({
      sections: [
        { type: "summary", id: "a", text: "one" },
        { type: "summary", id: "a", text: "two" },
      ],
    });
    expect(result.ok).toBe(false);
  });

  it("rejects empty sections", () => {
    expect(validateRichAnswer({ sections: [] }).ok).toBe(false);
  });

  it("rejects a table row width mismatch", () => {
    const result = validateRichAnswer({
      sections: [{ type: "table", id: "t-0", columns: ["a", "b"], rows: [["only"]] }],
    });
    expect(result.ok).toBe(false);
  });
});

describe("safety rules mirror the backend", () => {
  it("flags script tags", () => {
    expect(hasUnsafeHtml("hi <script>alert(1)</script>")).toBe(true);
    expect(hasUnsafeHtml('<img src=x onerror="alert(1)">')).toBe(true);
    expect(hasUnsafeHtml("[click](javascript:alert(1))")).toBe(true);
    expect(hasUnsafeHtml("plain <b>bold</b> is not flagged as executable")).toBe(false);
  });

  it("rejects xss in sections", () => {
    const result = validateRichAnswer({
      sections: [{ type: "explanation", id: "x-0", text: "ok <script>alert(1)</script>" }],
    });
    expect(result.ok).toBe(false);
  });

  it("rejects mermaid click handlers", () => {
    const result = validateRichAnswer({
      sections: [
        {
          type: "diagram",
          id: "d-0",
          format: "mermaid",
          source: 'graph TD\nA-->B\nclick A href "javascript:alert(1)"',
          alt: "diagram",
        },
      ],
    });
    expect(result.ok).toBe(false);
  });

  it("flags skipped heading levels", () => {
    expect(hasHeadingLevelSkip("## One\n\n#### Skipped")).toBe(true);
    expect(hasHeadingLevelSkip("## One\n\n### Fine")).toBe(false);
  });

  it("rejects heading skips in markdown sections", () => {
    const result = validateRichAnswer({
      sections: [{ type: "explanation", id: "x-0", text: "## One\n\n#### Skipped" }],
    });
    expect(result.ok).toBe(false);
  });
});
