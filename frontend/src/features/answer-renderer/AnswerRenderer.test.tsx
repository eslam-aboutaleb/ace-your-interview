import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AnswerRenderer from "./AnswerRenderer";

const mermaidRender = vi.fn();
vi.mock("mermaid", () => ({
  default: {
    initialize: vi.fn(),
    render: (...args: unknown[]) => mermaidRender(...args),
  },
}));

beforeEach(() => {
  vi.clearAllMocks();
  mermaidRender.mockResolvedValue({ svg: "<svg><text>diagram</text></svg>" });
});

describe("AnswerRenderer", () => {
  it("renders markdown sections with nested lists and heading anchors", () => {
    render(
      <AnswerRenderer
        payload={{
          answer_schema_version: "1.0",
          sections: [
            {
              type: "explanation",
              id: "why",
              title: "Why it works",
              text: "Top level\n\n- one\n  - nested a\n  - nested b\n- two",
            },
          ],
        }}
      />,
    );
    expect(screen.getByRole("heading", { level: 2, name: "Why it works" })).toBeInTheDocument();
    expect(document.getElementById("why")).toBeInTheDocument();
    expect(screen.getByText("nested a")).toBeInTheDocument();
  });

  it("renders a GFM table section as a semantic table", () => {
    render(
      <AnswerRenderer
        payload={{
          sections: [
            {
              type: "table",
              title: "Comparison",
              columns: ["Approach", "Cost"],
              rows: [
                ["A", "O(n)"],
                ["B", "O(n log n)"],
              ],
            },
          ],
        }}
      />,
    );
    const table = screen.getByRole("table");
    expect(table).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Approach" })).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(3);
    expect(screen.getByRole("cell", { name: "O(n log n)" })).toBeInTheDocument();
  });

  it("renders code sections through the code panel", () => {
    const { container } = render(
      <AnswerRenderer
        payload={{
          sections: [
            { type: "code", title: "Solution", language: "python", code: "def f():\n    return 1" },
          ],
        }}
      />,
    );
    // CodePanel syntax-highlights into spans; assert on the joined text.
    const codeEl = container.querySelector("pre, code");
    expect(codeEl?.textContent).toContain("def f");
    expect(codeEl?.textContent).toContain("return 1");
  });

  it("renders tradeoffs, mistakes, references and follow-ups", () => {
    render(
      <AnswerRenderer
        payload={{
          sections: [
            {
              type: "tradeoffs",
              title: "Options",
              items: [{ option: "Cache", pros: ["Fast"], cons: ["Stale risk"] }],
            },
            {
              type: "common_mistakes",
              items: [{ mistake: "Off by one", correction: "Use inclusive bounds" }],
            },
            {
              type: "references",
              items: [
                {
                  label: "Docs",
                  source_ref: "https://example.com/docs",
                  quote: "quoted line",
                },
                { label: "Internal", source_ref: "topic://backend/section-3" },
              ],
            },
            { type: "follow_up", prompts: ["How does this scale?"] },
          ],
        }}
      />,
    );
    expect(screen.getByText("Cache")).toBeInTheDocument();
    expect(screen.getByText("Fast")).toBeInTheDocument();
    expect(screen.getByText("Stale risk")).toBeInTheDocument();
    expect(screen.getByText("Off by one")).toBeInTheDocument();
    expect(screen.getByText("Use inclusive bounds")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "https://example.com/docs" });
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    expect(link).toHaveAttribute("target", "_blank");
    expect(screen.getByText("topic://backend/section-3")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "topic://backend/section-3" })).toBeNull();
    expect(screen.getByText("quoted line")).toBeInTheDocument();
    expect(screen.getByText("How does this scale?")).toBeInTheDocument();
  });

  it("rejects malicious HTML in a section and still renders the rest", () => {
    render(
      <AnswerRenderer
        payload={{
          sections: [
            {
              type: "summary",
              title: "Evil",
              text: 'hello <script>alert("x")</script><img src=x onerror=alert(1)>',
            },
            { type: "summary", title: "Good", text: "Safe content here." },
          ],
        }}
      />,
    );
    expect(screen.getByText(/can't be displayed yet/)).toBeInTheDocument();
    expect(screen.getByText("Safe content here.")).toBeInTheDocument();
    expect(document.querySelector("script")).toBeNull();
    expect(document.querySelector("img")).toBeNull();
  });

  it("treats raw HTML in markdown as inert text", () => {
    render(
      <AnswerRenderer
        payload={{
          sections: [
            {
              type: "summary",
              title: "Htmlish",
              text: "Use <div> tags\n\n<b>not bold html</b>",
            },
          ],
        }}
      />,
    );
    // react-markdown has no raw-HTML plugin: markup renders as text.
    expect(screen.getByText(/<div>/)).toBeInTheDocument();
    expect(document.querySelector("b")).toBeNull();
  });

  it("renders unknown section types as an accessible fallback", () => {
    render(
      <AnswerRenderer
        payload={{
          sections: [
            { type: "flashcard", title: "Future section", front: "q", back: "a" },
            { type: "summary", text: "Still works." },
          ],
        }}
      />,
    );
    expect(screen.getByText(/can't be displayed yet: Future section/)).toBeInTheDocument();
    expect(screen.getByText("Still works.")).toBeInTheDocument();
  });

  it("renders diagrams with an accessible label and a failure fallback", async () => {
    render(
      <AnswerRenderer
        payload={{
          sections: [
            {
              type: "diagram",
              format: "mermaid",
              source: "graph TD; A-->B",
              alt: "Flow from A to B",
            },
          ],
        }}
      />,
    );
    expect(screen.getByRole("img", { name: "Flow from A to B" })).toBeInTheDocument();
    await waitFor(() => expect(mermaidRender).toHaveBeenCalled());
  });

  it("shows the source fallback when diagram rendering fails", async () => {
    mermaidRender.mockRejectedValue(new Error("bad diagram"));
    render(
      <AnswerRenderer
        payload={{
          sections: [
            {
              type: "diagram",
              format: "mermaid",
              source: "graph TD; A-->B",
              alt: "Flow from A to B",
            },
          ],
        }}
      />,
    );
    expect(await screen.findByText(/Diagram rendering failed/)).toBeInTheDocument();
  });

  it("handles broken markdown and very long content without crashing", () => {
    const longText = `${"paragraph with **unclosed bold and | broken | table\n\n".repeat(200)}`;
    render(
      <AnswerRenderer
        payload={{ sections: [{ type: "explanation", title: "Long", text: longText }] }}
      />,
    );
    expect(screen.getByRole("heading", { level: 2, name: "Long" })).toBeInTheDocument();
  });

  it("shows an alert for payloads without sections", () => {
    render(<AnswerRenderer payload={{ nope: true }} />);
    expect(screen.getByRole("alert")).toHaveTextContent("could not be displayed");
  });

  it("generates stable anchors when section ids are missing", () => {
    render(
      <AnswerRenderer
        payload={{
          sections: [
            { type: "summary", title: "One", text: "first" },
            { type: "summary", title: "Two", text: "second" },
          ],
        }}
      />,
    );
    expect(document.getElementById("answer-section-0")).toBeInTheDocument();
    expect(document.getElementById("answer-section-1")).toBeInTheDocument();
  });
});
