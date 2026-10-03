import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MarkdownRenderer from "./MarkdownRenderer";

vi.mock("@/components/common/CodePanel", () => ({
  default: ({
    code,
    language,
  }: {
    code: string;
    language: string;
  }) => (
    <div data-testid="code-panel" data-language={language}>
      {code}
    </div>
  ),
}));

vi.mock("@/components/common/MermaidDiagram", () => ({
  default: ({ code }: { code: string }) => (
    <div data-testid="mermaid-diagram">{code}</div>
  ),
}));

describe("MarkdownRenderer", () => {
  it("renders fenced code blocks through CodePanel with the language tag", () => {
    render(<MarkdownRenderer content={"```python\nprint('hi')\n```"} />);

    const panel = screen.getByTestId("code-panel");
    expect(panel).toHaveAttribute("data-language", "python");
    expect(panel).toHaveTextContent("print('hi')");
  });

  it("renders mermaid fences through MermaidDiagram", () => {
    render(
      <MarkdownRenderer content={"```mermaid\ngraph TD\n  A --> B\n```"} />,
    );

    expect(screen.getByTestId("mermaid-diagram")).toHaveTextContent(
      "graph TD",
    );
  });

  it("renders headings and coach-labelled paragraphs", () => {
    render(
      <MarkdownRenderer
        content={"## Complexity\n\n**Answer:** O(1) lookups."}
      />,
    );

    const heading = screen.getByRole("heading", { level: 2 });
    expect(heading).toHaveClass("markdown-heading-2");
    expect(heading).toHaveTextContent("Complexity");

    const paragraph = screen.getByText("O(1) lookups.");
    expect(paragraph).toHaveAttribute("data-coach-label", "Answer:");
  });

  it("normalises star-styled labels and titles before rendering", () => {
    render(
      <MarkdownRenderer content={"*Bold label*: value\n\n*Section Title*"} />,
    );

    expect(screen.getByText("Bold label:")).toBeInTheDocument();
    const heading = screen.getByRole("heading", { level: 3 });
    expect(heading).toHaveTextContent("Section Title");
  });
});
