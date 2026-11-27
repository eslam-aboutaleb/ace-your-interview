import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CodePanel from "./CodePanel";

const writeText = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  writeText.mockResolvedValue(undefined);
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText },
    configurable: true,
  });
});

describe("CodePanel", () => {
  it("renders code with its language label", () => {
    const { container } = render(<CodePanel language="python" code={"def f():\n    return 1"} />);
    expect(screen.getByText("python")).toBeInTheDocument();
    expect(container.querySelector("code")?.textContent).toContain("def f");
  });

  it("shows the optional filename", () => {
    render(<CodePanel language="sql" filename="schema.sql" code="SELECT 1" />);
    expect(screen.getByText(/schema\.sql/)).toBeInTheDocument();
  });

  it("announces copy success through the live region", async () => {
    render(<CodePanel code="SELECT 1" language="sql" />);
    fireEvent.click(screen.getByRole("button", { name: "Copy code" }));
    await waitFor(() => {
      expect(screen.getByRole("status", { name: "Copy status" })).toHaveTextContent(
        "Copied to clipboard",
      );
    });
    expect(writeText).toHaveBeenCalledWith("SELECT 1");
  });

  it("reports clipboard failure instead of pretending success", async () => {
    writeText.mockRejectedValue(new Error("denied"));
    render(<CodePanel code="SELECT 1" language="sql" />);
    fireEvent.click(screen.getByRole("button", { name: "Copy code" }));
    await waitFor(() => {
      expect(screen.getByRole("status", { name: "Copy status" })).toHaveTextContent(
        "Copy failed",
      );
    });
  });

  it("defaults to horizontal overflow and toggles soft wrap", () => {
    const longLine = `const x = "${"very-long-".repeat(40)}";`;
    const { container } = render(<CodePanel language="typescript" code={longLine} />);
    const codeEl = container.querySelector("code");
    expect(codeEl?.style.whiteSpace).toBe("pre");

    const wrapButton = screen.getByRole("button", { name: "Toggle line wrap" });
    expect(wrapButton).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(wrapButton);
    expect(wrapButton).toHaveAttribute("aria-pressed", "true");
    expect(container.querySelector("code")?.style.whiteSpace).toBe("pre-wrap");
  });

  it("shows line numbers by default and toggles them off", () => {
    const { container } = render(<CodePanel language="python" code={"a = 1\nb = 2"} />);
    expect(container.querySelectorAll(".linenumber").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Toggle line numbers" }));
    expect(container.querySelectorAll(".linenumber")).toHaveLength(0);
  });

  it("defaults to no line numbers in compact mode", () => {
    const { container } = render(<CodePanel compact language="python" code={"a = 1\nb = 2"} />);
    expect(container.querySelectorAll(".linenumber")).toHaveLength(0);
  });

  it("highlights the requested lines", () => {
    const { container } = render(
      <CodePanel language="python" code={"a = 1\nb = 2\nc = 3"} lineHighlights={[2]} />,
    );
    const highlighted = Array.from(container.querySelectorAll("span")).filter((el) =>
      el.style.backgroundColor.includes("250, 204, 21"),
    );
    expect(highlighted.length).toBeGreaterThan(0);
    expect(highlighted[0].textContent).toContain("b = 2");
  });

  it("renders an unknown language as plain text without crashing", () => {
    const { container } = render(
      <CodePanel language="madeuplang" code={"some exotic code here"} />,
    );
    expect(container.querySelector("code")?.textContent).toContain("some exotic code here");
    expect(screen.getByText("madeuplang")).toBeInTheDocument();
  });

  it("decodes structurally-escaped newlines before rendering and copying", async () => {
    // Two or more escaped newlines with no real ones trip the decoder.
    render(<CodePanel language="python" code={"a = 1\\nb = 2\\nc = 3"} />);
    fireEvent.click(screen.getByRole("button", { name: "Copy code" }));
    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith("a = 1\nb = 2\nc = 3");
    });
  });

  it("leaves a lone escaped newline untouched (real code may contain one)", async () => {
    render(<CodePanel language="python" code={'print("a\\nb")'} />);
    fireEvent.click(screen.getByRole("button", { name: "Copy code" }));
    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith('print("a\\nb")');
    });
  });
});
