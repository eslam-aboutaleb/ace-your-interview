import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import ReaderShell from "./ReaderShell";

function Harness({ toc }: { toc?: React.ReactNode }) {
  const [railOpen, setRailOpen] = useState(true);
  const [mobileRailOpen, setMobileRailOpen] = useState(false);
  return (
    <ReaderShell
      rail={<div>rail content</div>}
      toc={toc}
      breadcrumbs={<div>crumbs</div>}
      toolbar={<div>toolbar</div>}
      railOpen={railOpen}
      onToggleRail={() => setRailOpen((open) => !open)}
      mobileRailOpen={mobileRailOpen}
      onToggleMobileRail={() => setMobileRailOpen((open) => !open)}
    >
      <div>reader content</div>
    </ReaderShell>
  );
}

describe("ReaderShell", () => {
  it("renders rail, reader and toolbar regions", () => {
    render(<Harness />);
    expect(screen.getByRole("complementary", { name: "Curriculum" })).toBeInTheDocument();
    expect(screen.getByRole("main", { name: "Reader" })).toBeInTheDocument();
    expect(screen.getByText("rail content")).toBeInTheDocument();
    expect(screen.getByText("reader content")).toBeInTheDocument();
    expect(screen.getByText("toolbar")).toBeInTheDocument();
  });

  it("collapses and expands the desktop rail", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "Collapse curriculum rail" }));
    expect(screen.queryByRole("complementary", { name: "Curriculum" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Expand curriculum rail" }));
    expect(screen.getByRole("complementary", { name: "Curriculum" })).toBeInTheDocument();
  });

  it("opens the mobile drawer and closes it on Escape", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "Open curriculum" }));
    const drawer = screen.getByRole("dialog", { name: "Curriculum" });
    expect(drawer).toBeInTheDocument();
    fireEvent.keyDown(screen.getByRole("dialog", { name: "Curriculum" }), { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Curriculum" })).not.toBeInTheDocument();
  });

  it("renders the table of contents when provided", () => {
    render(<Harness toc={<div>toc content</div>} />);
    expect(screen.getByRole("complementary", { name: "Table of contents" })).toBeInTheDocument();
    expect(screen.getByText("toc content")).toBeInTheDocument();
  });
});
