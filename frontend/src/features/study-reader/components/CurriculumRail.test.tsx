import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import type { JourneyDetail, JourneyQuestionSlot } from "@/types";
import CurriculumRail from "./CurriculumRail";

const journey: JourneyDetail = {
  id: "j1",
  title: "Backend Journey",
  goal: "goal",
  source_type: "static_topic",
  source_topic_id: "backend",
  track: "backend",
  level: "mid",
  status: "active",
  plan_version: 1,
  objective_count: 2,
  batch_count: 1,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  phases: [
    {
      id: "p1",
      position: 0,
      title: "Phase One",
      description: "",
      modules: [
        {
          id: "m1",
          phase_id: "p1",
          position: 0,
          title: "Module One",
          description: "",
          objectives: [
            {
              id: "o1",
              module_id: "m1",
              position: 0,
              title: "Objective One",
              kind: "concept",
              source_section: "A",
              status: "in_progress",
            },
            {
              id: "o2",
              module_id: "m1",
              position: 1,
              title: "Objective Two",
              kind: "practice",
              source_section: "B",
              status: "pending",
            },
            {
              id: "o3",
              module_id: "m1",
              position: 2,
              title: "Objective Three",
              kind: "concept",
              source_section: "C",
              status: "pending",
            },
          ],
        },
      ],
    },
  ],
};

const slots: JourneyQuestionSlot[] = [
  {
    id: "q1",
    batch_id: "b1",
    position: 0,
    objective_id: "o1",
    objective_title: "Objective One",
    question_ref: "r1",
    status: "accepted",
  },
  {
    id: "q2",
    batch_id: "b1",
    position: 1,
    objective_id: "o2",
    objective_title: "Objective Two",
    question_ref: "r2",
    status: "accepted",
  },
];

function renderRail(activeQuestionId: string | null = "q1") {
  return render(
    <MemoryRouter>
      <CurriculumRail journey={journey} slots={slots} activeQuestionId={activeQuestionId} />
    </MemoryRouter>,
  );
}

describe("CurriculumRail", () => {
  it("links objectives with generated slots and leaves others unlinked", () => {
    renderRail();
    expect(screen.getByRole("link", { name: /Objective One/ })).toHaveAttribute(
      "href",
      "/journeys/j1/batches/b1/questions/q1",
    );
    expect(screen.getByRole("link", { name: /Objective Two/ })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Objective Three/ })).not.toBeInTheDocument();
    expect(screen.getByText("Objective Three")).toBeInTheDocument();
  });

  it("marks the active question with aria-current", () => {
    renderRail("q2");
    expect(screen.getByRole("link", { name: /Objective Two/ })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(screen.getByRole("link", { name: /Objective One/ })).not.toHaveAttribute(
      "aria-current",
    );
  });

  it("moves focus between objective links with arrow keys", () => {
    renderRail();
    const first = screen.getByRole("link", { name: /Objective One/ });
    const second = screen.getByRole("link", { name: /Objective Two/ });
    first.focus();
    fireEvent.keyDown(first, { key: "ArrowDown" });
    expect(second).toHaveFocus();
    fireEvent.keyDown(second, { key: "ArrowDown" });
    expect(second).toHaveFocus(); // clamped at the end
    fireEvent.keyDown(second, { key: "ArrowUp" });
    expect(first).toHaveFocus();
  });

  it("uses one heading level per plan tier (phase h2, module h3)", () => {
    renderRail();
    expect(screen.getByRole("heading", { level: 2, name: "Phase One" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 3, name: "Module One" })).toBeInTheDocument();
  });
});
