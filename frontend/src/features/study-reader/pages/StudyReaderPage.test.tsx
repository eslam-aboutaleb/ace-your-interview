import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import axe from "axe-core";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  fetchJourney,
  fetchReaderState,
  patchJourneyCheckpoint,
} from "@/services/api";
import type {
  JourneyDetail,
  JourneyQuestionSlot,
  ReaderCheckpoint,
  ReaderStateResponse,
} from "@/types";
import StudyReaderPage from "./StudyReaderPage";

vi.mock("@/services/api", () => ({
  fetchJourney: vi.fn(),
  fetchReaderState: vi.fn(),
  patchJourneyCheckpoint: vi.fn(),
}));

function journeyDetail(): JourneyDetail {
  return {
    id: "j1",
    title: "Backend Journey",
    goal: "Pass the interview",
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
        description: "Foundations",
        modules: [
          {
            id: "m1",
            phase_id: "p1",
            position: 0,
            title: "Module One",
            description: "Basics",
            objectives: [
              {
                id: "o1",
                module_id: "m1",
                position: 0,
                title: "Objective One",
                kind: "concept",
                source_section: "Section A",
                status: "in_progress",
              },
              {
                id: "o2",
                module_id: "m1",
                position: 1,
                title: "Objective Two",
                kind: "practice",
                source_section: "Section B",
                status: "pending",
              },
            ],
          },
        ],
      },
    ],
  };
}

function slot(overrides: Partial<JourneyQuestionSlot> = {}): JourneyQuestionSlot {
  return {
    id: "q1",
    batch_id: "b1",
    position: 0,
    objective_id: "o1",
    objective_title: "Objective One",
    question_ref: "ref-1",
    status: "accepted",
    ...overrides,
  };
}

function checkpoint(overrides: Partial<ReaderCheckpoint> = {}): ReaderCheckpoint {
  return {
    last_batch_id: "b1",
    last_question_id: "q1",
    last_section_id: "",
    scroll_offset: 0,
    revision: 3,
    updated_at: "2026-01-02T00:00:00Z",
    ...overrides,
  };
}

function readerState(overrides: Partial<ReaderStateResponse> = {}): ReaderStateResponse {
  return {
    api_version: "v1",
    journey_id: "j1",
    checkpoint: checkpoint(),
    latest_batch: {
      id: "b1",
      journey_id: "j1",
      batch_number: 1,
      idempotency_key: "key-1",
      plan_version: 1,
      prompt_version: "p1",
      model_version: "m1",
      schema_version: "s1",
      status: "completed",
      accepted_count: 2,
      rejected_count: 0,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    questions: [
      slot(),
      slot({ id: "q2", position: 1, objective_id: "o2", objective_title: "Objective Two" }),
    ],
    ...overrides,
  };
}

function LocationProbe() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderPage(initialEntry: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/journeys/:journeyId" element={<StudyReaderPage />} />
        <Route
          path="/journeys/:journeyId/batches/:batchId/questions/:questionId"
          element={<StudyReaderPage />}
        />
      </Routes>
      <LocationProbe />
    </MemoryRouter>,
  );
}

describe("StudyReaderPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchJourney).mockResolvedValue(journeyDetail());
    vi.mocked(fetchReaderState).mockResolvedValue(readerState());
    vi.mocked(patchJourneyCheckpoint).mockResolvedValue({
      api_version: "v1",
      checkpoint: checkpoint({ revision: 4, last_question_id: "q2" }),
    });
  });

  it("shows a skeleton while loading, then the objective overview", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q1");
    expect(screen.getByTestId("reader-skeleton")).toBeInTheDocument();
    expect(
      await screen.findByRole("heading", { level: 1, name: "Objective One" }),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("reader-skeleton")).not.toBeInTheDocument();
    expect(screen.getByText("Section A")).toBeInTheDocument();
    expect(screen.getByText("1 of 2")).toBeInTheDocument();
    expect(screen.getByText("Backend Journey")).toBeInTheDocument();
  });

  it("restores the checkpoint position when the URL has no question", async () => {
    renderPage("/journeys/j1");
    await waitFor(() => {
      expect(screen.getByTestId("location")).toHaveTextContent(
        "/journeys/j1/batches/b1/questions/q1",
      );
    });
  });

  it("navigates between questions with the toolbar", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q2");
    expect(
      await screen.findByRole("heading", { level: 1, name: "Objective Two" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next question" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Previous question" }));
    await waitFor(() => {
      expect(screen.getByTestId("location")).toHaveTextContent(
        "/journeys/j1/batches/b1/questions/q1",
      );
    });
    expect(screen.getByRole("button", { name: "Previous question" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Next question" })).toBeEnabled();
  });

  it("persists the checkpoint when the active question changes", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q2");
    await screen.findByRole("heading", { level: 1, name: "Objective Two" });
    await waitFor(() => {
      expect(patchJourneyCheckpoint).toHaveBeenCalledWith("j1", {
        expected_revision: 3,
        mutation_id: expect.any(String),
        last_batch_id: "b1",
        last_question_id: "q2",
      });
    });
  });

  it("re-syncs from the server when the checkpoint patch conflicts", async () => {
    vi.mocked(patchJourneyCheckpoint).mockRejectedValue({
      isAxiosError: true,
      response: { status: 409 },
    });
    renderPage("/journeys/j1/batches/b1/questions/q2");
    await screen.findByRole("heading", { level: 1, name: "Objective Two" });
    await waitFor(() => {
      expect(fetchReaderState).toHaveBeenCalledTimes(2);
    });
  });

  it("shows a not-found error and recovers on retry", async () => {
    vi.mocked(fetchJourney).mockRejectedValueOnce({
      isAxiosError: true,
      response: { status: 404 },
    });
    renderPage("/journeys/j1/batches/b1/questions/q1");
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This journey could not be found.",
    );

    vi.mocked(fetchJourney).mockResolvedValueOnce(journeyDetail());
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(
      await screen.findByRole("heading", { level: 1, name: "Objective One" }),
    ).toBeInTheDocument();
  });

  it("shows an empty state when no questions are generated yet", async () => {
    vi.mocked(fetchReaderState).mockResolvedValue(
      readerState({ questions: [], latest_batch: null }),
    );
    renderPage("/journeys/j1/batches/b1/questions/q1");
    expect(
      await screen.findByText("No questions generated yet."),
    ).toBeInTheDocument();
  });

  it("turns pages with plain arrow keys and announces the change once", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q1");
    await screen.findByRole("heading", { level: 1, name: "Objective One" });
    expect(screen.getByRole("status")).toHaveTextContent("Question 1 of 2: Objective One");

    fireEvent.keyDown(document.body, { key: "ArrowRight" });
    await waitFor(() => {
      expect(screen.getByTestId("location")).toHaveTextContent(
        "/journeys/j1/batches/b1/questions/q2",
      );
    });
    expect(screen.getByRole("status")).toHaveTextContent("Question 2 of 2: Objective Two");

    fireEvent.keyDown(document.body, { key: "ArrowRight" });
    expect(screen.getByTestId("location")).toHaveTextContent(
      "/journeys/j1/batches/b1/questions/q2",
    ); // clamped at the last question

    fireEvent.keyDown(document.body, { key: "ArrowLeft" });
    await waitFor(() => {
      expect(screen.getByTestId("location")).toHaveTextContent(
        "/journeys/j1/batches/b1/questions/q1",
      );
    });
  });

  it("ignores arrow keys while typing, editing, or selecting", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q1");
    await screen.findByRole("heading", { level: 1, name: "Objective One" });

    const input = document.createElement("input");
    document.body.appendChild(input);
    fireEvent.keyDown(input, { key: "ArrowRight" });
    expect(screen.getByTestId("location")).toHaveTextContent(
      "/journeys/j1/batches/b1/questions/q1",
    );
    input.remove();

    const editable = document.createElement("div");
    editable.setAttribute("contenteditable", "true");
    document.body.appendChild(editable);
    fireEvent.keyDown(editable, { key: "ArrowRight" });
    expect(screen.getByTestId("location")).toHaveTextContent(
      "/journeys/j1/batches/b1/questions/q1",
    );
    editable.remove();

    fireEvent.keyDown(document.body, { key: "ArrowRight", ctrlKey: true });
    expect(screen.getByTestId("location")).toHaveTextContent(
      "/journeys/j1/batches/b1/questions/q1",
    );

    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(screen.getByRole("heading", { level: 1 }));
    selection?.removeAllRanges();
    selection?.addRange(range);
    fireEvent.keyDown(document.body, { key: "ArrowRight" });
    expect(screen.getByTestId("location")).toHaveTextContent(
      "/journeys/j1/batches/b1/questions/q1",
    );
    selection?.removeAllRanges();
  });

  it("has no axe violations on the loaded reader", async () => {
    const { container } = renderPage("/journeys/j1/batches/b1/questions/q1");
    await screen.findByRole("heading", { level: 1, name: "Objective One" });
    const results = await axe.run(container, {
      rules: { "color-contrast": { enabled: false } },
    });
    expect(results.violations).toEqual([]);
  });

  it("only navigates to accepted slots", async () => {
    vi.mocked(fetchReaderState).mockResolvedValue(
      readerState({
        checkpoint: checkpoint({ last_question_id: "q-rejected" }),
        questions: [
          slot({ id: "q-rejected", status: "rejected" }),
          slot({ id: "q2", position: 1, objective_id: "o2", objective_title: "Objective Two" }),
        ],
      }),
    );
    renderPage("/journeys/j1");
    await waitFor(() => {
      expect(screen.getByTestId("location")).toHaveTextContent(
        "/journeys/j1/batches/b1/questions/q2",
      );
    });
  });
});
