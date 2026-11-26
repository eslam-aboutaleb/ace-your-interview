import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import axe from "axe-core";
import {
  createMemoryRouter,
  MemoryRouter,
  Route,
  RouterProvider,
  Routes,
  useLocation,
} from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  fetchJourney,
  fetchJourneyProgress,
  fetchReaderState,
  patchJourneyCheckpoint,
  recordQuestionAttempt,
  setQuestionCompletion,
} from "@/services/api";
import type {
  JourneyDetail,
  JourneyQuestionSlot,
  ReaderCheckpoint,
  ReaderStateResponse,
} from "@/types";
import StudyReaderPage from "./StudyReaderPage";
import { invalidateReaderData } from "../readerData";
import { useReaderUiStore } from "../store/readerUiStore";

vi.mock("@/services/api", () => ({
  fetchJourney: vi.fn(),
  fetchReaderState: vi.fn(),
  patchJourneyCheckpoint: vi.fn(),
  fetchJourneyProgress: vi.fn(),
  setQuestionCompletion: vi.fn(),
  recordQuestionAttempt: vi.fn(),
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
  return (
    <div data-testid="location">
      {location.pathname}
      {location.search}
    </div>
  );
}

const emptySummary = {
  journey_id: "j1",
  total_questions: 2,
  completed: 0,
  studied: 0,
  practiced: 0,
  mastered: 0,
  percent_complete: 0,
  states: [],
};

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
    invalidateReaderData();
    useReaderUiStore.setState({ prefsByUser: {} });
    vi.mocked(fetchJourney).mockResolvedValue(journeyDetail());
    vi.mocked(fetchReaderState).mockResolvedValue(readerState());
    vi.mocked(fetchJourneyProgress).mockResolvedValue(emptySummary);
    vi.mocked(setQuestionCompletion).mockResolvedValue({
      api_version: "v1",
      state: { ...emptySummary, slot_id: "q1", completed: true } as never,
      summary: { ...emptySummary, completed: 1 },
    });
    vi.mocked(recordQuestionAttempt).mockResolvedValue({
      api_version: "v1",
      state: { ...emptySummary, slot_id: "q1", practiced: true } as never,
      summary: { ...emptySummary, practiced: 1 },
    });
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
    await waitFor(
      () => {
        expect(patchJourneyCheckpoint).toHaveBeenCalledWith("j1", {
          expected_revision: 3,
          mutation_id: expect.any(String),
          last_batch_id: "b1",
          last_question_id: "q2",
          scroll_offset: 0,
        });
      },
      { timeout: 3000 },
    );
  });

  it("re-syncs from the server when the checkpoint patch conflicts", async () => {
    vi.mocked(patchJourneyCheckpoint).mockRejectedValue({
      isAxiosError: true,
      response: { status: 409 },
    });
    renderPage("/journeys/j1/batches/b1/questions/q2");
    await screen.findByRole("heading", { level: 1, name: "Objective Two" });
    await waitFor(
      () => {
        expect(fetchReaderState).toHaveBeenCalledTimes(2);
      },
      { timeout: 3000 },
    );
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
    expect(screen.getByRole("status", { name: "Current question" })).toHaveTextContent("Question 1 of 2: Objective One");

    fireEvent.keyDown(document.body, { key: "ArrowRight" });
    await waitFor(() => {
      expect(screen.getByTestId("location")).toHaveTextContent(
        "/journeys/j1/batches/b1/questions/q2",
      );
    });
    expect(screen.getByRole("status", { name: "Current question" })).toHaveTextContent("Question 2 of 2: Objective Two");

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

  it("serves a param change from the cache without refetching", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q1");
    await screen.findByRole("heading", { level: 1, name: "Objective One" });
    fireEvent.click(screen.getByRole("button", { name: "Next question" }));
    await screen.findByRole("heading", { level: 1, name: "Objective Two" });
    expect(fetchJourney).toHaveBeenCalledTimes(1);
    expect(fetchReaderState).toHaveBeenCalledTimes(1);
  });

  it("supports back and forward navigation via the URL", async () => {
    const router = createMemoryRouter(
      [
        { path: "/journeys/:journeyId", element: <StudyReaderPage /> },
        {
          path: "/journeys/:journeyId/batches/:batchId/questions/:questionId",
          element: <StudyReaderPage />,
        },
      ],
      {
        initialEntries: [
          "/journeys/j1/batches/b1/questions/q1",
          "/journeys/j1/batches/b1/questions/q2",
        ],
        initialIndex: 1,
      },
    );
    render(<RouterProvider router={router} />);
    await screen.findByRole("heading", { level: 1, name: "Objective Two" });
    await router.navigate(-1);
    await screen.findByRole("heading", { level: 1, name: "Objective One" });
    await router.navigate(1);
    await screen.findByRole("heading", { level: 1, name: "Objective Two" });
  });

  it("applies UI prefs: focus mode hides the rail, width and font scale persist", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q1");
    await screen.findByRole("heading", { level: 1, name: "Objective One" });
    expect(screen.getByRole("complementary", { name: "Curriculum" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Enter focus mode" }));
    expect(screen.queryByRole("complementary", { name: "Curriculum" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Exit focus mode" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    fireEvent.click(screen.getByRole("button", { name: "Exit focus mode" }));

    fireEvent.click(screen.getByRole("button", { name: "Increase font size" }));
    fireEvent.click(screen.getByRole("button", { name: "Toggle wide layout" }));
    const prefs = useReaderUiStore.getState().prefsByUser.anonymous;
    expect(prefs.fontScale).toBe(1.125);
    expect(prefs.contentWidth).toBe("wide");
    expect(screen.getByRole("article", { name: "Question overview" }).className).toContain(
      "max-w-5xl",
    );
  });

  it("shows saving then saved status while the checkpoint autosaves", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q2");
    await screen.findByRole("heading", { level: 1, name: "Objective Two" });
    await waitFor(
      () => {
        expect(screen.getByRole("status", { name: "Save status" })).toHaveTextContent(
          "Saved",
        );
      },
      { timeout: 3000 },
    );
  });

  it("switches modes through the URL query param", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q1");
    await screen.findByRole("heading", { level: 1, name: "Objective One" });
    expect(screen.getByRole("button", { name: "Study" })).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(screen.getByRole("button", { name: "Practice" }));
    await waitFor(() => {
      expect(screen.getByTestId("location")).toHaveTextContent("mode=practice");
    });
    expect(screen.getByRole("button", { name: "Practice" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("practice mode masks the source section until attempt + reveal, then grades", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q1?mode=practice");
    await screen.findByRole("heading", { level: 1, name: "Objective One" });
    expect(screen.getByText("Hidden until you reveal")).toBeInTheDocument();

    const reveal = screen.getByRole("button", { name: "Reveal answer" });
    expect(reveal).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Your attempt"), {
      target: { value: "My attempt" },
    });
    expect(reveal).toBeEnabled();
    fireEvent.click(reveal);
    expect(screen.getByText("Section A")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "I got it" }));
    await waitFor(() => {
      expect(recordQuestionAttempt).toHaveBeenCalledWith("j1", "q1", {
        outcome: "correct",
        confidence: 4,
        response_time_ms: expect.any(Number),
        mutation_id: expect.any(String),
      });
    });
    expect(await screen.findByText("Attempt recorded.")).toBeInTheDocument();
  });

  it("review mode offers the four FSRS-style grades after reveal", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q1?mode=review");
    await screen.findByRole("heading", { level: 1, name: "Objective One" });
    fireEvent.change(screen.getByLabelText("Your attempt"), {
      target: { value: "Recall attempt" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Reveal answer" }));
    for (const label of ["Again", "Hard", "Good", "Easy"]) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
    fireEvent.click(screen.getByRole("button", { name: "Again" }));
    await waitFor(() => {
      expect(recordQuestionAttempt).toHaveBeenCalledWith("j1", "q1", {
        outcome: "incorrect",
        confidence: 1,
        response_time_ms: expect.any(Number),
        mutation_id: expect.any(String),
      });
    });
  });

  it("interview mode requires a submitted response before reveal", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q1?mode=interview");
    await screen.findByRole("heading", { level: 1, name: "Objective One" });
    expect(screen.queryByRole("button", { name: "Reveal answer" })).not.toBeInTheDocument();

    const submit = screen.getByRole("button", { name: "Submit response" });
    expect(submit).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Your response"), {
      target: { value: "My interview answer" },
    });
    fireEvent.click(submit);
    fireEvent.click(screen.getByRole("button", { name: "Reveal answer" }));
    expect(screen.getByText("Section A")).toBeInTheDocument();
  });

  it("study mode toggles explicit completion", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q1");
    await screen.findByRole("heading", { level: 1, name: "Objective One" });
    fireEvent.click(screen.getByRole("button", { name: "Mark as complete" }));
    await waitFor(() => {
      expect(setQuestionCompletion).toHaveBeenCalledWith("j1", "q1", {
        completed: true,
        mutation_id: expect.any(String),
      });
    });
  });

  it("never inflates progress from navigation alone", async () => {
    renderPage("/journeys/j1/batches/b1/questions/q1");
    await screen.findByRole("heading", { level: 1, name: "Objective One" });
    fireEvent.keyDown(document.body, { key: "ArrowRight" });
    await screen.findByRole("heading", { level: 1, name: "Objective Two" });
    expect(recordQuestionAttempt).not.toHaveBeenCalled();
    expect(setQuestionCompletion).not.toHaveBeenCalled();
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
