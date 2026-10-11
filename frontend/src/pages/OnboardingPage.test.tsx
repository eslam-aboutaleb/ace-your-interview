import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import OnboardingPage from "./OnboardingPage";
import {
  answerPlacement,
  finishPlacement,
  startPlacement,
} from "@/services/api";
import type {
  PlacementAnswerResponse,
  PlacementFinishResponse,
  PlacementQuestion,
  PlacementStartResponse,
} from "@/types";

vi.mock("@/services/api", () => ({
  startPlacement: vi.fn(),
  answerPlacement: vi.fn(),
  finishPlacement: vi.fn(),
  fetchPlacementStatus: vi.fn(),
}));

function placementQuestion(
  overrides: Partial<PlacementQuestion> = {},
): PlacementQuestion {
  return {
    question_id: "placement:01-backend-fundamentals-and-http:get-method:mid",
    question: 'Which curriculum topic covers "GET method"?',
    options: [
      "Backend Fundamentals and HTTP",
      "Frontend Core Architecture",
      "System Design Foundations",
      "AI Stack RAG and Evaluation",
    ],
    level: "mid",
    topic_id: "01-backend-fundamentals-and-http",
    topic_title: "Backend Fundamentals and HTTP",
    concept: "GET method",
    source: "catalog",
    ...overrides,
  };
}

function startResponse(
  overrides: Partial<PlacementStartResponse> = {},
): PlacementStartResponse {
  return {
    session_id: "placement-session-1",
    track: "backend",
    status: "active",
    level: "mid",
    step_index: 0,
    max_steps: 12,
    question: placementQuestion(),
    ...overrides,
  };
}

function answerResponse(
  overrides: Partial<PlacementAnswerResponse> = {},
): PlacementAnswerResponse {
  return {
    session_id: "placement-session-1",
    status: "active",
    level: "senior",
    step_index: 1,
    max_steps: 12,
    is_correct: true,
    correct_answer: "A",
    explanation: '"GET method" belongs to Backend Fundamentals and HTTP.',
    complete: false,
    question: placementQuestion({
      question_id: "placement:01-backend-fundamentals-and-http:post-method:senior",
      question: 'Which curriculum topic covers "POST method"?',
      concept: "POST method",
      level: "senior",
    }),
    ...overrides,
  };
}

function finishResponse(): PlacementFinishResponse {
  return {
    session_id: "placement-session-1",
    track: "backend",
    derived_level: "mid",
    per_track_levels: { backend: "mid" },
    steps_taken: 8,
    weak_objectives: [
      {
        topic_id: "11-system-design-scaling-and-reliability",
        topic_title: "System Design Scaling and Reliability",
        objective: "Database sharding strategies",
        level: "senior",
      },
    ],
    level_path: ["mid", "senior", "mid", "senior"],
    level_stats: {
      junior: { correct: 0, answered: 0 },
      mid: { correct: 4, answered: 4 },
      senior: { correct: 1, answered: 3 },
    },
    profile: null,
  };
}

function renderPage() {
  return render(
    <MemoryRouter>
      <OnboardingPage />
    </MemoryRouter>,
  );
}

async function startQuiz() {
  vi.mocked(startPlacement).mockResolvedValue(startResponse());
  renderPage();
  fireEvent.click(screen.getByRole("button", { name: /get started/i }));
  fireEvent.click(
    screen.getByRole("button", { name: /start placement quiz/i }),
  );
}

describe("OnboardingPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("walks from the welcome step into the goal intake", async () => {
    vi.mocked(startPlacement).mockResolvedValue(startResponse());
    renderPage();

    expect(
      screen.getByRole("heading", { name: /find your starting level/i }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /get started/i }));

    expect(
      await screen.findByRole("heading", { name: "Your goal" }),
    ).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText(/e.g. Senior Backend Engineer/i),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("combobox")).toHaveLength(2);
    expect(screen.getByText(/daily goal \(items per day\)/i)).toBeInTheDocument();
  });

  it("starts the placement quiz with the intake answers", async () => {
    await startQuiz();

    await waitFor(() =>
      expect(startPlacement).toHaveBeenCalledWith({
        track: "backend",
        target_role: undefined,
        daily_goal: 3,
      }),
    );
    expect(
      screen.getByRole("heading", {
        name: /which curriculum topic covers "get method"/i,
      }),
    ).toBeInTheDocument();
    expect(screen.getByText(/level: mid/i)).toBeInTheDocument();
    expect(screen.getByTestId("placement-progress")).toHaveAttribute(
      "aria-label",
      "Placement progress: 0 of 12",
    );
    expect(screen.getAllByRole("button", { name: /^[A-D]\./ })).toHaveLength(4);
  });

  it("passes the role, track and daily goal from the intake", async () => {
    vi.mocked(startPlacement).mockResolvedValue(startResponse());
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /get started/i }));

    fireEvent.change(
      screen.getByPlaceholderText(/e.g. Senior Backend Engineer/i),
      {
        target: { value: "Senior Backend Engineer" },
      },
    );
    const [trackSelect, goalSelect] = screen.getAllByRole("combobox");
    fireEvent.change(trackSelect, { target: { value: "frontend" } });
    fireEvent.change(goalSelect, { target: { value: "5" } });
    fireEvent.click(
      screen.getByRole("button", { name: /start placement quiz/i }),
    );

    await waitFor(() =>
      expect(startPlacement).toHaveBeenCalledWith({
        track: "frontend",
        target_role: "Senior Backend Engineer",
        daily_goal: 5,
      }),
    );
  });

  it("grades an answer and reveals the next question", async () => {
    vi.mocked(answerPlacement).mockResolvedValue(answerResponse());
    await startQuiz();

    fireEvent.click(
      await screen.findByRole("button", {
        name: /A\. Backend Fundamentals and HTTP/,
      }),
    );

    await waitFor(() =>
      expect(answerPlacement).toHaveBeenCalledWith({
        session_id: "placement-session-1",
        question_id: "placement:01-backend-fundamentals-and-http:get-method:mid",
        selected_answer: "A",
      }),
    );
    expect(screen.getByText("Correct")).toBeInTheDocument();
    expect(screen.getByText(/level: senior/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /next question/i }));
    expect(
      screen.getByRole("heading", {
        name: /which curriculum topic covers "post method"/i,
      }),
    ).toBeInTheDocument();
  });

  it("shows the right answer after a wrong pick", async () => {
    vi.mocked(answerPlacement).mockResolvedValue(
      answerResponse({
        is_correct: false,
        correct_answer: "B",
        level: "junior",
        question: placementQuestion({
          question_id: "placement:02-backend-api-design-and-contracts:resource-naming:junior",
          question: 'Which curriculum topic covers "Resource naming conventions"?',
          concept: "Resource naming conventions",
          level: "junior",
        }),
      }),
    );
    await startQuiz();

    fireEvent.click(
      await screen.findByRole("button", {
        name: /A\. Backend Fundamentals and HTTP/,
      }),
    );

    await waitFor(() =>
      expect(screen.getByText(/incorrect/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/the correct answer is B/i)).toBeInTheDocument();
    expect(screen.getByText(/level: junior/i)).toBeInTheDocument();
  });

  it("summarises the placement once the walk completes", async () => {
    vi.mocked(answerPlacement).mockResolvedValue(
      answerResponse({ complete: true, question: null, step_index: 8 }),
    );
    vi.mocked(finishPlacement).mockResolvedValue(finishResponse());
    await startQuiz();

    fireEvent.click(
      await screen.findByRole("button", {
        name: /A\. Backend Fundamentals and HTTP/,
      }),
    );

    await waitFor(() =>
      expect(finishPlacement).toHaveBeenCalledWith({
        session_id: "placement-session-1",
      }),
    );
    expect(
      await screen.findByRole("heading", {
        name: /you start at mid level in backend/i,
      }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("placement-summary")).toBeInTheDocument();
    expect(screen.getByText(/Database sharding strategies/)).toBeInTheDocument();
    expect(screen.getByText(/Backend: Mid/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /open my study plan/i }),
    ).toBeInTheDocument();
  });
});
