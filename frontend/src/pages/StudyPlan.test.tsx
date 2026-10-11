import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import StudyPlanPage from "./StudyPlan";
import {
  completePlanTask,
  fetchGamificationSummary,
  fetchLearnerProfile,
  fetchRecommendations,
  fetchStudyPlan,
  fetchTopics,
  uncompletePlanTask,
  updateDailyGoal,
} from "@/services/api";
import type {
  GamificationSummaryResponse,
  LearnerProfile,
  PlanCommitmentSummary,
  StudyPlanResponse,
} from "@/types";

vi.mock("@/services/api", () => ({
  fetchStudyPlan: vi.fn(),
  fetchTopics: vi.fn(),
  fetchLearnerProfile: vi.fn(),
  fetchRecommendations: vi.fn(),
  fetchGamificationSummary: vi.fn(),
  runProfileDiagnostic: vi.fn(),
  completePlanTask: vi.fn(),
  uncompletePlanTask: vi.fn(),
  updateDailyGoal: vi.fn(),
  updateLearnerProfile: vi.fn(),
}));

const profile: LearnerProfile = {
  target_role: "Backend Engineer",
  target_date: "",
  weekly_minutes: 240,
  preferred_session_minutes: 30,
  current_level: "mid",
  target_level: "senior",
  primary_track: "backend",
  focus_topic_ids: [],
  target_companies: [],
  preferred_modalities: ["study", "quiz", "interview"],
  confidence_by_track: { backend: 3, frontend: 3, system_design: 3, ai_stack: 3 },
  created_at: "2026-10-01T00:00:00Z",
  updated_at: "2026-10-01T00:00:00Z",
  diagnostic_updated_at: "",
};

const commitment: PlanCommitmentSummary = {
  daily_goal: 3,
  today_progress: 1,
  current_streak: 4,
  longest_streak: 6,
  streak_at_risk: false,
  streak_lapsed: false,
  behind: false,
  catch_up: true,
  requested_daily_items: 3,
  effective_daily_items: 3,
  completed_tasks: 1,
  pending_tasks: 5,
};

function studyPlan(overrides: Partial<StudyPlanResponse> = {}): StudyPlanResponse {
  return {
    generated_at: "2026-10-11T09:00:00Z",
    days: 7,
    daily_items: 3,
    total_tasks: 6,
    days_plan: [
      {
        day_index: 1,
        label: "Today",
        date: "2026-10-11",
        tasks: [
          {
            task_type: "topic_study",
            topic_id: "01-backend-fundamentals-and-http",
            title: "Deepen Backend Fundamentals And Http",
            reason: "Your mastery is still low here.",
            estimated_minutes: 30,
            cta_route: "/topics/01-backend-fundamentals-and-http",
            task_key: "key-today-1",
            completed: true,
          },
          {
            task_type: "quiz",
            topic_id: "01-backend-fundamentals-and-http",
            title: "Validate retention in Backend Fundamentals And Http",
            reason: "Use a quiz pass to check retention.",
            estimated_minutes: 15,
            cta_route: "/quiz/01-backend-fundamentals-and-http",
            task_key: "key-today-2",
            completed: false,
          },
        ],
      },
      {
        day_index: 2,
        label: "Day 2",
        date: "2026-10-12",
        tasks: [],
      },
    ],
    commitment,
    ...overrides,
  };
}

const gamification: GamificationSummaryResponse = {
  total_xp: 240,
  level: 1,
  xp_progress: 0.48,
  current_streak: 4,
  longest_streak: 6,
  daily_goal: 3,
  today_progress: 1,
  badges: ["first_review"],
};

function renderPage() {
  return render(
    <MemoryRouter>
      <StudyPlanPage />
    </MemoryRouter>,
  );
}

function checkboxes(): HTMLInputElement[] {
  return screen
    .getAllByRole("checkbox")
    .filter((node): node is HTMLInputElement => node.tagName === "INPUT");
}

describe("StudyPlanPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchStudyPlan).mockResolvedValue(studyPlan());
    vi.mocked(fetchTopics).mockResolvedValue([]);
    vi.mocked(fetchLearnerProfile).mockResolvedValue(profile);
    vi.mocked(fetchRecommendations).mockResolvedValue({
      generated_at: "2026-10-11T09:00:00Z",
      profile,
      days_until_target: null,
      items: [],
    });
    vi.mocked(fetchGamificationSummary).mockResolvedValue(gamification);
  });

  it("renders the persisted completion state for today's tasks", async () => {
    renderPage();

    expect(await screen.findByText("7-Day Plan")).toBeInTheDocument();
    const [first, second] = await waitFor(() => {
      const boxes = checkboxes();
      expect(boxes).toHaveLength(2);
      return boxes;
    });
    expect(first.checked).toBe(true);
    expect(second.checked).toBe(false);
    expect(screen.getByText("Today: 1/2 task(s) complete")).toBeInTheDocument();
  });

  it("completes an unchecked task through the API", async () => {
    vi.mocked(completePlanTask).mockResolvedValue({
      task_key: "key-today-2",
      task_date: "2026-10-11",
      task_type: "quiz",
      topic_id: "01-backend-fundamentals-and-http",
      title: "Validate retention in Backend Fundamentals And Http",
      reason: "",
      estimated_minutes: 15,
      cta_route: "/quiz/01-backend-fundamentals-and-http",
      completed: true,
      completed_at: "2026-10-11T10:00:00Z",
    });
    renderPage();

    const boxes = await waitFor(() => {
      const found = checkboxes();
      expect(found).toHaveLength(2);
      return found;
    });
    fireEvent.click(boxes[1]);

    await waitFor(() =>
      expect(completePlanTask).toHaveBeenCalledWith("key-today-2", "2026-10-11"),
    );
    await waitFor(() => expect(checkboxes()[1].checked).toBe(true));
    expect(screen.getByText("Today: 2/2 task(s) complete")).toBeInTheDocument();
  });

  it("reverts the checkbox when completion fails", async () => {
    vi.mocked(completePlanTask).mockRejectedValue(new Error("offline"));
    renderPage();

    const boxes = await waitFor(() => {
      const found = checkboxes();
      expect(found).toHaveLength(2);
      return found;
    });
    fireEvent.click(boxes[1]);

    await waitFor(() =>
      expect(screen.getByText(/Could not save that plan task/)).toBeInTheDocument(),
    );
    expect(checkboxes()[1].checked).toBe(false);
    expect(screen.getByText("Today: 1/2 task(s) complete")).toBeInTheDocument();
  });

  it("un-completes a checked task through the API", async () => {
    vi.mocked(uncompletePlanTask).mockResolvedValue({
      task_key: "key-today-1",
      task_date: "2026-10-11",
      task_type: "topic_study",
      topic_id: "01-backend-fundamentals-and-http",
      title: "Deepen Backend Fundamentals And Http",
      reason: "",
      estimated_minutes: 30,
      cta_route: "/topics/01-backend-fundamentals-and-http",
      completed: false,
      completed_at: null,
    });
    renderPage();

    const boxes = await waitFor(() => {
      const found = checkboxes();
      expect(found).toHaveLength(2);
      return found;
    });
    fireEvent.click(boxes[0]);

    await waitFor(() =>
      expect(uncompletePlanTask).toHaveBeenCalledWith("key-today-1", "2026-10-11"),
    );
    await waitFor(() => expect(checkboxes()[0].checked).toBe(false));
  });

  it("edits the daily goal through the gamification API", async () => {
    vi.mocked(updateDailyGoal).mockResolvedValue({ daily_goal: 5 });
    renderPage();

    const input = (await screen.findByLabelText(
      /Daily goal/i,
    )) as HTMLInputElement;
    expect(input.value).toBe("3");

    fireEvent.change(input, { target: { value: "5" } });
    fireEvent.click(screen.getByRole("button", { name: /Save Goal/i }));

    await waitFor(() => expect(updateDailyGoal).toHaveBeenCalledWith(5));
    await waitFor(() =>
      expect(
        screen.getByText(/Daily goal set to 5 learning event/),
      ).toBeInTheDocument(),
    );
  });

  it("surfaces the streak-risk banner and halves the displayed daily load", async () => {
    vi.mocked(fetchStudyPlan).mockResolvedValue(
      studyPlan({
        daily_items: 2,
        commitment: {
          ...commitment,
          streak_at_risk: true,
          behind: true,
          catch_up: true,
          requested_daily_items: 3,
          effective_daily_items: 2,
        },
      }),
    );
    renderPage();

    expect(
      await screen.findByText(/today's load was halved/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/2 item\(s\)\/day/)).toBeInTheDocument();
  });
});
