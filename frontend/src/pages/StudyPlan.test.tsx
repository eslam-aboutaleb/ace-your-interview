import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import StudyPlanPage from "./StudyPlan";
import {
  completePlanTask,
  fetchGamificationSummary,
  fetchLearnerProfile,
  fetchReadiness,
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
  RoleReadinessResponse,
  StudyPlanResponse,
} from "@/types";

vi.mock("@/services/api", () => ({
  fetchStudyPlan: vi.fn(),
  fetchTopics: vi.fn(),
  fetchLearnerProfile: vi.fn(),
  fetchRecommendations: vi.fn(),
  fetchGamificationSummary: vi.fn(),
  fetchReadiness: vi.fn(),
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

function readiness(
  overrides: Partial<RoleReadinessResponse> = {},
): RoleReadinessResponse {
  return {
    generated_at: "2026-10-11T09:00:00Z",
    profile,
    role: {
      target_role: "Backend Engineer",
      role_id: "backend_engineer",
      role_label: "Backend Engineer",
      match_source: "catalog",
      readiness: 72.4,
      readiness_label: "On Track",
      track_weights: { backend: 0.6, system_design: 0.4 },
      competency_weights: { technical_accuracy: 0.6 },
      track_breakdown: [],
      competency_breakdown: [],
      gap_dimensions: [
        {
          kind: "track",
          id: "backend",
          label: "Backend",
          weight: 0.6,
          score: 62,
          evidence_count: 12,
          gap_points: 22.8,
          evidence: [
            {
              kind: "topic",
              ref: "03-indexing-and-query-plans",
              label: "Indexing And Query Plans",
              score: 41,
              count: 0,
              detail: "4 review(s), 2 due",
            },
          ],
        },
        {
          kind: "interview_competency",
          id: "technical_accuracy",
          label: "Technical Accuracy",
          weight: 0.6,
          score: 55,
          evidence_count: 6,
          gap_points: 9.6,
          evidence: [],
        },
        {
          kind: "track",
          id: "system_design",
          label: "System Design",
          weight: 0.4,
          score: 88,
          evidence_count: 4,
          gap_points: 4.8,
          evidence: [],
        },
      ],
      evidence: {
        topic_reviews: 12,
        interview_reports: 2,
        interview_competency_reviews: 18,
        error_patterns: 5,
      },
    },
    best_fit_roles: [],
    companies: [
      {
        company: "Datadog",
        company_key: "datadog",
        company_label: "Datadog",
        match_source: "default",
        fit: 83.1,
        fit_label: "Strong",
        bar: 74,
        clears_bar: true,
        track_weights: {},
        competency_weights: {},
        gap_dimensions: [],
      },
      {
        company: "Stripe",
        company_key: "stripe",
        company_label: "Stripe",
        match_source: "catalog",
        fit: 61.3,
        fit_label: "Developing",
        bar: 70,
        clears_bar: false,
        track_weights: {},
        competency_weights: {},
        gap_dimensions: [],
      },
    ],
    ...overrides,
  };
}

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
    vi.mocked(fetchReadiness).mockResolvedValue(readiness());
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

  it("shows the role and company readiness strip on the 7-day plan card", async () => {
    renderPage();

    const strip = await screen.findByTestId("readiness-strip");
    expect(strip).toHaveTextContent(/Ranked for/i);
    expect(strip).toHaveTextContent("Backend Engineer");
    expect(strip).toHaveTextContent("72% ready");
    expect(strip).toHaveTextContent("Datadog");
    expect(strip).toHaveTextContent("83%");
    expect(strip).toHaveTextContent("Strong");
    expect(strip).not.toHaveTextContent("Stripe");
    expect(
      within(strip).getByRole("link", { name: /Role readiness/i }),
    ).toHaveAttribute("href", "/readiness");
  });

  it("lists the weighted gaps the plan prioritises", async () => {
    renderPage();

    const strip = await screen.findByTestId("readiness-strip");
    expect(strip).toHaveTextContent(/Biggest readiness levers/i);
    expect(strip).toHaveTextContent("Backend (+22.8 pts)");
    expect(strip).toHaveTextContent("Technical Accuracy (+9.6 pts)");
    expect(strip).not.toHaveTextContent("System Design (+4.8 pts)");
  });

  it("hides the readiness strip when readiness cannot load", async () => {
    vi.mocked(fetchReadiness).mockRejectedValue(new Error("offline"));
    renderPage();

    expect(await screen.findByText("7-Day Plan")).toBeInTheDocument();
    await waitFor(() => expect(fetchReadiness).toHaveBeenCalled());
    expect(screen.queryByTestId("readiness-strip")).not.toBeInTheDocument();
  });
});
