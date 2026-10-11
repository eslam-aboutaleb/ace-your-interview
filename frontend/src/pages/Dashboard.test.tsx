import { render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Dashboard from "./Dashboard";
import {
  fetchForecast,
  fetchGamificationSummary,
  fetchInterviewStats,
  fetchLearnerProfile,
  fetchPlacementStatus,
  fetchReadiness,
  fetchRecommendations,
  fetchTopicMastery,
  fetchTopics,
} from "@/services/api";
import { useAuthStore } from "@/store/authStore";
import { useProgressStore } from "@/store/progressStore";
import type {
  ForecastResponse,
  GamificationSummaryResponse,
  LearnerProfile,
  RoleReadinessResponse,
  TopicSummary,
} from "@/types";

vi.mock("@/services/api", () => ({
  fetchTopics: vi.fn(),
  fetchTopicMastery: vi.fn(),
  fetchInterviewStats: vi.fn(),
  fetchForecast: vi.fn(),
  fetchPlacementStatus: vi.fn(),
  fetchLearnerProfile: vi.fn(),
  fetchRecommendations: vi.fn(),
  fetchGamificationSummary: vi.fn(),
  fetchReadiness: vi.fn(),
}));

const topics: TopicSummary[] = [
  {
    id: "01-backend-fundamentals-and-http",
    title: "Backend Fundamentals And Http",
    description: "Requests, responses, and status codes.",
    track: "backend",
    levels: ["junior", "mid"],
    section_count: 4,
    estimated_questions: 24,
  },
];

const profile: LearnerProfile = {
  target_role: "Backend Engineer",
  target_date: "",
  weekly_minutes: 240,
  preferred_session_minutes: 30,
  current_level: "mid",
  target_level: "senior",
  primary_track: "backend",
  focus_topic_ids: [],
  target_companies: ["Stripe", "Datadog"],
  preferred_modalities: ["study", "quiz", "interview"],
  confidence_by_track: { backend: 3, frontend: 3, system_design: 3, ai_stack: 3 },
  created_at: "2026-10-01T00:00:00Z",
  updated_at: "2026-10-01T00:00:00Z",
  diagnostic_updated_at: "2026-10-10T00:00:00Z",
};

const forecast: ForecastResponse = {
  total_due: 3,
  due_counts: [{ date: "2026-10-11", count: 3 }],
};

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
          evidence: [
            {
              kind: "interview_competency",
              ref: "technical_accuracy",
              label: "Technical Accuracy",
              score: 55,
              count: 0,
              detail: "2 review(s)",
            },
          ],
        },
      ],
      evidence: {
        topic_reviews: 12,
        interview_reports: 2,
        interview_competency_reviews: 18,
        error_patterns: 5,
      },
    },
    best_fit_roles: [
      {
        role_id: "backend_engineer",
        role_label: "Backend Engineer",
        readiness: 72.4,
        readiness_label: "On Track",
        is_target: true,
      },
    ],
    companies: [
      {
        company: "Stripe",
        company_key: "stripe",
        company_label: "Stripe",
        match_source: "catalog",
        fit: 61.3,
        fit_label: "Developing",
        bar: 70,
        clears_bar: false,
        track_weights: { backend: 0.6, system_design: 0.4 },
        competency_weights: { technical_accuracy: 0.6 },
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
        ],
      },
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
    ],
    ...overrides,
  };
}

function renderPage() {
  return render(
    <MemoryRouter>
      <Dashboard />
    </MemoryRouter>,
  );
}

describe("Dashboard interview readiness", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchTopics).mockResolvedValue(topics);
    vi.mocked(fetchTopicMastery).mockResolvedValue({ topics: [] });
    vi.mocked(fetchInterviewStats).mockResolvedValue({
      total_sessions: 1,
      completed_sessions: 1,
      interview_readiness_score: 60,
    });
    vi.mocked(fetchForecast).mockResolvedValue(forecast);
    vi.mocked(fetchPlacementStatus).mockResolvedValue({
      completed: true,
      completed_at: "2026-10-10T00:00:00Z",
      track: "backend",
      derived_level: "mid",
      steps_taken: 10,
    });
    vi.mocked(fetchLearnerProfile).mockResolvedValue(profile);
    vi.mocked(fetchRecommendations).mockResolvedValue({
      generated_at: "2026-10-11T09:00:00Z",
      profile,
      days_until_target: null,
      items: [],
    });
    vi.mocked(fetchGamificationSummary).mockResolvedValue(gamification);
    vi.mocked(fetchReadiness).mockResolvedValue(readiness());
    useAuthStore.setState({
      user: { user: "octocat", provider: "github", is_admin: false },
      isAuthenticated: true,
      isLoading: false,
    });
  });

  afterEach(() => {
    useAuthStore.setState({
      user: null,
      isAuthenticated: false,
      isLoading: true,
    });
  });

  it("shows readiness for the target role and each target company", async () => {
    renderPage();

    expect(
      await screen.findByRole("heading", { name: /Interview readiness/i }),
    ).toBeInTheDocument();
    const role = within(await screen.findByTestId("readiness-role"));
    expect(role.getByText("Target role")).toBeInTheDocument();
    expect(role.getByText("Backend Engineer")).toBeInTheDocument();
    expect(role.getByText("72%")).toBeInTheDocument();
    expect(role.getByText("On Track")).toBeInTheDocument();
    expect(role.getByText("Built-in role model")).toBeInTheDocument();

    const companies = await screen.findAllByTestId("readiness-company");
    expect(companies).toHaveLength(2);
    expect(companies[0]).toHaveTextContent("Datadog");
    expect(within(companies[0]).getByText("83%")).toBeInTheDocument();
    expect(companies[0]).toHaveTextContent("Strong");
    expect(companies[0]).toHaveTextContent("Clears the bar");

    expect(companies[1]).toHaveTextContent("Stripe");
    expect(within(companies[1]).getByText("61%")).toBeInTheDocument();
    expect(companies[1]).toHaveTextContent("Developing");
    expect(companies[1]).toHaveTextContent("Below the bar");
    expect(companies[1]).toHaveTextContent("bar 70%");
    expect(companies[1]).toHaveTextContent("Built-in company profile");
  });

  it("ranks target companies by fit and cites the gaps that remain", async () => {
    renderPage();

    const companies = await screen.findAllByTestId("readiness-company");
    expect(companies[0]).toHaveTextContent("Datadog");
    expect(companies[1]).toHaveTextContent("Stripe");

    const stripe = within(companies[1]);
    expect(stripe.getByText("Top gaps")).toBeInTheDocument();
    expect(stripe.getByText("Backend")).toBeInTheDocument();
    expect(stripe.getByText("· +22.8 pts headroom")).toBeInTheDocument();
    expect(stripe.getByText("· Indexing And Query Plans (41%)")).toBeInTheDocument();

    const role = within(await screen.findByTestId("readiness-role"));
    expect(role.getByText("Technical Accuracy")).toBeInTheDocument();
    expect(role.getByText("· +9.6 pts headroom")).toBeInTheDocument();
  });

  it("links to the study plan and the full readiness breakdown", async () => {
    renderPage();

    await screen.findByRole("heading", { name: /Interview readiness/i });
    expect(screen.getByRole("link", { name: /Study plan/i })).toHaveAttribute(
      "href",
      "/study-plan",
    );
    expect(
      screen.getByRole("link", { name: /Full breakdown/i }),
    ).toHaveAttribute("href", "/readiness");
  });

  it("asks for targets instead of a score when no role or company is saved", async () => {
    vi.mocked(fetchReadiness).mockResolvedValue(
      readiness({
        profile: { ...profile, target_role: "", target_companies: [] },
        role: {
          ...readiness().role,
          target_role: "",
          role_label: "",
          readiness: 0,
          readiness_label: "Not Started",
        },
        companies: [],
      }),
    );
    renderPage();

    expect(
      await screen.findByText(/Set your interview targets/i),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /Update targets/i }),
    ).toHaveAttribute("href", "/study-plan");
    expect(screen.queryByText("Datadog")).not.toBeInTheDocument();
  });

  it("links a new learner to onboarding when placement is not done", async () => {
    vi.mocked(fetchPlacementStatus).mockResolvedValue({
      completed: false,
      completed_at: "",
      track: "backend",
      derived_level: "junior",
      steps_taken: 0,
    });
    vi.mocked(fetchReadiness).mockResolvedValue(
      readiness({
        profile: { ...profile, target_role: "", target_companies: [] },
        role: {
          ...readiness().role,
          target_role: "",
          role_label: "",
          readiness: 0,
          readiness_label: "Not Started",
        },
        companies: [],
      }),
    );
    renderPage();

    expect(
      await screen.findByRole("link", { name: /Set targets/i }),
    ).toHaveAttribute("href", "/onboarding");
  });

  it("reports a failed readiness load without breaking the page", async () => {
    vi.mocked(fetchReadiness).mockRejectedValue(new Error("offline"));
    renderPage();

    expect(
      await screen.findByText(/Role readiness is unavailable right now/i),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("heading", { name: /Interview Topics/i }),
    ).toBeInTheDocument();
  });

  it("hides the section for visitors who are not signed in", async () => {
    useAuthStore.setState({
      user: null,
      isAuthenticated: false,
      isLoading: false,
    });
    renderPage();

    expect(
      await screen.findByRole("heading", { name: /Interview Topics/i }),
    ).toBeInTheDocument();
    await waitFor(() => expect(fetchReadiness).not.toHaveBeenCalled());
    expect(
      screen.queryByRole("heading", { name: /Interview readiness/i }),
    ).not.toBeInTheDocument();
  });

  it("refetches readiness when the page reloads with a fresh auth state", async () => {
    useProgressStore.setState({ completedTopics: [], masteryByTopic: {} });
    renderPage();

    await screen.findByRole("heading", { name: /Interview readiness/i });
    expect(fetchReadiness).toHaveBeenCalledTimes(1);
  });
});
