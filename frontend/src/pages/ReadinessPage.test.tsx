import { render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ReadinessPage from "./ReadinessPage";
import { fetchReadiness } from "@/services/api";
import { useAuthStore } from "@/store/authStore";
import type { RoleReadinessResponse } from "@/types";

vi.mock("@/services/api", () => ({
  fetchReadiness: vi.fn(),
}));

function readiness(
  overrides: Partial<RoleReadinessResponse> = {},
): RoleReadinessResponse {
  return {
    generated_at: "2026-10-11T09:00:00Z",
    profile: {
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
    },
    role: {
      target_role: "Backend Engineer",
      role_id: "backend_engineer",
      role_label: "Backend Engineer",
      match_source: "catalog",
      readiness: 72.4,
      readiness_label: "On Track",
      track_weights: { backend: 0.6, system_design: 0.4 },
      competency_weights: { technical_accuracy: 0.6, communication_clarity: 0.4 },
      track_breakdown: [
        {
          kind: "track",
          id: "backend",
          label: "Backend",
          weight: 0.6,
          score: 62,
          evidence_count: 12,
          contribution: 37.2,
          gap_points: 22.8,
        },
        {
          kind: "track",
          id: "system_design",
          label: "System Design",
          weight: 0.4,
          score: 88,
          evidence_count: 4,
          contribution: 35.2,
          gap_points: 4.8,
        },
      ],
      competency_breakdown: [
        {
          kind: "interview_competency",
          id: "technical_accuracy",
          label: "Technical Accuracy",
          weight: 0.6,
          score: 70,
          evidence_count: 6,
          contribution: 42,
          gap_points: 18,
        },
      ],
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
            {
              kind: "error_pattern",
              ref: "off_by_one",
              label: "Off By One in Indexing And Query Plans",
              score: null,
              count: 3,
              detail: "Boundary handling",
            },
          ],
        },
        {
          kind: "interview_competency",
          id: "communication_clarity",
          label: "Communication Clarity",
          weight: 0.4,
          score: 55,
          evidence_count: 2,
          gap_points: 9.6,
          evidence: [
            {
              kind: "interview_competency",
              ref: "communication_clarity",
              label: "Communication Clarity",
              score: 55,
              count: 0,
              detail: "2 review(s), 1 lapse(s)",
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
      {
        role_id: "fullstack_engineer",
        role_label: "Full-Stack Engineer",
        readiness: 51.2,
        readiness_label: "Developing",
        is_target: false,
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
        track_weights: { backend: 0.5, system_design: 0.5 },
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
      <ReadinessPage />
    </MemoryRouter>,
  );
}

function apiError(message: string) {
  return {
    response: { status: 404, data: { detail: message } },
    message,
  };
}

describe("ReadinessPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAuthStore.setState({
      user: { user: "octocat", provider: "github", is_admin: false },
      isAuthenticated: true,
      isLoading: false,
    });
  });

  afterEach(() => {
    useAuthStore.setState({ user: null, isAuthenticated: false, isLoading: true });
  });

  it("renders the target-role readiness, band and evidence counters", async () => {
    vi.mocked(fetchReadiness).mockResolvedValue(readiness());
    renderPage();

    const card = await screen.findByTestId("target-role");
    expect(
      within(card).getByRole("heading", { name: "Backend Engineer" }),
    ).toBeInTheDocument();
    expect(card).toHaveTextContent("Your target role: Backend Engineer");
    expect(card).toHaveTextContent("Built-in role model");
    expect(within(card).getByText("72%")).toBeInTheDocument();
    expect(within(card).getByText("On Track")).toBeInTheDocument();
    expect(within(card).getByText("Topic reviews")).toBeInTheDocument();
    expect(within(card).getByText("12")).toBeInTheDocument();
    expect(within(card).getByText("Interview reports")).toBeInTheDocument();
    expect(within(card).getByText("Competency reviews")).toBeInTheDocument();
    expect(within(card).getByText("Error patterns")).toBeInTheDocument();
    expect(within(card).getByText("5")).toBeInTheDocument();
  });

  it("renders the per-track and per-competency breakdown", async () => {
    vi.mocked(fetchReadiness).mockResolvedValue(readiness());
    renderPage();

    const tracks = await screen.findByTestId("track-breakdown");
    const competencies = screen.getByTestId("competency-breakdown");

    const backendRow = within(tracks).getByText("Backend").closest("tr");
    expect(backendRow).toHaveTextContent("60%");
    expect(backendRow).toHaveTextContent("62%");
    expect(backendRow).toHaveTextContent("37.2 pts");
    expect(backendRow).toHaveTextContent("+22.8 pts");
    expect(backendRow).toHaveTextContent("evidence 12");

    const competencyRow =
      within(competencies).getByText("Technical Accuracy").closest("tr");
    expect(competencyRow).toHaveTextContent("70%");
    expect(competencyRow).toHaveTextContent("42.0 pts");
    expect(competencyRow).toHaveTextContent("+18.0 pts");
  });

  it("renders gap dimensions with their topic, error-pattern and competency evidence", async () => {
    vi.mocked(fetchReadiness).mockResolvedValue(readiness());
    renderPage();

    const gaps = await screen.findByTestId("gap-dimensions");
    const [trackGap, competencyGap] = gaps.querySelectorAll(":scope > div");

    expect(trackGap).toHaveTextContent("Backend");
    expect(trackGap).toHaveTextContent("Track");
    expect(trackGap).toHaveTextContent("60% of the model");
    expect(trackGap).toHaveTextContent("score 62%");
    expect(trackGap).toHaveTextContent("1 topic · 1 error pattern");
    expect(trackGap).toHaveTextContent("+22.8 pts");
    expect(trackGap).toHaveTextContent("Indexing And Query Plans");
    expect(trackGap).toHaveTextContent("41%");
    expect(trackGap).toHaveTextContent("4 review(s), 2 due");
    expect(
      trackGap?.textContent?.includes(
        "Off By One in Indexing And Query Plans",
      ),
    ).toBe(true);
    expect(trackGap).toHaveTextContent("3×");
    expect(trackGap).toHaveTextContent("Boundary handling");

    expect(competencyGap).toHaveTextContent("Communication Clarity");
    expect(competencyGap).toHaveTextContent("Interview competency");
    expect(competencyGap).toHaveTextContent("1 interview competency");
    expect(competencyGap).toHaveTextContent("2 review(s), 1 lapse(s)");
  });

  it("ranks best-fit roles and marks the target role", async () => {
    vi.mocked(fetchReadiness).mockResolvedValue(readiness());
    renderPage();

    const list = await screen.findByTestId("best-fit-roles");
    const items = list.querySelectorAll("li");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("Backend Engineer");
    expect(items[0]).toHaveTextContent("Target role");
    expect(items[0]).toHaveTextContent("72%");
    expect(items[1]).toHaveTextContent("Full-Stack Engineer");
    expect(items[1]).not.toHaveTextContent("Target role");
    expect(items[1]).toHaveTextContent("Developing");
  });

  it("renders ranked company fits with bar, verdict and gap evidence", async () => {
    vi.mocked(fetchReadiness).mockResolvedValue(readiness());
    renderPage();

    const list = await screen.findByTestId("company-fits");
    const cards = list.querySelectorAll("article");
    expect(cards).toHaveLength(2);

    expect(within(cards[0] as HTMLElement).getByText("Datadog")).toBeInTheDocument();
    expect(cards[0]).toHaveTextContent("83%");
    expect(cards[0]).toHaveTextContent("Strong");
    expect(cards[0]).toHaveTextContent("Clears the bar");
    expect(cards[0]).toHaveTextContent("bar 74%");
    expect(cards[0]).toHaveTextContent("Role-inherited default");

    expect(within(cards[1] as HTMLElement).getByText("Stripe")).toBeInTheDocument();
    expect(cards[1]).toHaveTextContent("61%");
    expect(cards[1]).toHaveTextContent("Below the bar");
    expect(cards[1]).toHaveTextContent("Built-in company profile");
    expect(cards[1]).toHaveTextContent("Indexing And Query Plans");
  });

  it("guides the learner to set target companies when none are saved", async () => {
    vi.mocked(fetchReadiness).mockResolvedValue(
      readiness({
        companies: [],
        profile: {
          ...readiness().profile,
          target_companies: [],
        },
      }),
    );
    renderPage();

    expect(
      await screen.findByText(/Add target companies to your learner profile/i),
    ).toBeInTheDocument();
    const link = screen.getByRole("link", { name: /Update target companies/i });
    expect(link).toHaveAttribute("href", "/study-plan");
    expect(screen.queryByText("Stripe")).not.toBeInTheDocument();
  });

  it("reports saved target companies that produced no scoreable entry", async () => {
    vi.mocked(fetchReadiness).mockResolvedValue(
      readiness({
        companies: [],
        profile: {
          ...readiness().profile,
          target_companies: ["Stripe"],
        },
      }),
    );
    renderPage();

    expect(
      await screen.findByText(/No target companies to score yet/i),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: /Update target companies/i }),
    ).not.toBeInTheDocument();
  });

  it("shows the disabled state when adaptive learning is off", async () => {
    vi.mocked(fetchReadiness).mockRejectedValue(
      apiError("Adaptive learning disabled"),
    );
    renderPage();

    expect(
      await screen.findByText(/Adaptive learning is turned off/i),
    ).toBeInTheDocument();
    expect(screen.queryByText("Best-fit roles")).not.toBeInTheDocument();
  });

  it("surfaces other load failures as an alert", async () => {
    vi.mocked(fetchReadiness).mockRejectedValue(apiError("Learning store down"));
    renderPage();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/Learning store down/i);
    expect(
      screen.queryByText(/Adaptive learning is turned off/i),
    ).not.toBeInTheDocument();
  });

  it("asks an unauthenticated visitor to sign in", async () => {
    useAuthStore.setState({
      user: null,
      isAuthenticated: false,
      isLoading: false,
    });
    vi.mocked(fetchReadiness).mockResolvedValue(readiness());
    renderPage();

    expect(await screen.findByText("Sign in to continue")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /Go to sign in/i }),
    ).toHaveAttribute("href", "/login");
    await waitFor(() => expect(fetchReadiness).not.toHaveBeenCalled());
  });

  it("renders a no-gaps state when nothing holds the score back", async () => {
    const base = readiness();
    vi.mocked(fetchReadiness).mockResolvedValue({
      ...base,
      role: {
        ...base.role,
        readiness: 91,
        readiness_label: "Strong",
        gap_dimensions: [],
      },
    });
    renderPage();

    const card = await screen.findByTestId("target-role");
    expect(
      await screen.findByText(/No weighted gaps left in this role model/i),
    ).toBeInTheDocument();
    expect(within(card).getByText("Strong")).toBeInTheDocument();
    expect(within(card).getByText("91%")).toBeInTheDocument();
    expect(screen.queryByTestId("gap-dimensions")).not.toBeInTheDocument();
  });

  it("refetches when refresh is pressed", async () => {
    vi.mocked(fetchReadiness).mockResolvedValue(readiness());
    renderPage();

    await screen.findByText("Role Readiness");
    const button = screen.getByRole("button", { name: /Refresh/i });
    expect(button).toBeEnabled();

    const second = readiness({
      generated_at: "2026-10-11T10:00:00Z",
    });
    vi.mocked(fetchReadiness).mockResolvedValue(second);
    button.click();
    await waitFor(() => expect(fetchReadiness).toHaveBeenCalledTimes(2));
  });
});
