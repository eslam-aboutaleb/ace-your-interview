import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AnalyticsPage from "./AnalyticsPage";
import {
  fetchCalibration,
  fetchHeatmap,
  fetchMasteryCurve,
  fetchRetention,
} from "@/services/api";
import type {
  CalibrationResponse,
  HeatmapResponse,
  MasteryCurveResponse,
  RetentionResponse,
} from "@/types";

vi.mock("@/services/api", () => ({
  fetchHeatmap: vi.fn(),
  fetchRetention: vi.fn(),
  fetchMasteryCurve: vi.fn(),
  fetchCalibration: vi.fn(),
}));

const heatmap: HeatmapResponse = {
  weeks: 26,
  days: [
    { date: "2026-10-01", count: 3, types: { attempt: 2, review: 1 } },
    { date: "2026-10-02", count: 1, types: { attempt: 1 } },
  ],
};

const retention: RetentionResponse = {
  buckets: [
    { bucket: "1d", reviews: 4, successful: 3, recall_rate: 0.75 },
    { bucket: "7d", reviews: 3, successful: 2, recall_rate: 0.6667 },
    { bucket: "30d", reviews: 2, successful: 2, recall_rate: 1 },
  ],
};

const mastery: MasteryCurveResponse = {
  points: [
    { date: "2026-10-01", mastery: 0.4 },
    { date: "2026-10-02", mastery: 0.6 },
  ],
  topics: [],
};

const calibration: CalibrationResponse = {
  eligible_reviews: 40,
  successful_reviews: 29,
  true_retention: 0.725,
  target_band_low: 0.8,
  target_band_high: 0.9,
  within_band: false,
  good_rating_pct: 0.55,
  low_signal: true,
};

function renderPage() {
  return render(
    <AnalyticsPage />,
  );
}

describe("AnalyticsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchHeatmap).mockResolvedValue(heatmap);
    vi.mocked(fetchRetention).mockResolvedValue(retention);
    vi.mocked(fetchMasteryCurve).mockResolvedValue(mastery);
    vi.mocked(fetchCalibration).mockResolvedValue(calibration);
  });

  it("renders the calibration section from the backend payload", async () => {
    renderPage();

    expect(
      await screen.findByRole("heading", { name: /Confidence calibration/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("True retention")).toBeInTheDocument();
    expect(screen.getByText("73%")).toBeInTheDocument();
    expect(screen.getByText("80–90%")).toBeInTheDocument();
    expect(screen.getByText("Good rating share")).toBeInTheDocument();
    expect(screen.getByText("55%")).toBeInTheDocument();
    expect(screen.getByText("29 successful")).toBeInTheDocument();
    expect(
      screen.getByText(/Below the target band/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Low signal: fewer than 50 eligible reviews/i),
    ).toBeInTheDocument();
  });

  it("renders the per-event-type breakdown with totals", async () => {
    renderPage();

    const attempts = await screen.findByRole("button", { name: /Attempts/ });
    expect(attempts).toHaveTextContent("Attempts");
    expect(attempts).toHaveTextContent("3");
    expect(screen.getByRole("button", { name: /Reviews/ })).toHaveTextContent(
      "1",
    );
    expect(screen.getByRole("button", { name: /Exams/ })).toHaveTextContent("0");
  });

  it("toggles the heatmap filter when a type chip is clicked", async () => {
    renderPage();

    const attempts = await screen.findByRole("button", { name: /Attempts/ });
    const reviews = screen.getByRole("button", { name: /Reviews/ });

    expect(attempts).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(reviews);
    expect(reviews).toHaveAttribute("aria-pressed", "true");
    expect(attempts).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(reviews);
    expect(reviews).toHaveAttribute("aria-pressed", "false");
  });

  it("shows the calibration empty state without eligible reviews", async () => {
    vi.mocked(fetchCalibration).mockResolvedValue({
      ...calibration,
      eligible_reviews: 0,
      successful_reviews: 0,
      true_retention: 0,
      good_rating_pct: 0,
      within_band: false,
      low_signal: true,
    });

    renderPage();

    expect(
      await screen.findByText(
        /Complete some spaced reviews to build the calibration estimate/i,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText("True retention")).not.toBeInTheDocument();
  });
});
