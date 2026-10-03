import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ReviewQueue from "./ReviewQueue";
import { fetchReviewQueue, fetchTopics } from "@/services/api";
import type { ReviewQueueItem } from "@/types";

vi.mock("@/services/api", () => ({
  fetchReviewQueue: vi.fn(),
  fetchTopics: vi.fn(),
}));

function queueItem(overrides: Partial<ReviewQueueItem> = {}): ReviewQueueItem {
  return {
    topic_id: "custom-java",
    question_id: "q-keep-alive",
    due_at: new Date(Date.now() - 60_000).toISOString(),
    mastery_score: 0.42,
    last_confidence: 3,
    review_bucket: 2,
    attempts: 4,
    ...overrides,
  };
}

describe("ReviewQueue", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders due items with their mastery and bucket badges", async () => {
    vi.mocked(fetchReviewQueue).mockResolvedValue({
      items: [queueItem()],
      total_due: 1,
    });
    vi.mocked(fetchTopics).mockResolvedValue([
      { id: "custom-java", title: "Custom Java" },
    ]);

    render(
      <MemoryRouter>
        <ReviewQueue />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Custom Java")).toBeInTheDocument();
    expect(screen.getByText(/Mastery: 42%/)).toBeInTheDocument();
    expect(screen.getByText(/Bucket: 2/)).toBeInTheDocument();
    expect(screen.getByText(/Attempts: 4/)).toBeInTheDocument();
  });

  it("shows the empty state when nothing is due", async () => {
    vi.mocked(fetchReviewQueue).mockResolvedValue({
      items: [],
      total_due: 0,
    });
    vi.mocked(fetchTopics).mockResolvedValue([]);

    render(
      <MemoryRouter>
        <ReviewQueue />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText("No due items right now."),
    ).toBeInTheDocument();
  });

  it("shows an error state when the queue fails to load", async () => {
    vi.mocked(fetchReviewQueue).mockRejectedValue(new Error("boom"));
    vi.mocked(fetchTopics).mockResolvedValue([]);

    render(
      <MemoryRouter>
        <ReviewQueue />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText("Could not load review queue. Please retry."),
    ).toBeInTheDocument();
  });
});
