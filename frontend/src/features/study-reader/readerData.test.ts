import { beforeEach, describe, expect, it, vi } from "vitest";
import { fetchJourney, fetchReaderState } from "@/services/api";
import type { ReaderCheckpoint } from "@/types";
import {
  getReaderData,
  invalidateReaderData,
  updateCachedCheckpoint,
} from "./readerData";

vi.mock("@/services/api", () => ({
  fetchJourney: vi.fn(),
  fetchReaderState: vi.fn(),
}));

const journey = { id: "j1", title: "Journey" };
const readerState = { journey_id: "j1", checkpoint: { revision: 1 } };

describe("readerData cache", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    invalidateReaderData();
    // Reset the cache owner by reading as a fresh user.
    vi.mocked(fetchJourney).mockResolvedValue(journey as never);
    vi.mocked(fetchReaderState).mockResolvedValue(readerState as never);
  });

  it("fetches once and serves the cache within the TTL", async () => {
    await getReaderData("j1", "alice");
    await getReaderData("j1", "alice");
    expect(fetchJourney).toHaveBeenCalledTimes(1);
    expect(fetchReaderState).toHaveBeenCalledTimes(1);
  });

  it("deduplicates in-flight requests", async () => {
    let release!: (value: unknown) => void;
    vi.mocked(fetchJourney).mockReturnValue(
      new Promise((resolve) => {
        release = resolve;
      }) as never,
    );
    const first = getReaderData("j1", "alice");
    const second = getReaderData("j1", "alice");
    release(journey);
    await Promise.all([first, second]);
    expect(fetchJourney).toHaveBeenCalledTimes(1);
  });

  it("clears the cache when the authenticated user changes", async () => {
    await getReaderData("j1", "alice");
    await getReaderData("j1", "bob");
    expect(fetchJourney).toHaveBeenCalledTimes(2);
  });

  it("refetches after invalidation", async () => {
    await getReaderData("j1", "alice");
    invalidateReaderData("j1");
    await getReaderData("j1", "alice");
    expect(fetchJourney).toHaveBeenCalledTimes(2);
  });

  it("applies checkpoint updates to the cached reader state", async () => {
    await getReaderData("j1", "alice");
    updateCachedCheckpoint("j1", { revision: 7 } as unknown as ReaderCheckpoint);
    const data = await getReaderData("j1", "alice");
    expect(data.readerState.checkpoint.revision).toBe(7);
    expect(fetchReaderState).toHaveBeenCalledTimes(1); // served from cache
  });
});
