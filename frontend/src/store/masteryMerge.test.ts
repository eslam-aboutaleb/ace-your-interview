import { describe, expect, it } from "vitest";
import { mergeServerMastery } from "./masteryMerge";
import type { TopicMasteryItem } from "@/types";

function serverResponse(topics: TopicMasteryItem[]) {
  return { topics };
}

describe("mergeServerMastery", () => {
  it("server FSRS mastery wins over the local EWMA mirror", () => {
    const merged = mergeServerMastery(
      { "01-http": 0.2 },
      serverResponse([
        {
          topic_id: "01-http",
          mastery_score: 0.91,
          attempts: 4,
          accuracy: 1,
          due_count: 0,
        },
      ]),
    );

    expect(merged["01-http"]).toBe(0.91);
  });

  it("keeps local-only topics as the optimistic cache", () => {
    const merged = mergeServerMastery(
      { "01-http": 0.2, "offline-topic": 0.65 },
      serverResponse([
        {
          topic_id: "01-http",
          mastery_score: 0.91,
          attempts: 4,
          accuracy: 1,
          due_count: 0,
        },
      ]),
    );

    expect(merged["offline-topic"]).toBe(0.65);
  });

  it("clamps server values into [0, 1]", () => {
    const merged = mergeServerMastery(
      {},
      serverResponse([
        {
          topic_id: "high",
          mastery_score: 1.4,
          attempts: 0,
          accuracy: 0,
          due_count: 0,
        },
        {
          topic_id: "low",
          mastery_score: -0.2,
          attempts: 0,
          accuracy: 0,
          due_count: 0,
        },
      ]),
    );

    expect(merged.high).toBe(1);
    expect(merged.low).toBe(0);
  });

  it("returns a local copy when the server payload is empty", () => {
    const local = { "01-http": 0.2 };
    const merged = mergeServerMastery(local, null);

    expect(merged).toEqual(local);
    expect(merged).not.toBe(local);
  });

  it("keeps the local value when a server score is not usable", () => {
    const merged = mergeServerMastery(
      { "01-http": 0.2 },
      serverResponse([
        {
          topic_id: "01-http",
          mastery_score: Number.NaN,
          attempts: 0,
          accuracy: 0,
          due_count: 0,
        },
      ]),
    );

    expect(merged["01-http"]).toBe(0.2);
  });

  it("skips items without a topic id", () => {
    const merged = mergeServerMastery(
      { "01-http": 0.2 },
      serverResponse([
        {
          topic_id: "",
          mastery_score: 0.9,
          attempts: 0,
          accuracy: 0,
          due_count: 0,
        },
      ]),
    );

    expect(merged).toEqual({ "01-http": 0.2 });
  });
});
