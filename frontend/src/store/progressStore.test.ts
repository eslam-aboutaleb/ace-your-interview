import { beforeEach, describe, expect, it } from "vitest";
import { useProgressStore } from "./progressStore";
import type { TopicMasteryItem, TopicMasteryResponse } from "@/types";

function serverResponse(topics: TopicMasteryItem[]): TopicMasteryResponse {
  return { topics };
}

describe("progressStore", () => {
  beforeEach(() => {
    useProgressStore.setState({
      completedTopics: [],
      answeredQuestions: {},
      masteryByTopic: {},
      topicCount: 0,
    });
  });

  it("recordAttempt keeps echoing local writes into the EWMA mirror", () => {
    useProgressStore.getState().recordAttempt("01-http", true, 4);
    useProgressStore.getState().recordAttempt("01-http", true, 5);

    const state = useProgressStore.getState();
    expect(state.masteryByTopic["01-http"]).toBeGreaterThan(0.3);
    expect(state.answeredQuestions["01-http"]).toBe(2);
  });

  it("hydrateMasteryFromServer overrides a stale local mirror", () => {
    useProgressStore.getState().recordAttempt("01-http", false, 1);
    const stale = useProgressStore.getState().masteryByTopic["01-http"];

    useProgressStore.getState().hydrateMasteryFromServer(
      serverResponse([
        {
          topic_id: "01-http",
          mastery_score: 0.87,
          attempts: 3,
          accuracy: 0.67,
          due_count: 1,
        },
      ]),
    );

    expect(useProgressStore.getState().masteryByTopic["01-http"]).toBe(0.87);
    expect(stale).not.toBe(0.87);
  });

  it("server hydration keeps local-only topics as the optimistic cache", () => {
    useProgressStore.getState().recordAttempt("offline-topic", true, 5);

    useProgressStore.getState().hydrateMasteryFromServer(
      serverResponse([
        {
          topic_id: "01-http",
          mastery_score: 0.5,
          attempts: 1,
          accuracy: 1,
          due_count: 0,
        },
      ]),
    );

    const mastery = useProgressStore.getState().masteryByTopic;
    expect(mastery["01-http"]).toBe(0.5);
    expect(mastery["offline-topic"]).toBeGreaterThan(0.3);
  });

  it("hydrated mastery drives getTopicProgress", () => {
    useProgressStore.getState().hydrateMasteryFromServer(
      serverResponse([
        {
          topic_id: "01-http",
          mastery_score: 0.42,
          attempts: 4,
          accuracy: 0.5,
          due_count: 0,
        },
      ]),
    );

    expect(useProgressStore.getState().getTopicProgress("01-http")).toBe(42);
  });

  it("reset clears hydrated mastery", () => {
    useProgressStore.getState().hydrateMasteryFromServer(
      serverResponse([
        {
          topic_id: "01-http",
          mastery_score: 0.42,
          attempts: 4,
          accuracy: 0.5,
          due_count: 0,
        },
      ]),
    );

    useProgressStore.getState().reset();

    expect(useProgressStore.getState().masteryByTopic).toEqual({});
  });
});
