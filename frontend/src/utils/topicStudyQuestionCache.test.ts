import { describe, expect, it } from "vitest";
import {
  loadTopicStudyQuestionCache,
  makeTopicStudyQuestionCacheStorageKey,
  persistTopicStudyQuestionCache,
} from "./topicStudyQuestionCache";
import type { TopicStudyQuestionCacheEntry } from "./topicStudyQuestionCache";

const entry: TopicStudyQuestionCacheEntry = {
  key: "sig-auto",
  signature: {
    topicId: "01-http",
    sectionIndex: 0,
    sectionFingerprint: "fp",
    questionCount: 5,
    difficulty: "",
    level: "auto",
    responseDetail: "very_detailed",
    preferredLanguage: "",
    requiresProgramming: false,
    llmProvider: "openai",
    llmModel: "gpt-4o-mini",
    llmTemperature: 0,
    llmMaxTokens: 0,
  },
  status: "ready",
  questions: [],
  providerInfo: { provider: "openai", model: "gpt-4o-mini", retries: 0, malformed: 0 },
  errorMsg: "",
  startedAt: 1,
  completedAt: 2,
  updatedAt: 3,
};

describe("topic study question cache", () => {
  it("keeps the auto level through a persist/load round trip", () => {
    window.sessionStorage.clear();
    persistTopicStudyQuestionCache("01-http", { [entry.key]: entry });

    const loaded = loadTopicStudyQuestionCache("01-http");
    expect(Object.keys(loaded)).toEqual([entry.key]);
    expect(loaded[entry.key].signature.level).toBe("auto");
  });

  it("falls back to mid for an unreadable level", () => {
    window.sessionStorage.clear();
    const raw = JSON.stringify({
      version: 4,
      entries: [
        {
          ...entry,
          signature: { ...entry.signature, level: "expert" },
        },
      ],
    });
    window.sessionStorage.setItem(
      makeTopicStudyQuestionCacheStorageKey("01-http"),
      raw,
    );

    const loaded = loadTopicStudyQuestionCache("01-http");
    expect(loaded[entry.key].signature.level).toBe("mid");
  });
});
