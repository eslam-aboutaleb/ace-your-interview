import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ProgressState {
  completedTopics: string[];
  answeredQuestions: Record<string, number>;
  masteryByTopic: Record<string, number>;
  topicCount: number;
  setTopicCount: (count: number) => void;
  markTopicComplete: (topicId: string) => void;
  addAnswered: (topicId: string, count: number) => void;
  recordAttempt: (topicId: string, isCorrect: boolean, confidence: number) => void;
  setMastery: (topicId: string, mastery: number) => void;
  getTopicProgress: (topicId: string) => number;
  totalProgress: () => number;
  reset: () => void;
}

export const useProgressStore = create<ProgressState>()(
  persist(
    (set, get) => ({
      completedTopics: [],
      answeredQuestions: {},
      masteryByTopic: {},
      topicCount: 0,

      setTopicCount: (count) => set({ topicCount: Math.max(0, count) }),

      markTopicComplete: (topicId) =>
        set((s) => ({
          completedTopics: s.completedTopics.includes(topicId)
            ? s.completedTopics
            : [...s.completedTopics, topicId],
        })),

      addAnswered: (topicId, count) =>
        set((s) => ({
          answeredQuestions: {
            ...s.answeredQuestions,
              [topicId]: (s.answeredQuestions[topicId] || 0) + count,
            },
          })),

      recordAttempt: (topicId, isCorrect, confidence) =>
        set((s) => {
          const nextAttempts = (s.answeredQuestions[topicId] || 0) + 1;
          const oldMastery = s.masteryByTopic[topicId] ?? 0.3;
          const clampedConfidence = Math.max(1, Math.min(confidence, 5));
          const confidenceNorm = clampedConfidence / 5;
          const calibrated = isCorrect ? confidenceNorm : 1 - confidenceNorm;
          const eventScore = (isCorrect ? 0.7 : 0) + 0.3 * calibrated;
          const nextMastery = Math.max(
            0,
            Math.min(1, 0.7 * oldMastery + 0.3 * eventScore),
          );

          const nextCompleted =
            nextMastery >= 0.8 && nextAttempts >= 3
              ? s.completedTopics.includes(topicId)
                ? s.completedTopics
                : [...s.completedTopics, topicId]
              : s.completedTopics;

          return {
            answeredQuestions: {
              ...s.answeredQuestions,
              [topicId]: nextAttempts,
            },
            masteryByTopic: {
              ...s.masteryByTopic,
              [topicId]: nextMastery,
            },
            completedTopics: nextCompleted,
          };
        }),

      setMastery: (topicId, mastery) =>
        set((s) => ({
          masteryByTopic: {
            ...s.masteryByTopic,
            [topicId]: Math.max(0, Math.min(1, mastery)),
          },
        })),

      getTopicProgress: (topicId) => {
        const s = get();
        const mastery = s.masteryByTopic[topicId];
        if (typeof mastery === "number") {
          return Math.round(mastery * 100);
        }
        return s.completedTopics.includes(topicId)
          ? 100
          : Math.min(90, (s.answeredQuestions[topicId] || 0) * 8);
      },

      totalProgress: () => {
        const s = get();
        const total = s.topicCount || Object.keys(s.masteryByTopic).length || 1;
        let masterySum = 0;
        for (const [topicId, mastery] of Object.entries(s.masteryByTopic)) {
          masterySum += Math.max(0, Math.min(1, mastery));
          if (s.completedTopics.includes(topicId) && mastery < 1) {
            masterySum += 1 - mastery;
          }
        }
        return Math.round((masterySum / total) * 100);
      },

      reset: () =>
        set({
          completedTopics: [],
          answeredQuestions: {},
          masteryByTopic: {},
          topicCount: 0,
        }),
    }),
    { name: "study-hub-progress" },
  ),
);
