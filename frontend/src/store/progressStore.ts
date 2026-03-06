import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ProgressState {
  completedTopics: string[];
  answeredQuestions: Record<string, number>;
  markTopicComplete: (topicId: string) => void;
  addAnswered: (topicId: string, count: number) => void;
  getTopicProgress: (topicId: string) => number;
  totalProgress: () => number;
  reset: () => void;
}

export const useProgressStore = create<ProgressState>()(
  persist(
    (set, get) => ({
      completedTopics: [],
      answeredQuestions: {},

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

      getTopicProgress: (topicId) => {
        const s = get();
        return s.completedTopics.includes(topicId)
          ? 100
          : Math.min(95, (s.answeredQuestions[topicId] || 0) * 10);
      },

      totalProgress: () => {
        const s = get();
        return Math.round((s.completedTopics.length / 14) * 100); // 14 topics total
      },

      reset: () => set({ completedTopics: [], answeredQuestions: {} }),
    }),
    { name: "study-hub-progress" },
  ),
);
