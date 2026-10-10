/**
 * Zustand store for quiz session state.
 *
 * Centralizes the quiz phase, questions, current
 * question + answer contract, degraded flags, and
 * generation/submission state that previously lived
 * in QuizMode local useState hooks. Purely local UI
 * state (setup form config, modals, per-question
 * hint UI) stays in the component.
 *
 * Not persisted — quiz session state is transient
 * per page visit (the page resets the store on
 * unmount).
 */

import { create } from "zustand";
import type { QuizQuestionV2 } from "../types";

export type QuizPhase = "setup" | "playing" | "results";

export interface QuizProviderInfo {
  provider: string;
  model: string;
  retries: number;
  malformed: number;
}

export interface QuizAnswerRecord {
  selected: string;
  correct: boolean;
  confidence: number;
}

export interface DegradedQuizInfo {
  requested: number;
  returned: number;
}

interface QuizSessionState {
  /* ── Session ── */
  phase: QuizPhase;
  questions: QuizQuestionV2[];
  currentIdx: number;
  selectedAnswer: string | null;
  answered: boolean;
  answerConfidence: number;
  answers: Record<number, QuizAnswerRecord>;
  questionStartMs: Record<number, number>;
  providerInfo: QuizProviderInfo;

  /* ── Generation ── */
  generating: boolean;
  generationProgress: string;
  generatedPreviewCount: number;
  degradedQuiz: DegradedQuizInfo | null;
  errorMsg: string;

  /* ── Exam ── */
  examFinalized: boolean;
  remainingSeconds: number | null;

  /* ── Actions ── */
  setPhase: (phase: QuizPhase) => void;
  setQuestions: (questions: QuizQuestionV2[]) => void;
  setCurrentIdx: (currentIdx: number) => void;
  setSelectedAnswer: (selectedAnswer: string | null) => void;
  setAnswered: (answered: boolean) => void;
  setAnswerConfidence: (answerConfidence: number) => void;
  recordAnswer: (
    idx: number,
    answer: QuizAnswerRecord,
  ) => void;
  setQuestionStartMs: (starts: Record<number, number>) => void;
  touchQuestionStart: (idx: number) => void;
  setProviderInfo: (providerInfo: QuizProviderInfo) => void;
  setGenerating: (generating: boolean) => void;
  setGenerationProgress: (progress: string) => void;
  setGeneratedPreviewCount: (count: number) => void;
  setDegradedQuiz: (degraded: DegradedQuizInfo | null) => void;
  setErrorMsg: (errorMsg: string) => void;
  setExamFinalized: (examFinalized: boolean) => void;
  setRemainingSeconds: (
    remainingSeconds:
      | number
      | null
      | ((prev: number | null) => number | null),
  ) => void;
  beginPlaying: () => void;
  resetToSetup: () => void;
  reset: () => void;
}

const INITIAL_STATE = {
  phase: "setup" as QuizPhase,
  questions: [] as QuizQuestionV2[],
  currentIdx: 0,
  selectedAnswer: null as string | null,
  answered: false,
  answerConfidence: 3,
  answers: {} as Record<number, QuizAnswerRecord>,
  questionStartMs: {} as Record<number, number>,
  providerInfo: {
    provider: "",
    model: "",
    retries: 0,
    malformed: 0,
  },
  generating: false,
  generationProgress: "",
  generatedPreviewCount: 0,
  degradedQuiz: null,
  errorMsg: "",
  examFinalized: false,
  remainingSeconds: null as number | null,
};

export const useQuizStore = create<QuizSessionState>()(
  (set) => ({
    ...INITIAL_STATE,

    setPhase: (phase) => set({ phase }),
    setQuestions: (questions) => set({ questions }),
    setCurrentIdx: (currentIdx) => set({ currentIdx }),
    setSelectedAnswer: (selectedAnswer) => set({ selectedAnswer }),
    setAnswered: (answered) => set({ answered }),
    setAnswerConfidence: (answerConfidence) => set({ answerConfidence }),

    recordAnswer: (idx, answer) =>
      set((s) => ({
        answers: { ...s.answers, [idx]: answer },
      })),

    setQuestionStartMs: (starts) => set({ questionStartMs: starts }),

    touchQuestionStart: (idx) =>
      set((s) =>
        s.questionStartMs[idx]
          ? s
          : {
              questionStartMs: {
                ...s.questionStartMs,
                [idx]: Date.now(),
              },
            },
      ),

    setProviderInfo: (providerInfo) => set({ providerInfo }),
    setGenerating: (generating) => set({ generating }),
    setGenerationProgress: (generationProgress) =>
      set({ generationProgress }),
    setGeneratedPreviewCount: (generatedPreviewCount) =>
      set({ generatedPreviewCount }),
    setDegradedQuiz: (degradedQuiz) => set({ degradedQuiz }),
    setErrorMsg: (errorMsg) => set({ errorMsg }),
    setExamFinalized: (examFinalized) => set({ examFinalized }),
    setRemainingSeconds: (remainingSeconds) =>
      set((s) => ({
        remainingSeconds:
          typeof remainingSeconds === "function"
            ? (
                remainingSeconds as (
                  prev: number | null,
                ) => number | null
              )(s.remainingSeconds)
            : remainingSeconds,
      })),

    beginPlaying: () =>
      set({
        phase: "playing",
        currentIdx: 0,
        answers: {},
        selectedAnswer: null,
        answered: false,
        answerConfidence: 3,
        questionStartMs: { 0: Date.now() },
        examFinalized: false,
        remainingSeconds: null,
      }),

    resetToSetup: () =>
      set((s) => ({
        ...INITIAL_STATE,
        phase: "setup",
        providerInfo: s.providerInfo,
      })),

    reset: () => set({ ...INITIAL_STATE }),
  }),
);
