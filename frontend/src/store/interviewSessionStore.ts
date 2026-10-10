/**
 * Zustand store for the live mock-interview session.
 *
 * Centralizes the session object, turns, current question,
 * degraded flags, hint state, and submission state that
 * previously lived in InterviewSessionPage local useState
 * hooks. Purely local UI state (the answer draft) stays in
 * the component.
 *
 * Not persisted — interview session state is transient per
 * page visit (the page resets the store on unmount).
 */

import { create } from "zustand";
import type {
  HintResponse,
  InterviewSessionResponse,
  InterviewTurnResponse,
} from "../types";

interface InterviewSessionState {
  /* ── Session ── */
  session: InterviewSessionResponse | null;
  loading: boolean;
  error: string;
  streamProgress: string;
  loadingNext: boolean;
  submitting: boolean;
  startedAt: number;
  lastTurn: InterviewTurnResponse | null;

  /* ── Hints ── */
  hintLevel: number;
  hintResult: HintResponse | null;
  hintLoading: boolean;
  hintError: string;
  hintLevelUsed: number;

  /* ── Actions ── */
  setSession: (session: InterviewSessionResponse | null) => void;
  updateSession: (
    updater: (
      prev: InterviewSessionResponse | null,
    ) => InterviewSessionResponse | null,
  ) => void;
  setCurrentQuestion: (question: string) => void;
  appendTurn: (response: InterviewTurnResponse) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string) => void;
  setStreamProgress: (progress: string) => void;
  setLoadingNext: (loadingNext: boolean) => void;
  setSubmitting: (submitting: boolean) => void;
  setStartedAt: (startedAt: number) => void;
  setLastTurn: (turn: InterviewTurnResponse | null) => void;
  setHintLevel: (level: number) => void;
  setHintResult: (result: HintResponse | null) => void;
  setHintLoading: (loading: boolean) => void;
  setHintError: (error: string) => void;
  setHintLevelUsed: (level: number) => void;
  resetHintState: () => void;
  reset: () => void;
}

const INITIAL_STATE = {
  session: null,
  loading: true,
  error: "",
  streamProgress: "",
  loadingNext: false,
  submitting: false,
  startedAt: Date.now(),
  lastTurn: null,
  hintLevel: 1,
  hintResult: null,
  hintLoading: false,
  hintError: "",
  hintLevelUsed: 0,
};

export const useInterviewSessionStore = create<InterviewSessionState>()(
  (set) => ({
    ...INITIAL_STATE,

    setSession: (session) => set({ session }),

    updateSession: (updater) =>
      set((s) => ({ session: updater(s.session) })),

    setCurrentQuestion: (question) =>
      set((s) => ({
        session: s.session
          ? {
              ...s.session,
              session: { ...s.session.session, current_question: question },
            }
          : s.session,
      })),

    appendTurn: (response) =>
      set((s) => ({
        session: s.session
          ? {
              ...s.session,
              session: response.session,
              turns: [...s.session.turns, response.turn],
            }
          : s.session,
        lastTurn: response,
      })),

    setLoading: (loading) => set({ loading }),
    setError: (error) => set({ error }),
    setStreamProgress: (streamProgress) => set({ streamProgress }),
    setLoadingNext: (loadingNext) => set({ loadingNext }),
    setSubmitting: (submitting) => set({ submitting }),
    setStartedAt: (startedAt) => set({ startedAt }),
    setLastTurn: (lastTurn) => set({ lastTurn }),

    setHintLevel: (hintLevel) => set({ hintLevel }),
    setHintResult: (hintResult) => set({ hintResult }),
    setHintLoading: (hintLoading) => set({ hintLoading }),
    setHintError: (hintError) => set({ hintError }),
    setHintLevelUsed: (hintLevelUsed) => set({ hintLevelUsed }),

    resetHintState: () =>
      set({
        hintResult: null,
        hintError: "",
        hintLevelUsed: 0,
        hintLevel: 1,
      }),

    reset: () => set({ ...INITIAL_STATE, startedAt: Date.now() }),
  }),
);
