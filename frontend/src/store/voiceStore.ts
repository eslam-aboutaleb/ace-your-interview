/**
 * Zustand store for voice agent state management.
 *
 * Tracks: connection status, current tier, recording state, transcripts,
 * responses, latency metrics, and user preferences.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  VoiceConfig,
  VoiceLatency,
  VoiceSessionType,
  VoiceTier,
} from "../types";

interface VoiceTurn {
  role: "user" | "assistant";
  text: string;
  audioBase64?: string;
  latency?: VoiceLatency;
  timestamp: number;
}

interface VoiceState {
  /* ── Config (from server) ── */
  config: VoiceConfig | null;
  configLoaded: boolean;

  /* ── Connection ── */
  connected: boolean;
  sessionActive: boolean;
  sessionId: string;
  currentTier: VoiceTier;
  currentSessionType: VoiceSessionType;

  /* ── Recording / VAD ── */
  isListening: boolean;
  isRecording: boolean;
  isSpeaking: boolean; // TTS playing

  /* ── Conversation ── */
  turns: VoiceTurn[];
  currentTranscript: string; // interim transcript

  /* ── Latency ── */
  lastLatency: VoiceLatency | null;

  /* ── User preferences (persisted) ── */
  preferredTier: VoiceTier;
  voiceEnabled: boolean; // user toggle
  autoPlayResponses: boolean;

  /* ── Error ── */
  error: string | null;

  /* ── Actions ── */
  setConfig: (config: VoiceConfig) => void;
  setConnected: (connected: boolean) => void;
  setSessionActive: (active: boolean, sessionId?: string) => void;
  setCurrentTier: (tier: VoiceTier) => void;
  setCurrentSessionType: (type: VoiceSessionType) => void;
  setListening: (listening: boolean) => void;
  setRecording: (recording: boolean) => void;
  setSpeaking: (speaking: boolean) => void;
  setCurrentTranscript: (text: string) => void;
  addUserTurn: (text: string) => void;
  addAssistantTurn: (
    text: string,
    audioBase64?: string,
    latency?: VoiceLatency,
  ) => void;
  setLastLatency: (l: VoiceLatency) => void;
  setPreferredTier: (tier: VoiceTier) => void;
  setVoiceEnabled: (enabled: boolean) => void;
  setAutoPlayResponses: (auto: boolean) => void;
  setError: (error: string | null) => void;
  clearTurns: () => void;
  reset: () => void;
}

const INITIAL_STATE = {
  config: null,
  configLoaded: false,
  connected: false,
  sessionActive: false,
  sessionId: "",
  currentTier: "browser" as VoiceTier,
  currentSessionType: "chat" as VoiceSessionType,
  isListening: false,
  isRecording: false,
  isSpeaking: false,
  turns: [] as VoiceTurn[],
  currentTranscript: "",
  lastLatency: null,
  preferredTier: "browser" as VoiceTier,
  voiceEnabled: false,
  autoPlayResponses: true,
  error: null,
};

export const useVoiceStore = create<VoiceState>()(
  persist(
    (set) => ({
      ...INITIAL_STATE,

      setConfig: (config) =>
        set({
          config,
          configLoaded: true,
          currentTier: config.user_tier || config.default_tier,
        }),

      setConnected: (connected) => set({ connected }),

      setSessionActive: (active, sessionId) =>
        set((s) => ({
          sessionActive: active,
          sessionId: sessionId ?? s.sessionId,
        })),

      setCurrentTier: (tier) => set({ currentTier: tier }),
      setCurrentSessionType: (type) => set({ currentSessionType: type }),

      setListening: (isListening) => set({ isListening }),
      setRecording: (isRecording) => set({ isRecording }),
      setSpeaking: (isSpeaking) => set({ isSpeaking }),

      setCurrentTranscript: (text) => set({ currentTranscript: text }),

      addUserTurn: (text) =>
        set((s) => ({
          turns: [...s.turns, { role: "user", text, timestamp: Date.now() }],
          currentTranscript: "",
        })),

      addAssistantTurn: (text, audioBase64, latency) =>
        set((s) => ({
          turns: [
            ...s.turns,
            {
              role: "assistant",
              text,
              audioBase64,
              latency,
              timestamp: Date.now(),
            },
          ],
        })),

      setLastLatency: (l) => set({ lastLatency: l }),

      setPreferredTier: (preferredTier) => set({ preferredTier }),
      setVoiceEnabled: (voiceEnabled) => set({ voiceEnabled }),
      setAutoPlayResponses: (autoPlayResponses) => set({ autoPlayResponses }),

      setError: (error) => set({ error }),

      clearTurns: () => set({ turns: [], currentTranscript: "" }),

      reset: () =>
        set({
          ...INITIAL_STATE,
          // Keep user preferences on reset
        }),
    }),
    {
      name: "ace-your-interview-voice",
      // Only persist user preferences, not transient state
      partialize: (state) => ({
        preferredTier: state.preferredTier,
        voiceEnabled: state.voiceEnabled,
        autoPlayResponses: state.autoPlayResponses,
      }),
    },
  ),
);
