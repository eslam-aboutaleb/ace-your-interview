import { create } from "zustand";
import { persist } from "zustand/middleware";

const SETTINGS_STORAGE_KEY = "ace-your-interview-settings";
const LEGACY_SETTINGS_STORAGE_KEY = "study-hub-settings";

function migrateLegacySettingsStorage(): void {
  if (typeof window === "undefined") return;
  try {
    const hasNewValue = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (hasNewValue !== null) return;

    const legacyValue = window.localStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY);
    if (legacyValue === null) return;

    window.localStorage.setItem(SETTINGS_STORAGE_KEY, legacyValue);
    window.localStorage.removeItem(LEGACY_SETTINGS_STORAGE_KEY);
  } catch {
    // Ignore localStorage failures (private mode, quota, etc).
  }
}

migrateLegacySettingsStorage();

interface SettingsState {
  llmSource: "personal" | "study_app";
  provider: string;
  model: string;
  temperature: number;
  maxTokens: number;
  requireAnswerReveal: boolean;
  hydrateFromServer: (payload: {
    llmSource: "personal" | "study_app";
    provider: string;
    model: string;
    temperature: number;
    maxTokens: number;
    requireAnswerReveal: boolean;
  }) => void;
  setLLMSource: (source: "personal" | "study_app") => void;
  setProvider: (p: string) => void;
  setModel: (m: string) => void;
  setTemperature: (t: number) => void;
  setMaxTokens: (t: number) => void;
  setRequireAnswerReveal: (enabled: boolean) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      llmSource: "personal",
      provider: "openai",
      model: "gpt-4o-mini",
      temperature: 0.7,
      maxTokens: 0,
      requireAnswerReveal: true,
      hydrateFromServer: ({
        llmSource,
        provider,
        model,
        temperature,
        maxTokens,
        requireAnswerReveal,
      }) =>
        set({
          llmSource,
          provider,
          model,
          temperature,
          maxTokens,
          requireAnswerReveal,
        }),
      setLLMSource: (llmSource) => set({ llmSource }),
      setProvider: (provider) => set({ provider, model: "" }),
      setModel: (model) => set({ model }),
      setTemperature: (temperature) => set({ temperature }),
      setMaxTokens: (maxTokens) => set({ maxTokens }),
      setRequireAnswerReveal: (requireAnswerReveal) => set({ requireAnswerReveal }),
    }),
    { name: SETTINGS_STORAGE_KEY },
  ),
);
