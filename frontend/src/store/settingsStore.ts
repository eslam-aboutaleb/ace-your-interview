import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SettingsState {
  provider: string;
  model: string;
  temperature: number;
  maxTokens: number;
  setProvider: (p: string) => void;
  setModel: (m: string) => void;
  setTemperature: (t: number) => void;
  setMaxTokens: (t: number) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      provider: "openai",
      model: "gpt-4o-mini",
      temperature: 0.7,
      maxTokens: 0,
      setProvider: (provider) => set({ provider, model: "" }),
      setModel: (model) => set({ model }),
      setTemperature: (temperature) => set({ temperature }),
      setMaxTokens: (maxTokens) => set({ maxTokens }),
    }),
    { name: "study-hub-settings" },
  ),
);
