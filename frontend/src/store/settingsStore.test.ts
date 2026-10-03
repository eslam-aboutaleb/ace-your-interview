import { beforeEach, describe, expect, it } from "vitest";
import { useSettingsStore } from "./settingsStore";

describe("settingsStore", () => {
  beforeEach(() => {
    useSettingsStore.setState({
      llmSource: "personal",
      provider: "openai",
      model: "gpt-4o-mini",
      temperature: 0.7,
      maxTokens: 0,
      requireAnswerReveal: false,
    });
  });

  it("hydrateFromServer overwrites local settings with server preferences", () => {
    useSettingsStore.getState().hydrateFromServer({
      llmSource: "study_app",
      provider: "google",
      model: "gemini-2.0-flash",
      temperature: 0.2,
      maxTokens: 4096,
      requireAnswerReveal: true,
    });

    const state = useSettingsStore.getState();
    expect(state.llmSource).toBe("study_app");
    expect(state.provider).toBe("google");
    expect(state.model).toBe("gemini-2.0-flash");
    expect(state.temperature).toBe(0.2);
    expect(state.maxTokens).toBe(4096);
    expect(state.requireAnswerReveal).toBe(true);
  });
});
