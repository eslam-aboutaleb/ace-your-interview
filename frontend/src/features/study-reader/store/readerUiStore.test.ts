import { beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_READER_UI_PREFS,
  nextFontScale,
  readerPrefsFor,
  useReaderUiStore,
} from "./readerUiStore";

describe("readerUiStore", () => {
  beforeEach(() => {
    useReaderUiStore.setState({ prefsByUser: {} });
  });

  it("returns defaults for a user with no saved prefs", () => {
    expect(readerPrefsFor(useReaderUiStore.getState(), "alice")).toEqual(
      DEFAULT_READER_UI_PREFS,
    );
  });

  it("scopes preferences per user", () => {
    useReaderUiStore.getState().setReaderPrefs("alice", { focusMode: true, fontScale: 1.125 });
    expect(readerPrefsFor(useReaderUiStore.getState(), "alice").focusMode).toBe(true);
    expect(readerPrefsFor(useReaderUiStore.getState(), "bob").focusMode).toBe(false);
    expect(readerPrefsFor(useReaderUiStore.getState(), "bob").fontScale).toBe(1);
  });

  it("merges patches without dropping other prefs", () => {
    const store = useReaderUiStore.getState();
    store.setReaderPrefs("alice", { sidebarOpen: false });
    store.setReaderPrefs("alice", { contentWidth: "wide" });
    const prefs = readerPrefsFor(useReaderUiStore.getState(), "alice");
    expect(prefs.sidebarOpen).toBe(false);
    expect(prefs.contentWidth).toBe("wide");
    expect(prefs.fontScale).toBe(1);
  });

  it("steps font scale within bounds", () => {
    expect(nextFontScale(1, 1)).toBe(1.125);
    expect(nextFontScale(1.125, 1)).toBe(1.125); // clamped at max
    expect(nextFontScale(1, -1)).toBe(0.875);
    expect(nextFontScale(0.875, -1)).toBe(0.875); // clamped at min
    expect(nextFontScale(2, -1)).toBe(0.875); // unknown value resets around 1
  });
});
