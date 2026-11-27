import { create } from "zustand";
import { persist } from "zustand/middleware";

/**
 * Reader UI preferences (issue #11). Zustand holds only harmless
 * client-side preferences; server data (journey, reader state,
 * checkpoint) never enters this store — the URL owns position and
 * the reader-data cache owns server payloads.
 *
 * Preferences are keyed by the authenticated user's handle so an
 * account switch on a shared device never leaks one person's reader
 * setup into another's.
 */

export type ReaderContentWidth = "normal" | "wide";

export interface ReaderUiPrefs {
  sidebarOpen: boolean;
  focusMode: boolean;
  fontScale: number;
  contentWidth: ReaderContentWidth;
}

export const DEFAULT_READER_UI_PREFS: ReaderUiPrefs = {
  sidebarOpen: true,
  focusMode: false,
  fontScale: 1,
  contentWidth: "normal",
};

export const READER_FONT_SCALES = [0.875, 1, 1.125] as const;

interface ReaderUiStore {
  prefsByUser: Record<string, ReaderUiPrefs>;
  setReaderPrefs: (userKey: string, patch: Partial<ReaderUiPrefs>) => void;
}

export const useReaderUiStore = create<ReaderUiStore>()(
  persist(
    (set) => ({
      prefsByUser: {},
      setReaderPrefs: (userKey, patch) =>
        set((state) => ({
          prefsByUser: {
            ...state.prefsByUser,
            [userKey]: { ...DEFAULT_READER_UI_PREFS, ...state.prefsByUser[userKey], ...patch },
          },
        })),
    }),
    { name: "ace-your-interview-reader-ui" },
  ),
);

export function readerPrefsFor(state: ReaderUiStore, userKey: string): ReaderUiPrefs {
  return { ...DEFAULT_READER_UI_PREFS, ...state.prefsByUser[userKey] };
}

export function nextFontScale(current: number, direction: 1 | -1): number {
  const index = READER_FONT_SCALES.indexOf(current as (typeof READER_FONT_SCALES)[number]);
  const safeIndex = index === -1 ? READER_FONT_SCALES.indexOf(1) : index;
  const next = Math.min(Math.max(safeIndex + direction, 0), READER_FONT_SCALES.length - 1);
  return READER_FONT_SCALES[next];
}
