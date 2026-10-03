import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Node >= 22 exposes an experimental `localStorage` global that is
// undefined unless --local-storage-file is passed, and in this
// environment it shadows the jsdom implementation (window.localStorage
// is undefined too). zustand's persist middleware resolves the bare
// `localStorage` global eagerly at store creation, so fall back to an
// in-memory Storage when neither global is usable.
if (typeof globalThis.localStorage === "undefined" || !globalThis.localStorage) {
  const memoryStorage = new Map<string, string>();
  const memoryLocalStorage = {
    getItem: (key: string) => memoryStorage.get(key) ?? null,
    setItem: (key: string, value: string) => {
      memoryStorage.set(key, String(value));
    },
    removeItem: (key: string) => {
      memoryStorage.delete(key);
    },
    clear: () => memoryStorage.clear(),
    key: (index: number) => Array.from(memoryStorage.keys())[index] ?? null,
    get length() {
      return memoryStorage.size;
    },
  };
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    writable: true,
    value: memoryLocalStorage,
  });
}

afterEach(() => {
  cleanup();
});
