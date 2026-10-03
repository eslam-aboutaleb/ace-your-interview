import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Node >= 22 exposes an experimental `localStorage` global that is
// undefined unless --localstorage-file is passed. zustand's persist
// middleware reads the bare `localStorage` global, so make sure the
// jsdom implementation wins in tests.
if (typeof globalThis.localStorage === "undefined") {
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    writable: true,
    value: window.localStorage,
  });
}

afterEach(() => {
  cleanup();
});
