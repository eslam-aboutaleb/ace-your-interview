import { afterEach, describe, expect, it, vi } from "vitest";
import { isStudyReaderEnabled } from "./flag";

describe("isStudyReaderEnabled", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("is off by default", () => {
    vi.stubEnv("VITE_STUDY_READER_V1", "");
    expect(isStudyReaderEnabled()).toBe(false);
  });

  it("is on only for the exact string 'true'", () => {
    vi.stubEnv("VITE_STUDY_READER_V1", "true");
    expect(isStudyReaderEnabled()).toBe(true);
  });

  it("rejects truthy-but-not-exact values", () => {
    vi.stubEnv("VITE_STUDY_READER_V1", "1");
    expect(isStudyReaderEnabled()).toBe(false);
  });
});
