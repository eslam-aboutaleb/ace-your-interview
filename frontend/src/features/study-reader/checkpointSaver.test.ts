import { beforeEach, describe, expect, it, vi } from "vitest";
import type { CheckpointResponse, ReaderStateResponse } from "@/types";
import { createCheckpointSaver, type CheckpointSaverOptions } from "./checkpointSaver";

function memoryStorage() {
  const map = new Map<string, string>();
  return {
    getItem: (key: string) => map.get(key) ?? null,
    setItem: (key: string, value: string) => void map.set(key, String(value)),
    removeItem: (key: string) => void map.delete(key),
    map,
  };
}

function makeOptions(overrides: Partial<CheckpointSaverOptions> = {}) {
  const patch = vi.fn<(journeyId: string, body: unknown) => Promise<CheckpointResponse>>().mockResolvedValue({
    api_version: "v1",
    checkpoint: {
      last_batch_id: "b1",
      last_question_id: "q1",
      last_section_id: "",
      scroll_offset: 0,
      revision: 5,
      updated_at: "2026-01-03T00:00:00Z",
    },
  });
  const fetchState = vi.fn<() => Promise<ReaderStateResponse>>().mockResolvedValue({
    api_version: "v1",
    journey_id: "j1",
    checkpoint: {
      last_batch_id: "b1",
      last_question_id: "q9",
      last_section_id: "",
      scroll_offset: 0,
      revision: 8,
      updated_at: "2026-01-03T00:00:00Z",
    },
    latest_batch: null,
    questions: [],
  });
  const onCheckpoint = vi.fn();
  const onStatus = vi.fn();
  const options: CheckpointSaverOptions = {
    journeyId: "j1",
    userKey: "alice",
    getRevision: () => 4,
    patch,
    fetchState,
    onCheckpoint,
    onStatus,
    debounceMs: 100,
    storage: memoryStorage(),
    ...overrides,
  };
  return { options, patch, fetchState, onCheckpoint, onStatus };
}

describe("checkpointSaver", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  it("debounces rapid position updates into one PATCH with the latest value", async () => {
    vi.useFakeTimers();
    const { options, patch } = makeOptions();
    const saver = createCheckpointSaver(options);
    saver.position({ last_batch_id: "b1", last_question_id: "q1" });
    saver.position({ last_batch_id: "b1", last_question_id: "q2" });
    saver.position({ last_batch_id: "b1", last_question_id: "q3" });
    await vi.advanceTimersByTimeAsync(150);
    expect(patch).toHaveBeenCalledTimes(1);
    expect(patch.mock.calls[0][1]).toMatchObject({ last_question_id: "q3" });
    saver.destroy();
  });

  it("flushes immediately on explicit flush with pending status transitions", async () => {
    const { options, patch, onStatus } = makeOptions();
    const saver = createCheckpointSaver(options);
    saver.position({ last_batch_id: "b1", last_question_id: "q2", scroll_offset: 120 });
    await saver.flush();
    expect(patch).toHaveBeenCalledTimes(1);
    expect(patch.mock.calls[0][1]).toMatchObject({
      expected_revision: 4,
      last_question_id: "q2",
      scroll_offset: 120,
      mutation_id: expect.any(String),
    });
    expect(onStatus.mock.calls.map((c) => c[0])).toEqual(
      expect.arrayContaining(["pending", "saving", "saved"]),
    );
    saver.destroy();
  });

  it("reconciles a 409 by accepting the server checkpoint and draining the queue", async () => {
    const { options, patch, fetchState, onCheckpoint } = makeOptions();
    patch.mockRejectedValueOnce({ response: { status: 409 } });
    const saver = createCheckpointSaver(options);
    saver.position({ last_batch_id: "b1", last_question_id: "q2" });
    await saver.flush();
    expect(fetchState).toHaveBeenCalledTimes(1);
    expect(onCheckpoint).toHaveBeenCalledWith(
      expect.objectContaining({ last_question_id: "q9", revision: 8 }),
    );
    // queue drained: a second flush is a no-op
    patch.mockResolvedValue({
      api_version: "v1",
      checkpoint: { last_batch_id: "b1", last_question_id: "q2", last_section_id: "", scroll_offset: 0, revision: 9, updated_at: "" },
    });
    await saver.flush();
    expect(patch).toHaveBeenCalledTimes(1);
    saver.destroy();
  });

  it("keeps the mutation queued on network failure and retries on flush", async () => {
    const { options, patch } = makeOptions();
    patch.mockRejectedValueOnce(new Error("network down"));
    const saver = createCheckpointSaver(options);
    saver.position({ last_batch_id: "b1", last_question_id: "q2" });
    await saver.flush();
    expect(saver.status()).toBe("error");
    await saver.flush();
    expect(patch).toHaveBeenCalledTimes(2);
    expect(saver.status()).toBe("saved");
    saver.destroy();
  });

  it("reuses the same mutation_id when retrying after an ambiguous failure", async () => {
    const { options, patch } = makeOptions();
    patch.mockRejectedValueOnce(new Error("response lost"));
    const saver = createCheckpointSaver(options);
    saver.position({ last_batch_id: "b1", last_question_id: "q2" });
    await saver.flush();
    await saver.flush();
    const first = (patch.mock.calls[0][1] as { mutation_id: string }).mutation_id;
    const second = (patch.mock.calls[1][1] as { mutation_id: string }).mutation_id;
    expect(first).toBe(second);
    saver.destroy();
  });

  it("persists the queue so a new saver instance flushes it after reload", async () => {
    const storage = memoryStorage();
    const first = makeOptions({ storage });
    first.patch.mockRejectedValueOnce(new Error("offline"));
    const saver1 = createCheckpointSaver(first.options);
    saver1.position({ last_batch_id: "b1", last_question_id: "q7" });
    await saver1.flush();
    saver1.destroy();

    const second = makeOptions({ storage });
    const saver2 = createCheckpointSaver(second.options);
    await saver2.flush();
    expect(second.patch).toHaveBeenCalledTimes(1);
    expect(second.patch.mock.calls[0][1]).toMatchObject({ last_question_id: "q7" });
    saver2.destroy();
  });

  it("fires a best-effort keepalive fetch on pagehide for queued mutations", async () => {
    const storage = memoryStorage();
    const fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetchMock);
    const { options } = makeOptions({ storage });
    const saver = createCheckpointSaver(options);
    saver.position({ last_batch_id: "b1", last_question_id: "q2", scroll_offset: 10 });
    saver.flushBestEffort();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/journeys/j1/checkpoints");
    expect(init.method).toBe("PATCH");
    expect(init.keepalive).toBe(true);
    vi.unstubAllGlobals();
    saver.destroy();
  });
});
