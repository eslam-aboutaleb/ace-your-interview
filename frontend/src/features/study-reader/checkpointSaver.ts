import type {
  CheckpointResponse,
  CheckpointUpdateRequest,
  ReaderCheckpoint,
  ReaderStateResponse,
} from "@/types";

/**
 * Checkpoint autosave controller (issue #12).
 *
 * - Reading position is saved after a debounce; an explicit flush()
 *  sends immediately (used for unload and tests).
 * - Every queued mutation carries a stable mutation_id generated at
 *   enqueue time, so retries after ambiguous failures are safe to
 *   repeat (the server dedupes by mutation id).
 * - The queue is durable (localStorage, keyed by user), so offline
 *   edits survive reloads and flush on reconnect.
 * - Conflicts (409) reconcile by accepting the server checkpoint and
 *   dropping the conflicting local mutation; checkpoints only hold
 *   reading position, never completion state, so nothing confirmed
 *   is overwritten.
 * - pagehide uses fetch keepalive as a best-effort extra path, never
 *   the sole persistence path.
 */

export type CheckpointSaveStatus =
  | "idle"
  | "pending"
  | "saving"
  | "saved"
  | "error"
  | "offline";

export interface CheckpointPosition {
  last_batch_id: string;
  last_question_id: string;
  last_section_id?: string;
  scroll_offset?: number;
}

interface QueuedMutation extends CheckpointPosition {
  journey_id: string;
  mutation_id: string;
  expected_revision: number;
  queued_at: string;
}

export interface CheckpointSaverOptions {
  journeyId: string;
  userKey: string;
  getRevision: () => number;
  patch: (journeyId: string, body: CheckpointUpdateRequest) => Promise<CheckpointResponse>;
  fetchState: () => Promise<ReaderStateResponse>;
  onCheckpoint: (checkpoint: ReaderCheckpoint) => void;
  onStatus?: (status: CheckpointSaveStatus) => void;
  debounceMs?: number;
  storage?: Pick<Storage, "getItem" | "setItem" | "removeItem">;
}

export interface CheckpointSaver {
  /** Debounced position update; partial updates merge with the latest. */
  position: (position: Partial<CheckpointPosition>) => void;
  /** Send any pending/queued mutations now. */
  flush: () => Promise<void>;
  /** Best-effort send for pagehide; never the sole persistence path. */
  flushBestEffort: () => void;
  status: () => CheckpointSaveStatus;
  destroy: () => void;
}

function newMutationId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `mutation-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function queueStorageKey(userKey: string): string {
  return `ace-reader-checkpoint-queue:${userKey}`;
}

function isConflict(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    (err as { response?: { status?: number } }).response?.status === 409
  );
}

export function createCheckpointSaver(options: CheckpointSaverOptions): CheckpointSaver {
  const {
    journeyId,
    userKey,
    getRevision,
    patch,
    fetchState,
    onCheckpoint,
    onStatus,
    debounceMs = 1500,
  } = options;
  const storage =
    options.storage ??
    (typeof globalThis.localStorage !== "undefined" ? globalThis.localStorage : undefined);

  let status: CheckpointSaveStatus = "idle";
  let latest: CheckpointPosition | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let flushing: Promise<void> | null = null;
  let destroyed = false;

  const setStatus = (next: CheckpointSaveStatus) => {
    status = next;
    onStatus?.(next);
  };

  const readQueue = (): QueuedMutation[] => {
    if (!storage) return [];
    try {
      const raw = storage.getItem(queueStorageKey(userKey));
      return raw ? (JSON.parse(raw) as QueuedMutation[]) : [];
    } catch {
      return [];
    }
  };

  const writeQueue = (queue: QueuedMutation[]) => {
    if (!storage) return;
    try {
      if (queue.length === 0) {
        storage.removeItem(queueStorageKey(userKey));
      } else {
        storage.setItem(queueStorageKey(userKey), JSON.stringify(queue));
      }
    } catch {
      // storage full or blocked: the in-memory queue still flushes this session
    }
  };

  // In-memory mirror so the module works without storage too.
  let memoryQueue: QueuedMutation[] = readQueue();
  const syncQueue = (queue: QueuedMutation[]) => {
    memoryQueue = queue;
    writeQueue(queue);
  };

  const position: CheckpointSaver["position"] = (partial) => {
    latest = { ...(latest ?? { last_batch_id: "", last_question_id: "" }), ...partial };
    if (!latest.last_batch_id || !latest.last_question_id) return;
    setStatus("pending");
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      enqueueLatest();
      void flush();
    }, debounceMs);
  };

  const enqueueLatest = () => {
    if (!latest) return;
    const queue = readQueue().length > 0 ? readQueue() : memoryQueue;
    const existing = queue.find(
      (m) => m.journey_id === journeyId && m.last_question_id === latest!.last_question_id,
    );
    if (existing) {
      Object.assign(existing, latest);
      syncQueue(queue);
    } else {
      syncQueue([
        ...queue,
        {
          ...latest,
          journey_id: journeyId,
          mutation_id: newMutationId(),
          expected_revision: getRevision(),
          queued_at: new Date().toISOString(),
        },
      ]);
    }
  };

  const flush: CheckpointSaver["flush"] = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
      enqueueLatest();
    }
    if (flushing) return flushing;
    flushing = (async () => {
      let queue = readQueue().length > 0 ? readQueue() : memoryQueue;
      while (queue.length > 0 && !destroyed) {
        const mutation = queue[0];
        setStatus("saving");
        const body: CheckpointUpdateRequest = {
          expected_revision: mutation.expected_revision,
          mutation_id: mutation.mutation_id,
          last_batch_id: mutation.last_batch_id,
          last_question_id: mutation.last_question_id,
          ...(mutation.last_section_id ? { last_section_id: mutation.last_section_id } : {}),
          ...(mutation.scroll_offset != null ? { scroll_offset: mutation.scroll_offset } : {}),
        };
        try {
          const response = await patch(mutation.journey_id, body);
          onCheckpoint(response.checkpoint);
          queue = queue.slice(1);
          syncQueue(queue);
        } catch (err) {
          if (isConflict(err)) {
            // Server wins for position; nothing confirmed is lost.
            try {
              const fresh = await fetchState();
              onCheckpoint(fresh.checkpoint);
            } catch {
              // even the re-sync failed; keep the mutation queued
              setStatus("error");
              return;
            }
            queue = queue.slice(1);
            syncQueue(queue);
          } else {
            setStatus(typeof navigator !== "undefined" && !navigator.onLine ? "offline" : "error");
            return; // keep the queue; retried on flush/online
          }
        }
      }
      if (!destroyed) setStatus("saved");
    })().finally(() => {
      flushing = null;
    });
    return flushing;
  };

  const flushBestEffort: CheckpointSaver["flushBestEffort"] = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
      enqueueLatest();
    }
    const queue = readQueue().length > 0 ? readQueue() : memoryQueue;
    if (queue.length === 0 || typeof fetch !== "function") return;
    const base = (import.meta.env.VITE_API_URL as string | undefined) || "/api";
    for (const mutation of queue) {
      const body: CheckpointUpdateRequest = {
        expected_revision: mutation.expected_revision,
        mutation_id: mutation.mutation_id,
        last_batch_id: mutation.last_batch_id,
        last_question_id: mutation.last_question_id,
        ...(mutation.last_section_id ? { last_section_id: mutation.last_section_id } : {}),
        ...(mutation.scroll_offset != null ? { scroll_offset: mutation.scroll_offset } : {}),
      };
      void fetch(`${base}/journeys/${mutation.journey_id}/checkpoints`, {
        method: "PATCH",
        keepalive: true,
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).catch(() => {
        // best effort only; the durable queue remains the fallback
      });
    }
  };

  const onOnline = () => {
    if (readQueue().length > 0 || memoryQueue.length > 0) void flush();
  };
  if (typeof window !== "undefined") {
    window.addEventListener("online", onOnline);
  }

  return {
    position,
    flush,
    flushBestEffort,
    status: () => status,
    destroy: () => {
      destroyed = true;
      if (timer) clearTimeout(timer);
      if (typeof window !== "undefined") {
        window.removeEventListener("online", onOnline);
      }
    },
  };
}
