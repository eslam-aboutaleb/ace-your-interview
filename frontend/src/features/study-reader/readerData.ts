import { fetchJourney, fetchReaderState } from "@/services/api";
import type { JourneyDetail, ReaderCheckpoint, ReaderStateResponse } from "@/types";

/**
 * Reader server-data cache (issue #11).
 *
 * The cache owns journey + reader-state payloads so page remounts and
 * rail/toolbar re-renders never refetch, while:
 * - entries are scoped to the authenticated user: a different userKey
 *   clears everything, so accounts never see each other's cache;
 * - in-flight requests are deduplicated (one network call per journey);
 * - entries expire after TTL so progress made elsewhere shows up;
 * - explicit invalidation happens on checkpoint conflict re-sync.
 *
 * Question payloads are intentionally NOT mirrored into any global UI
 * store; this module is the only cache and it holds server data only.
 */

export interface ReaderData {
  journey: JourneyDetail;
  readerState: ReaderStateResponse;
  fetchedAt: number;
}

const READER_CACHE_TTL_MS = 30_000;

const cache = new Map<string, ReaderData>();
const inflight = new Map<string, Promise<ReaderData>>();
let cacheOwner: string | null = null;

export async function getReaderData(
  journeyId: string,
  userKey: string,
  signal?: AbortSignal,
): Promise<ReaderData> {
  if (cacheOwner !== userKey) {
    cache.clear();
    inflight.clear();
    cacheOwner = userKey;
  }
  const hit = cache.get(journeyId);
  if (hit && Date.now() - hit.fetchedAt < READER_CACHE_TTL_MS) {
    return hit;
  }
  const pending = inflight.get(journeyId);
  if (pending) return pending;
  const request = Promise.all([
    fetchJourney(journeyId, signal),
    fetchReaderState(journeyId, signal),
  ])
    .then(([journey, readerState]) => {
      const data: ReaderData = { journey, readerState, fetchedAt: Date.now() };
      cache.set(journeyId, data);
      return data;
    })
    .finally(() => {
      inflight.delete(journeyId);
    });
  inflight.set(journeyId, request);
  return request;
}

/** Drop one journey (or everything) so the next read refetches. */
export function invalidateReaderData(journeyId?: string): void {
  if (journeyId) {
    cache.delete(journeyId);
  } else {
    cache.clear();
  }
}

/** Keep the cache coherent after a successful checkpoint write. */
export function updateCachedCheckpoint(journeyId: string, checkpoint: ReaderCheckpoint): void {
  const hit = cache.get(journeyId);
  if (hit) {
    cache.set(journeyId, {
      ...hit,
      readerState: { ...hit.readerState, checkpoint },
    });
  }
}
