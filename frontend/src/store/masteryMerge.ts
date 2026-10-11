import type { TopicMasteryResponse } from "@/types";

/**
 * Merge rule for the progress store's mastery mirror:
 *
 * The server FSRS state is authoritative. Every topic the server
 * reports replaces the local client-side EWMA mirror, while topics
 * the server does not report (local-only echo writes that have not
 * been synced yet) keep their optimistic local value. The local
 * store therefore stays an optimistic cache: QuizMode/TopicStudy
 * still record attempts for instant feedback, and the next server
 * hydration overrides any stale mirror.
 */
export function mergeServerMastery(
  localMastery: Record<string, number>,
  serverMastery: TopicMasteryResponse | null | undefined,
): Record<string, number> {
  const merged: Record<string, number> = { ...localMastery };
  for (const item of serverMastery?.topics ?? []) {
    const topicId = item?.topic_id;
    if (!topicId) continue;
    const score = Number(item.mastery_score);
    if (!Number.isFinite(score)) continue;
    merged[topicId] = Math.max(0, Math.min(1, score));
  }
  return merged;
}
