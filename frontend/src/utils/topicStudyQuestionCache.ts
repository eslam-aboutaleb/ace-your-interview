import type { InterviewLevel, QuestionAnswerV2, ResponseDetail } from "@/types";

export type TopicStudyQuestionStatus = "generating" | "ready" | "error";

export type TopicStudyProviderInfo = {
  provider: string;
  model: string;
  retries: number;
  malformed: number;
};

export type TopicStudyQuestionSignature = {
  topicId: string;
  sectionIndex: number;
  sectionFingerprint: string;
  questionCount: number;
  difficulty: string;
  level: InterviewLevel;
  responseDetail: ResponseDetail;
  preferredLanguage: string;
  requiresProgramming: boolean;
  llmProvider: string;
  llmModel: string;
  llmTemperature: number;
  llmMaxTokens: number;
};

export type TopicStudyQuestionCacheEntry = {
  key: string;
  signature: TopicStudyQuestionSignature;
  status: TopicStudyQuestionStatus;
  questions: QuestionAnswerV2[];
  providerInfo: TopicStudyProviderInfo;
  errorMsg: string;
  startedAt: number;
  completedAt: number;
  updatedAt: number;
};

type TopicStudyQuestionCachePayload = {
  version: 3;
  entries: TopicStudyQuestionCacheEntry[];
};

const STORAGE_PREFIX = "ace-your-interview:topic-question-cache:v3:";

function isQuotaExceededError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const maybeError = error as {
    name?: string;
    code?: number;
  };
  return (
    maybeError.name === "QuotaExceededError" ||
    maybeError.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
    maybeError.code === 22 ||
    maybeError.code === 1014
  );
}

function buildPayloadFromEntries(
  entriesByKey: Record<string, TopicStudyQuestionCacheEntry>,
): TopicStudyQuestionCachePayload {
  const entries = Object.values(entriesByKey)
    .filter((entry) => entry.status === "ready" || entry.status === "error")
    .sort((a, b) => b.updatedAt - a.updatedAt);
  return { version: 3, entries };
}

function normalizeProviderInfo(raw: unknown): TopicStudyProviderInfo {
  if (!raw || typeof raw !== "object") {
    return { provider: "", model: "", retries: 0, malformed: 0 };
  }
  const maybe = raw as {
    provider?: unknown;
    model?: unknown;
    retries?: unknown;
    malformed?: unknown;
  };
  return {
    provider: typeof maybe.provider === "string" ? maybe.provider : "",
    model: typeof maybe.model === "string" ? maybe.model : "",
    retries: Number.isFinite(Number(maybe.retries)) ? Number(maybe.retries) : 0,
    malformed: Number.isFinite(Number(maybe.malformed))
      ? Number(maybe.malformed)
      : 0,
  };
}

function normalizeSignature(raw: unknown): TopicStudyQuestionSignature | null {
  if (!raw || typeof raw !== "object") return null;
  const maybe = raw as Record<string, unknown>;
  const topicId = typeof maybe.topicId === "string" ? maybe.topicId : "";
  const sectionFingerprint =
    typeof maybe.sectionFingerprint === "string" ? maybe.sectionFingerprint : "";
  if (!topicId || !sectionFingerprint) return null;

  return {
    topicId,
    sectionIndex: Number.isFinite(Number(maybe.sectionIndex))
      ? Number(maybe.sectionIndex)
      : 0,
    sectionFingerprint,
    questionCount: Number.isFinite(Number(maybe.questionCount))
      ? Number(maybe.questionCount)
      : 0,
    difficulty: typeof maybe.difficulty === "string" ? maybe.difficulty : "",
    level:
      maybe.level === "junior" || maybe.level === "mid" || maybe.level === "senior"
        ? maybe.level
        : "mid",
    responseDetail:
      maybe.responseDetail === "concise" ? "concise" : "very_detailed",
    preferredLanguage:
      typeof maybe.preferredLanguage === "string" ? maybe.preferredLanguage : "",
    requiresProgramming: !!maybe.requiresProgramming,
    llmProvider: typeof maybe.llmProvider === "string" ? maybe.llmProvider : "",
    llmModel: typeof maybe.llmModel === "string" ? maybe.llmModel : "",
    llmTemperature: Number.isFinite(Number(maybe.llmTemperature))
      ? Number(maybe.llmTemperature)
      : 0,
    llmMaxTokens: Number.isFinite(Number(maybe.llmMaxTokens))
      ? Number(maybe.llmMaxTokens)
      : 0,
  };
}

function normalizeEntry(raw: unknown): TopicStudyQuestionCacheEntry | null {
  if (!raw || typeof raw !== "object") return null;
  const maybe = raw as Record<string, unknown>;
  const key = typeof maybe.key === "string" ? maybe.key : "";
  const signature = normalizeSignature(maybe.signature);
  const status = maybe.status;
  if (!key || !signature || (status !== "ready" && status !== "error")) {
    return null;
  }
  const questions = Array.isArray(maybe.questions)
    ? (maybe.questions as QuestionAnswerV2[])
    : [];
  return {
    key,
    signature,
    status,
    questions,
    providerInfo: normalizeProviderInfo(maybe.providerInfo),
    errorMsg: typeof maybe.errorMsg === "string" ? maybe.errorMsg : "",
    startedAt: Number.isFinite(Number(maybe.startedAt))
      ? Number(maybe.startedAt)
      : 0,
    completedAt: Number.isFinite(Number(maybe.completedAt))
      ? Number(maybe.completedAt)
      : 0,
    updatedAt: Number.isFinite(Number(maybe.updatedAt))
      ? Number(maybe.updatedAt)
      : Date.now(),
  };
}

export function makeTopicStudyQuestionCacheStorageKey(topicId: string): string {
  return `${STORAGE_PREFIX}${topicId}`;
}

export function loadTopicStudyQuestionCache(
  topicId: string,
): Record<string, TopicStudyQuestionCacheEntry> {
  if (typeof window === "undefined" || !topicId) return {};
  const key = makeTopicStudyQuestionCacheStorageKey(topicId);
  try {
    const raw = window.sessionStorage.getItem(key);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as TopicStudyQuestionCachePayload;
    if (parsed?.version !== 3) return {};
    const entriesRaw = Array.isArray(parsed?.entries) ? parsed.entries : [];
    const out: Record<string, TopicStudyQuestionCacheEntry> = {};
    for (const item of entriesRaw) {
      const normalized = normalizeEntry(item);
      if (!normalized) continue;
      out[normalized.key] = normalized;
    }
    return out;
  } catch {
    return {};
  }
}

export function persistTopicStudyQuestionCache(
  topicId: string,
  entriesByKey: Record<string, TopicStudyQuestionCacheEntry>,
): void {
  if (typeof window === "undefined" || !topicId) return;
  const key = makeTopicStudyQuestionCacheStorageKey(topicId);
  let payload = buildPayloadFromEntries(entriesByKey);

  while (true) {
    try {
      if (payload.entries.length === 0) {
        window.sessionStorage.removeItem(key);
      } else {
        window.sessionStorage.setItem(key, JSON.stringify(payload));
      }
      return;
    } catch (error) {
      if (!isQuotaExceededError(error)) return;
      if (payload.entries.length === 0) return;
      payload.entries.sort((a, b) => a.updatedAt - b.updatedAt);
      payload.entries.shift();
    }
  }
}
