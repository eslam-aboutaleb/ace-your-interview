import axios from "axios";
import type {
  LearningTrack,
  InterviewLevel,
  TopicSummary,
  TopicDetail,
  TopicVideosStatusResponse,
  TopicSectionVideosResponse,
  TopicVideoMetricsResponse,
  TopicPreferencesResponse,
  TopicPreferencesUpdateRequest,
  CreateCustomTopicRequest,
  CustomTopicStreamDoneEvent,
  CustomTopicStreamEvent,
  CustomTopicStreamHandlers,
  GenerateTopicContentRequest,
  TopicContentStreamDoneEvent,
  TopicContentStreamEvent,
  TopicContentStreamHandlers,
  GenerateQuestionsRequest,
  GenerateQuestionsResponse,
  GenerateQuestionsV2Response,
  GenerateQuestionsStreamDoneEvent,
  GenerateQuestionsStreamEvent,
  GenerateQuestionsStreamHandlers,
  GenerateQuizRequest,
  GenerateQuizResponse,
  GenerateQuizV2Response,
  GenerateQuizStreamDoneEvent,
  GenerateQuizStreamEvent,
  GenerateQuizStreamHandlers,
  ChatFollowUpRequest,
  ChatFollowUpResponse,
  LLMProvidersResponse,
  HealthStatus,
  OllamaModelsResponse,
  OllamaTestResponse,
  LearningAttemptRequest,
  LearningAttemptResponse,
  ReviewQueueResponse,
  StudyPlanResponse,
  WeakAreasResponse,
  TopicMasteryResponse,
  LearnerProfile,
  LearnerProfileUpdateRequest,
  ProfileDiagnosticResponse,
  RecommendationsResponse,
  CreateInterviewSessionRequest,
  SubmitInterviewAnswerRequest,
  NextInterviewQuestionRequest,
  InterviewSessionStreamDoneEvent,
  InterviewSessionStreamEvent,
  InterviewSessionStreamHandlers,
  InterviewQuestionStreamDoneEvent,
  InterviewQuestionStreamEvent,
  InterviewQuestionStreamHandlers,
  InterviewTurnStreamDoneEvent,
  InterviewTurnStreamEvent,
  InterviewTurnStreamHandlers,
  InterviewSessionResponse,
  InterviewTurnResponse,
  InterviewQuestionResponse,
  InterviewReportResponse,
  InterviewSessionsListResponse,
  InterviewStatsResponse,
  InterviewTrendsResponse,
  LLMAssignmentsUsersResponse,
  LLMAssignmentUserItem,
  LLMMyAssignmentResponse,
  UserSettingsResponse,
  UserPreferences,
} from "@/types";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "/api",
  timeout: 120_000, // LLM calls can be slow
  headers: { "Content-Type": "application/json" },
  withCredentials: true,
});

function _isApprovalRequired403(error: any): boolean {
  if (error?.response?.status !== 403 && error?.response?.status !== 400) return false;
  const detail = error?.response?.data?.detail;
  if (
    typeof detail === "object" &&
    ["llm_service_approval_required", "study_app_llm_not_assigned", "personal_credential_required"].includes(detail?.code)
  ) {
    return true;
  }
  if (
    typeof detail === "string" &&
    (
      detail.includes("llm_service_approval_required")
      || detail.includes("study_app_llm_not_assigned")
      || detail.includes("personal_credential_required")
    )
  ) {
    return true;
  }
  return false;
}

function _policyCode(error: any): string {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "object" && typeof detail?.code === "string") {
    return detail.code;
  }
  if (typeof detail === "string") {
    if (detail.includes("llm_service_approval_required")) return "llm_service_approval_required";
    if (detail.includes("study_app_llm_not_assigned")) return "study_app_llm_not_assigned";
    if (detail.includes("personal_credential_required")) return "personal_credential_required";
  }
  return "";
}

function _policyCodeFromDetail(detail: unknown): string {
  if (typeof detail === "object" && detail && typeof (detail as any).code === "string") {
    return (detail as any).code;
  }
  if (typeof detail === "string") {
    if (detail.includes("llm_service_approval_required")) return "llm_service_approval_required";
    if (detail.includes("study_app_llm_not_assigned")) return "study_app_llm_not_assigned";
    if (detail.includes("personal_credential_required")) return "personal_credential_required";
  }
  return "";
}

function _redirectToPolicySettings(code: string) {
  if (!code || window.location.pathname.startsWith("/user-settings")) return;
  window.location.href = `/user-settings?policy=${encodeURIComponent(code)}`;
}

function _normalizeApiBaseUrl(): string {
  return String(import.meta.env.VITE_API_URL || "/api").replace(/\/+$/, "");
}

function _resolveApiPath(path: string): string {
  const base = _normalizeApiBaseUrl();
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (base.startsWith("http://") || base.startsWith("https://")) {
    return `${base}${normalizedPath}`;
  }
  return `${base}${normalizedPath}`;
}

export class QuestionsStreamError extends Error {
  readonly code: string;
  readonly status: number;
  readonly fallbackEligible: boolean;

  constructor(
    message: string,
    opts: { code?: string; status?: number; fallbackEligible?: boolean } = {},
  ) {
    super(message);
    this.name = "QuestionsStreamError";
    this.code = opts.code || "";
    this.status = opts.status || 0;
    this.fallbackEligible = !!opts.fallbackEligible;
  }
}

export function isQuestionsStreamError(error: unknown): error is QuestionsStreamError {
  return error instanceof QuestionsStreamError;
}

export class CustomTopicStreamError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(message: string, opts: { code?: string; status?: number } = {}) {
    super(message);
    this.name = "CustomTopicStreamError";
    this.code = opts.code || "";
    this.status = opts.status || 0;
  }
}

export function isCustomTopicStreamError(error: unknown): error is CustomTopicStreamError {
  return error instanceof CustomTopicStreamError;
}

export class TopicContentStreamError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(message: string, opts: { code?: string; status?: number } = {}) {
    super(message);
    this.name = "TopicContentStreamError";
    this.code = opts.code || "";
    this.status = opts.status || 0;
  }
}

export function isTopicContentStreamError(error: unknown): error is TopicContentStreamError {
  return error instanceof TopicContentStreamError;
}

export class QuizStreamError extends Error {
  readonly code: string;
  readonly status: number;
  readonly fallbackEligible: boolean;

  constructor(
    message: string,
    opts: { code?: string; status?: number; fallbackEligible?: boolean } = {},
  ) {
    super(message);
    this.name = "QuizStreamError";
    this.code = opts.code || "";
    this.status = opts.status || 0;
    this.fallbackEligible = !!opts.fallbackEligible;
  }
}

export function isQuizStreamError(error: unknown): error is QuizStreamError {
  return error instanceof QuizStreamError;
}

export class InterviewStreamError extends Error {
  readonly code: string;
  readonly status: number;
  readonly fallbackEligible: boolean;

  constructor(
    message: string,
    opts: { code?: string; status?: number; fallbackEligible?: boolean } = {},
  ) {
    super(message);
    this.name = "InterviewStreamError";
    this.code = opts.code || "";
    this.status = opts.status || 0;
    this.fallbackEligible = !!opts.fallbackEligible;
  }
}

export function isInterviewStreamError(error: unknown): error is InterviewStreamError {
  return error instanceof InterviewStreamError;
}

type StreamErrorOptions = {
  code?: string;
  status?: number;
  fallbackEligible?: boolean;
};

type NdjsonStreamOptions<TEvent, TDone, TError extends Error> = {
  url: string;
  payload: unknown;
  signal?: AbortSignal;
  makeError: (message: string, opts?: StreamErrorOptions) => TError;
  parseEvent: (line: string) => TEvent;
  dispatchEvent: (event: TEvent) => void;
  getDoneEvent: () => TDone | null;
  unavailableMessage: string;
  unavailableCode?: string;
  unavailableFallbackEligible?: (status: number) => boolean;
  incompleteMessage: string;
  incompleteCode?: string;
  incompleteFallbackEligible?: boolean;
};

async function _streamNdjson<TEvent, TDone, TError extends Error>(
  options: NdjsonStreamOptions<TEvent, TDone, TError>,
): Promise<TDone> {
  const {
    url,
    payload,
    signal,
    makeError,
    parseEvent,
    dispatchEvent,
    getDoneEvent,
    unavailableMessage,
    unavailableCode = "stream_unavailable",
    unavailableFallbackEligible = () => false,
    incompleteMessage,
    incompleteCode = "stream_incomplete",
    incompleteFallbackEligible = false,
  } = options;
  let currentStatus = 200;
  let buffer = "";

  const processLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    dispatchEvent(parseEvent(trimmed));
  };
  const processChunk = (chunk: string) => {
    if (!chunk) return;
    buffer += chunk;
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      processLine(line);
    }
  };
  const flushTail = () => {
    if (buffer.trim()) {
      processLine(buffer);
    }
    buffer = "";
  };
  const throwHttpError = (status: number, detail: unknown): never => {
    const policyCode = _policyCodeFromDetail(detail);
    if (policyCode) {
      _redirectToPolicySettings(policyCode);
      throw makeError("LLM access policy blocked this request.", {
        code: policyCode,
        status,
      });
    }
    throw makeError(unavailableMessage, {
      code: unavailableCode,
      status,
      fallbackEligible: unavailableFallbackEligible(status),
    });
  };
  const validateDone = (): TDone => {
    const doneEvent = getDoneEvent();
    if (!doneEvent) {
      throw makeError(incompleteMessage, {
        code: incompleteCode,
        fallbackEligible: incompleteFallbackEligible,
      });
    }
    return doneEvent;
  };

  const streamViaXhr = (): Promise<TDone> =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      let cursor = 0;
      let settled = false;
      const onAbort = () => xhr.abort();
      const finalize = (fn: () => void) => {
        if (settled) return;
        settled = true;
        signal?.removeEventListener("abort", onAbort);
        fn();
      };
      const fail = (error: unknown) => finalize(() => reject(error));
      const succeed = (doneEvent: TDone) => finalize(() => resolve(doneEvent));

      if (signal?.aborted) {
        fail(new DOMException("The operation was aborted.", "AbortError"));
        return;
      }

      signal?.addEventListener("abort", onAbort, { once: true });
      xhr.open("POST", url, true);
      xhr.withCredentials = true;
      xhr.setRequestHeader("Content-Type", "application/json");
      xhr.setRequestHeader("Accept", "application/x-ndjson");

      xhr.onerror = () => {
        fail(
          makeError("Streaming request failed.", {
            code: unavailableCode,
            fallbackEligible: unavailableFallbackEligible(0),
          }),
        );
      };
      xhr.onabort = () => fail(new DOMException("The operation was aborted.", "AbortError"));
      xhr.onprogress = () => {
        if (settled) return;
        if (xhr.status >= 400) return;
        currentStatus = xhr.status || currentStatus;
        const chunk = xhr.responseText.slice(cursor);
        cursor = xhr.responseText.length;
        try {
          processChunk(chunk);
        } catch (error) {
          fail(error);
        }
      };
      xhr.onload = () => {
        if (settled) return;
        currentStatus = xhr.status || currentStatus;
        try {
          const tailChunk = xhr.responseText.slice(cursor);
          cursor = xhr.responseText.length;
          if (xhr.status >= 400) {
            let detail: unknown = null;
            if (tailChunk) {
              try {
                const parsed = JSON.parse(tailChunk);
                detail = (parsed as any)?.detail ?? parsed;
              } catch {
                detail = tailChunk;
              }
            }
            throwHttpError(xhr.status, detail);
            return;
          }
          processChunk(tailChunk);
          flushTail();
          succeed(validateDone());
        } catch (error) {
          fail(error);
        }
      };

      xhr.send(JSON.stringify(payload));
    });

  const supportsFetchStreaming =
    typeof ReadableStream !== "undefined"
    && typeof Response !== "undefined"
    && "body" in Response.prototype;
  if (!supportsFetchStreaming) {
    return streamViaXhr();
  }

  const response = await fetch(url, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/x-ndjson",
      "Cache-Control": "no-cache",
    },
    body: JSON.stringify(payload),
    signal,
  });
  currentStatus = response.status;
  if (!response.ok) {
    let detail: unknown = null;
    try {
      const parsed = await response.json();
      detail = (parsed as any)?.detail ?? parsed;
    } catch {
      detail = null;
    }
    throwHttpError(response.status, detail);
  }
  if (!response.body) {
    return streamViaXhr();
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    processChunk(decoder.decode(value, { stream: true }));
  }
  processChunk(decoder.decode());
  flushTail();

  return validateDone();
}

// Redirect to /login on 401
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const reqUrl = String(error.config?.url || "");
    const skip401Redirect =
      reqUrl.includes("/learning/mastery") ||
      reqUrl.includes("/learning/weak-areas") ||
      reqUrl.includes("/learning/review-queue") ||
      reqUrl.includes("/learning/study-plan");
    if (
      error.response?.status === 401 &&
      !window.location.pathname.startsWith("/login") &&
      !skip401Redirect
    ) {
      window.location.href = "/login";
    }
    if (
      _isApprovalRequired403(error) &&
      !window.location.pathname.startsWith("/user-settings")
    ) {
      const code = _policyCode(error) || "llm_service_approval_required";
      window.location.href = `/user-settings?policy=${encodeURIComponent(code)}`;
    }
    return Promise.reject(error);
  },
);

// ── Topics ──────────────────────────────────────────────────
export interface FetchTopicsParams {
  track?: LearningTrack;
  level?: InterviewLevel;
  q?: string;
  limit?: number;
  offset?: number;
}

export async function fetchTopics(
  params?: FetchTopicsParams,
): Promise<TopicSummary[]> {
  const { data } = await api.get<TopicSummary[]>("/topics", { params });
  return data;
}

export async function fetchTopic(topicId: string): Promise<TopicDetail> {
  const { data } = await api.get<TopicDetail>(`/topics/${topicId}`);
  return data;
}

export async function fetchTopicPreferences(
  topicId: string,
): Promise<TopicPreferencesResponse> {
  const { data } = await api.get<TopicPreferencesResponse>(
    `/topics/${topicId}/preferences`,
  );
  return data;
}

export async function updateTopicPreferences(
  topicId: string,
  payload: TopicPreferencesUpdateRequest,
): Promise<TopicPreferencesResponse> {
  const { data } = await api.put<TopicPreferencesResponse>(
    `/topics/${topicId}/preferences`,
    payload,
  );
  return data;
}

export async function fetchTopicVideosStatus(): Promise<TopicVideosStatusResponse> {
  const { data } = await api.get<TopicVideosStatusResponse>(
    "/features/topic-videos/status",
  );
  return data;
}

export interface FetchTopicSectionVideosParams {
  section_index: number;
  preferred_language?: string;
  limit?: number;
  force_refresh?: boolean;
}

export async function fetchTopicSectionVideos(
  topicId: string,
  params: FetchTopicSectionVideosParams,
): Promise<TopicSectionVideosResponse> {
  const { data } = await api.get<TopicSectionVideosResponse>(
    `/topics/${topicId}/videos`,
    { params },
  );
  return data;
}

export async function recordTopicVideoEvent(payload: {
  event_name: "video_panel_viewed" | "video_click" | "video_panel_hidden_quota" | "video_feature_reenabled";
  topic_id?: string;
  section_index?: number;
  section_heading?: string;
  video_id?: string;
  metadata?: Record<string, unknown>;
}): Promise<void> {
  await api.post("/features/topic-videos/events", payload);
}

export async function fetchTopicVideoMetrics(): Promise<TopicVideoMetricsResponse> {
  const { data } = await api.get<TopicVideoMetricsResponse>(
    "/features/topic-videos/metrics",
  );
  return data;
}

export async function createCustomTopic(
  req: CreateCustomTopicRequest,
): Promise<TopicDetail> {
  const { data } = await api.post<TopicDetail>("/topics/custom", req);
  return data;
}

export async function createCustomTopicStream(
  req: CreateCustomTopicRequest,
  handlers: CustomTopicStreamHandlers,
  signal?: AbortSignal,
): Promise<CustomTopicStreamDoneEvent> {
  let doneEvent: CustomTopicStreamDoneEvent | null = null;

  const dispatchEvent = (event: CustomTopicStreamEvent) => {
    if (event.type === "start") {
      handlers.onStart?.(event);
      return;
    }
    if (event.type === "progress") {
      handlers.onProgress?.(event);
      return;
    }
    if (event.type === "section") {
      handlers.onSection?.(event);
      return;
    }
    if (event.type === "done") {
      doneEvent = event;
      handlers.onDone?.(event);
      return;
    }
    handlers.onError?.(event);
    throw new CustomTopicStreamError(event.message || "Custom topic generation failed.", {
      code: event.code || "generation_failed",
      status: 200,
    });
  };

  return _streamNdjson({
    url: _resolveApiPath("/topics/custom/stream"),
    payload: req,
    signal,
    makeError: (message, opts = {}) =>
      new CustomTopicStreamError(message, {
        code: opts.code,
        status: opts.status,
      }),
    parseEvent: (line) => {
      let parsed: CustomTopicStreamEvent;
      try {
        parsed = JSON.parse(line) as CustomTopicStreamEvent;
      } catch {
        throw new CustomTopicStreamError("Invalid custom topic streaming event payload.", {
          code: "invalid_stream_payload",
        });
      }
      return parsed;
    },
    dispatchEvent,
    getDoneEvent: () => doneEvent,
    unavailableMessage: "Custom topic streaming endpoint unavailable.",
    unavailableCode: "stream_unavailable",
    incompleteMessage: "Custom topic streaming ended before completion event.",
    incompleteCode: "incomplete_stream",
  });
}

export async function generateTopicContentStream(
  topicId: string,
  req: GenerateTopicContentRequest,
  handlers: TopicContentStreamHandlers,
  signal?: AbortSignal,
): Promise<TopicContentStreamDoneEvent> {
  let doneEvent: TopicContentStreamDoneEvent | null = null;

  const dispatchEvent = (event: TopicContentStreamEvent) => {
    if (event.type === "start") {
      handlers.onStart?.(event);
      return;
    }
    if (event.type === "progress") {
      handlers.onProgress?.(event);
      return;
    }
    if (event.type === "section") {
      handlers.onSection?.(event);
      return;
    }
    if (event.type === "done") {
      doneEvent = event;
      handlers.onDone?.(event);
      return;
    }
    handlers.onError?.(event);
    throw new TopicContentStreamError(event.message || "Topic content generation failed.", {
      code: event.code || "generation_failed",
      status: 200,
    });
  };

  return _streamNdjson({
    url: _resolveApiPath(`/topics/${encodeURIComponent(topicId)}/content/generate/stream`),
    payload: req,
    signal,
    makeError: (message, opts = {}) =>
      new TopicContentStreamError(message, {
        code: opts.code,
        status: opts.status,
      }),
    parseEvent: (line) => {
      let parsed: TopicContentStreamEvent;
      try {
        parsed = JSON.parse(line) as TopicContentStreamEvent;
      } catch {
        throw new TopicContentStreamError("Invalid topic content stream payload.", {
          code: "invalid_stream_payload",
        });
      }
      return parsed;
    },
    dispatchEvent,
    getDoneEvent: () => doneEvent,
    unavailableMessage: "Topic content streaming endpoint unavailable.",
    unavailableCode: "stream_unavailable",
    incompleteMessage: "Topic content stream ended before completion event.",
    incompleteCode: "incomplete_stream",
  });
}

// ── Questions ───────────────────────────────────────────────
export async function generateQuestions(
  req: GenerateQuestionsRequest,
): Promise<GenerateQuestionsResponse> {
  const { data } = await api.post<GenerateQuestionsResponse>(
    "/questions/generate",
    req,
  );
  return data;
}

export async function generateQuestionsV2(
  req: GenerateQuestionsRequest,
): Promise<GenerateQuestionsV2Response> {
  const { data } = await api.post<GenerateQuestionsV2Response>(
    "/questions/generate-v2",
    req,
  );
  return data;
}

export async function generateQuestionsV2Stream(
  req: GenerateQuestionsRequest,
  handlers: GenerateQuestionsStreamHandlers,
  signal?: AbortSignal,
): Promise<GenerateQuestionsStreamDoneEvent> {
  let doneEvent: GenerateQuestionsStreamDoneEvent | null = null;

  const dispatchEvent = (event: GenerateQuestionsStreamEvent) => {
    if (event.type === "start") {
      handlers.onStart?.(event);
      return;
    }
    if (event.type === "question") {
      handlers.onQuestion?.(event);
      return;
    }
    if (event.type === "done") {
      doneEvent = event;
      handlers.onDone?.(event);
      return;
    }
    handlers.onError?.(event);
    if (event.code === "llm_service_approval_required"
      || event.code === "study_app_llm_not_assigned"
      || event.code === "personal_credential_required") {
      _redirectToPolicySettings(event.code);
    }
    throw new QuestionsStreamError(event.message || "Question generation failed.", {
      code: event.code || "generation_failed",
      fallbackEligible: false,
    });
  };

  return _streamNdjson({
    url: _resolveApiPath("/questions/generate-v2/stream"),
    payload: req,
    signal,
    makeError: (message, opts = {}) =>
      new QuestionsStreamError(message, {
        code: opts.code,
        status: opts.status,
        fallbackEligible: opts.fallbackEligible,
      }),
    parseEvent: (line) => {
      let parsed: GenerateQuestionsStreamEvent;
      try {
        parsed = JSON.parse(line);
      } catch {
        throw new QuestionsStreamError("Invalid streaming event payload.", {
          code: "stream_parse_failed",
          fallbackEligible: true,
        });
      }
      return parsed;
    },
    dispatchEvent,
    getDoneEvent: () => doneEvent,
    unavailableMessage: "Streaming endpoint unavailable.",
    unavailableCode: "stream_unavailable",
    unavailableFallbackEligible: (status) => status === 404 || status === 405,
    incompleteMessage: "Streaming ended before completion event.",
    incompleteCode: "stream_incomplete",
    incompleteFallbackEligible: true,
  });
}

export async function generateQuizV2Stream(
  req: GenerateQuizRequest,
  handlers: GenerateQuizStreamHandlers,
  signal?: AbortSignal,
): Promise<GenerateQuizStreamDoneEvent> {
  let doneEvent: GenerateQuizStreamDoneEvent | null = null;

  const dispatchEvent = (event: GenerateQuizStreamEvent) => {
    if (event.type === "start") {
      handlers.onStart?.(event);
      return;
    }
    if (event.type === "progress") {
      handlers.onProgress?.(event);
      return;
    }
    if (event.type === "question") {
      handlers.onQuestion?.(event);
      return;
    }
    if (event.type === "done") {
      doneEvent = event;
      handlers.onDone?.(event);
      return;
    }
    handlers.onError?.(event);
    if (event.code === "llm_service_approval_required"
      || event.code === "study_app_llm_not_assigned"
      || event.code === "personal_credential_required") {
      _redirectToPolicySettings(event.code);
    }
    throw new QuizStreamError(event.message || "Quiz generation failed.", {
      code: event.code || "generation_failed",
      fallbackEligible: false,
    });
  };

  return _streamNdjson({
    url: _resolveApiPath("/questions/quiz/generate-v2/stream"),
    payload: req,
    signal,
    makeError: (message, opts = {}) =>
      new QuizStreamError(message, {
        code: opts.code,
        status: opts.status,
        fallbackEligible: opts.fallbackEligible,
      }),
    parseEvent: (line) => {
      let parsed: GenerateQuizStreamEvent;
      try {
        parsed = JSON.parse(line);
      } catch {
        throw new QuizStreamError("Invalid streaming event payload.", {
          code: "stream_parse_failed",
          fallbackEligible: true,
        });
      }
      return parsed;
    },
    dispatchEvent,
    getDoneEvent: () => doneEvent,
    unavailableMessage: "Quiz streaming endpoint unavailable.",
    unavailableCode: "stream_unavailable",
    unavailableFallbackEligible: (status) => status === 404 || status === 405,
    incompleteMessage: "Quiz streaming ended before completion event.",
    incompleteCode: "stream_incomplete",
    incompleteFallbackEligible: true,
  });
}

// ── Quiz ────────────────────────────────────────────────────
export async function generateQuiz(
  req: GenerateQuizRequest,
): Promise<GenerateQuizResponse> {
  const { data } = await api.post<GenerateQuizResponse>(
    "/questions/quiz/generate",
    req,
  );
  return data;
}

export async function generateQuizV2(
  req: GenerateQuizRequest,
): Promise<GenerateQuizV2Response> {
  const { data } = await api.post<GenerateQuizV2Response>(
    "/questions/quiz/generate-v2",
    req,
  );
  return data;
}

// ── Chat Follow-Up ──────────────────────────────────────────
export async function chatFollowUp(
  req: ChatFollowUpRequest,
): Promise<ChatFollowUpResponse> {
  const { data } = await api.post<ChatFollowUpResponse>("/chat/follow-up", req);
  return data;
}

// ── Adaptive Learning ────────────────────────────────────────
export async function recordLearningAttempt(
  req: LearningAttemptRequest,
): Promise<LearningAttemptResponse> {
  const { data } = await api.post<LearningAttemptResponse>("/learning/attempts", req);
  return data;
}

export async function fetchReviewQueue(
  limit = 50,
): Promise<ReviewQueueResponse> {
  const { data } = await api.get<ReviewQueueResponse>("/learning/review-queue", {
    params: { limit },
  });
  return data;
}

export async function fetchWeakAreas(
  limit = 10,
): Promise<WeakAreasResponse> {
  const { data } = await api.get<WeakAreasResponse>("/learning/weak-areas", {
    params: { limit },
  });
  return data;
}

export async function fetchTopicMastery(
  limit = 500,
): Promise<TopicMasteryResponse> {
  const { data } = await api.get<TopicMasteryResponse>("/learning/mastery", {
    params: { limit },
  });
  return data;
}

export async function fetchLearnerProfile(): Promise<LearnerProfile> {
  const { data } = await api.get<LearnerProfile>("/learning/profile");
  return data;
}

export async function updateLearnerProfile(
  payload: LearnerProfileUpdateRequest,
): Promise<LearnerProfile> {
  const { data } = await api.put<LearnerProfile>("/learning/profile", payload);
  return data;
}

export async function runProfileDiagnostic(): Promise<ProfileDiagnosticResponse> {
  const { data } = await api.post<ProfileDiagnosticResponse>(
    "/learning/profile/diagnostic",
  );
  return data;
}

export async function fetchRecommendations(
  limit = 8,
): Promise<RecommendationsResponse> {
  const { data } = await api.get<RecommendationsResponse>(
    "/learning/recommendations",
    { params: { limit } },
  );
  return data;
}

export async function fetchStudyPlan(
  days = 7,
  dailyItems = 3,
): Promise<StudyPlanResponse> {
  const { data } = await api.get<StudyPlanResponse>("/learning/study-plan", {
    params: { days, daily_items: dailyItems },
  });
  return data;
}

// ── Mock Interview ──────────────────────────────────────────
export async function createInterviewSession(
  req: CreateInterviewSessionRequest,
): Promise<InterviewSessionResponse> {
  const { data } = await api.post<InterviewSessionResponse>(
    "/interview-sessions",
    req,
  );
  return data;
}

export async function createInterviewSessionStream(
  req: CreateInterviewSessionRequest,
  handlers: InterviewSessionStreamHandlers,
  signal?: AbortSignal,
): Promise<InterviewSessionStreamDoneEvent> {
  let doneEvent: InterviewSessionStreamDoneEvent | null = null;

  const dispatchEvent = (event: InterviewSessionStreamEvent) => {
    if (event.type === "start") {
      handlers.onStart?.(event);
      return;
    }
    if (event.type === "progress") {
      handlers.onProgress?.(event);
      return;
    }
    if (event.type === "done") {
      doneEvent = event;
      handlers.onDone?.(event);
      return;
    }
    handlers.onError?.(event);
    if (event.code === "llm_service_approval_required"
      || event.code === "study_app_llm_not_assigned"
      || event.code === "personal_credential_required") {
      _redirectToPolicySettings(event.code);
    }
    throw new InterviewStreamError(event.message || "Interview generation failed.", {
      code: event.code || "generation_failed",
      fallbackEligible: false,
    });
  };

  return _streamNdjson({
    url: _resolveApiPath("/interview-sessions/stream"),
    payload: req,
    signal,
    makeError: (message, opts = {}) =>
      new InterviewStreamError(message, {
        code: opts.code,
        status: opts.status,
        fallbackEligible: opts.fallbackEligible,
      }),
    parseEvent: (line) => {
      let parsed: InterviewSessionStreamEvent;
      try {
        parsed = JSON.parse(line);
      } catch {
        throw new InterviewStreamError("Invalid interview stream payload.", {
          code: "stream_parse_failed",
          fallbackEligible: true,
        });
      }
      return parsed;
    },
    dispatchEvent,
    getDoneEvent: () => doneEvent,
    unavailableMessage: "Interview streaming endpoint unavailable.",
    unavailableCode: "stream_unavailable",
    unavailableFallbackEligible: (status) => status === 404 || status === 405,
    incompleteMessage: "Interview stream ended before completion event.",
    incompleteCode: "stream_incomplete",
    incompleteFallbackEligible: true,
  });
}

export async function fetchInterviewSessions(
  limit = 20,
  offset = 0,
): Promise<InterviewSessionsListResponse> {
  const { data } = await api.get<InterviewSessionsListResponse>(
    "/interview-sessions",
    { params: { limit, offset } },
  );
  return data;
}

export async function fetchInterviewSession(
  sessionId: string,
): Promise<InterviewSessionResponse> {
  const { data } = await api.get<InterviewSessionResponse>(
    `/interview-sessions/${sessionId}`,
  );
  return data;
}

export async function submitInterviewAnswer(
  sessionId: string,
  req: SubmitInterviewAnswerRequest,
): Promise<InterviewTurnResponse> {
  const { data } = await api.post<InterviewTurnResponse>(
    `/interview-sessions/${sessionId}/answer`,
    req,
  );
  return data;
}

export async function submitInterviewAnswerStream(
  sessionId: string,
  req: SubmitInterviewAnswerRequest,
  handlers: InterviewTurnStreamHandlers,
  signal?: AbortSignal,
): Promise<InterviewTurnStreamDoneEvent> {
  let doneEvent: InterviewTurnStreamDoneEvent | null = null;

  const dispatchEvent = (event: InterviewTurnStreamEvent) => {
    if (event.type === "start") {
      handlers.onStart?.(event);
      return;
    }
    if (event.type === "progress") {
      handlers.onProgress?.(event);
      return;
    }
    if (event.type === "done") {
      doneEvent = event;
      handlers.onDone?.(event);
      return;
    }
    handlers.onError?.(event);
    if (event.code === "llm_service_approval_required"
      || event.code === "study_app_llm_not_assigned"
      || event.code === "personal_credential_required") {
      _redirectToPolicySettings(event.code);
    }
    throw new InterviewStreamError(event.message || "Interview evaluation failed.", {
      code: event.code || "generation_failed",
      fallbackEligible: false,
    });
  };

  return _streamNdjson({
    url: _resolveApiPath(`/interview-sessions/${encodeURIComponent(sessionId)}/answer/stream`),
    payload: req,
    signal,
    makeError: (message, opts = {}) =>
      new InterviewStreamError(message, {
        code: opts.code,
        status: opts.status,
        fallbackEligible: opts.fallbackEligible,
      }),
    parseEvent: (line) => {
      let parsed: InterviewTurnStreamEvent;
      try {
        parsed = JSON.parse(line);
      } catch {
        throw new InterviewStreamError("Invalid interview stream payload.", {
          code: "stream_parse_failed",
          fallbackEligible: true,
        });
      }
      return parsed;
    },
    dispatchEvent,
    getDoneEvent: () => doneEvent,
    unavailableMessage: "Interview streaming endpoint unavailable.",
    unavailableCode: "stream_unavailable",
    unavailableFallbackEligible: (status) => status === 404 || status === 405,
    incompleteMessage: "Interview stream ended before completion event.",
    incompleteCode: "stream_incomplete",
    incompleteFallbackEligible: true,
  });
}

export async function generateNextInterviewQuestion(
  sessionId: string,
  req: NextInterviewQuestionRequest,
): Promise<InterviewQuestionResponse> {
  const { data } = await api.post<InterviewQuestionResponse>(
    `/interview-sessions/${sessionId}/next-question`,
    req,
  );
  return data;
}

export async function generateNextInterviewQuestionStream(
  sessionId: string,
  req: NextInterviewQuestionRequest,
  handlers: InterviewQuestionStreamHandlers,
  signal?: AbortSignal,
): Promise<InterviewQuestionStreamDoneEvent> {
  let doneEvent: InterviewQuestionStreamDoneEvent | null = null;

  const dispatchEvent = (event: InterviewQuestionStreamEvent) => {
    if (event.type === "start") {
      handlers.onStart?.(event);
      return;
    }
    if (event.type === "progress") {
      handlers.onProgress?.(event);
      return;
    }
    if (event.type === "done") {
      doneEvent = event;
      handlers.onDone?.(event);
      return;
    }
    handlers.onError?.(event);
    if (event.code === "llm_service_approval_required"
      || event.code === "study_app_llm_not_assigned"
      || event.code === "personal_credential_required") {
      _redirectToPolicySettings(event.code);
    }
    throw new InterviewStreamError(event.message || "Interview generation failed.", {
      code: event.code || "generation_failed",
      fallbackEligible: false,
    });
  };

  return _streamNdjson({
    url: _resolveApiPath(`/interview-sessions/${encodeURIComponent(sessionId)}/next-question/stream`),
    payload: req,
    signal,
    makeError: (message, opts = {}) =>
      new InterviewStreamError(message, {
        code: opts.code,
        status: opts.status,
        fallbackEligible: opts.fallbackEligible,
      }),
    parseEvent: (line) => {
      let parsed: InterviewQuestionStreamEvent;
      try {
        parsed = JSON.parse(line);
      } catch {
        throw new InterviewStreamError("Invalid interview stream payload.", {
          code: "stream_parse_failed",
          fallbackEligible: true,
        });
      }
      return parsed;
    },
    dispatchEvent,
    getDoneEvent: () => doneEvent,
    unavailableMessage: "Interview streaming endpoint unavailable.",
    unavailableCode: "stream_unavailable",
    unavailableFallbackEligible: (status) => status === 404 || status === 405,
    incompleteMessage: "Interview stream ended before completion event.",
    incompleteCode: "stream_incomplete",
    incompleteFallbackEligible: true,
  });
}

export async function fetchInterviewReport(
  sessionId: string,
): Promise<InterviewReportResponse> {
  const { data } = await api.get<InterviewReportResponse>(
    `/interview-sessions/${sessionId}/report`,
  );
  return data;
}

export async function fetchInterviewStats(): Promise<InterviewStatsResponse> {
  const { data } = await api.get<InterviewStatsResponse>(
    "/interview-sessions/stats",
  );
  return data;
}

export async function fetchInterviewTrends(
  limit = 50,
): Promise<InterviewTrendsResponse> {
  const { data } = await api.get<InterviewTrendsResponse>(
    "/interview-sessions/trends",
    { params: { limit } },
  );
  return data;
}

// ── LLM Providers ───────────────────────────────────────────
export async function fetchProviders(): Promise<LLMProvidersResponse> {
  const { data } = await api.get<LLMProvidersResponse>("/llm/providers");
  return data;
}

export async function fetchHealth(): Promise<HealthStatus> {
  const { data } = await api.get<HealthStatus>("/llm/health");
  return data;
}

// ── Ollama ──────────────────────────────────────────────────
export async function testOllamaConnection(): Promise<OllamaTestResponse> {
  const { data } = await api.get<OllamaTestResponse>("/llm/ollama/test");
  return data;
}

export async function fetchOllamaModels(): Promise<OllamaModelsResponse> {
  const { data } = await api.get<OllamaModelsResponse>("/llm/ollama/models");
  return data;
}

// ── Auth – Allowed Users ────────────────────────────────────
export interface AllowedUsers {
  github_users: string[];
  google_emails: string[];
}

export interface LLMServiceUsers {
  users: string[];
}

export async function fetchAllowedUsers(): Promise<AllowedUsers> {
  const { data } = await api.get<AllowedUsers>("/auth/allowed-users");
  return data;
}

export async function addAllowedUser(
  provider: string,
  identifier: string,
): Promise<AllowedUsers> {
  const { data } = await api.post<AllowedUsers>("/auth/allowed-users", {
    provider,
    identifier,
  });
  return data;
}

export async function removeAllowedUser(
  provider: string,
  identifier: string,
): Promise<AllowedUsers> {
  const { data } = await api.delete<AllowedUsers>(
    `/auth/allowed-users/${provider}/${encodeURIComponent(identifier)}`,
  );
  return data;
}

export async function fetchLLMServiceUsers(): Promise<LLMServiceUsers> {
  const { data } = await api.get<LLMServiceUsers>("/auth/llm-service-users");
  return data;
}

export async function addLLMServiceUser(
  provider: string,
  identifier: string,
): Promise<LLMServiceUsers> {
  const { data } = await api.post<LLMServiceUsers>("/auth/llm-service-users", {
    provider,
    identifier,
  });
  return data;
}

export async function removeLLMServiceUser(
  provider: string,
  identifier: string,
): Promise<LLMServiceUsers> {
  const { data } = await api.delete<LLMServiceUsers>(
    `/auth/llm-service-users/${provider}/${encodeURIComponent(identifier)}`,
  );
  return data;
}

export async function fetchLLMAssignmentUsers(): Promise<LLMAssignmentsUsersResponse> {
  const { data } = await api.get<LLMAssignmentsUsersResponse>("/auth/llm-assignments/users");
  return data;
}

export async function saveLLMAssignmentForUser(
  loginProvider: "google" | "github",
  identifier: string,
  provider: string,
  model: string,
): Promise<LLMAssignmentUserItem> {
  const { data } = await api.put<LLMAssignmentUserItem>(
    `/auth/llm-assignments/${loginProvider}/${encodeURIComponent(identifier)}`,
    { provider, model },
  );
  return data;
}

export async function deleteLLMAssignmentForUser(
  loginProvider: "google" | "github",
  identifier: string,
): Promise<LLMAssignmentUserItem> {
  const { data } = await api.delete<LLMAssignmentUserItem>(
    `/auth/llm-assignments/${loginProvider}/${encodeURIComponent(identifier)}`,
  );
  return data;
}

export async function fetchMyLLMAssignment(): Promise<LLMMyAssignmentResponse> {
  const { data } = await api.get<LLMMyAssignmentResponse>("/auth/llm-assignments/me");
  return data;
}

export async function saveMyLLMAssignment(
  provider: string,
  model: string,
): Promise<LLMMyAssignmentResponse> {
  const { data } = await api.put<LLMMyAssignmentResponse>("/auth/llm-assignments/me", {
    provider,
    model,
  });
  return data;
}

// ── User Settings ──────────────────────────────────────────
export async function fetchUserSettings(): Promise<UserSettingsResponse> {
  const { data } = await api.get<UserSettingsResponse>("/user-settings");
  return data;
}

export async function updateUserPreferences(
  payload: UserPreferences,
): Promise<UserPreferences> {
  const { data } = await api.put<UserPreferences>(
    "/user-settings/preferences",
    payload,
  );
  return data;
}

export async function saveUserApiKey(
  provider: string,
  apiKey: string,
): Promise<UserSettingsResponse> {
  const { data } = await api.put<UserSettingsResponse>(
    `/user-settings/api-key/${provider}`,
    { api_key: apiKey },
  );
  return data;
}

export async function deleteUserApiKey(
  provider: string,
): Promise<UserSettingsResponse> {
  const { data } = await api.delete<UserSettingsResponse>(
    `/user-settings/api-key/${provider}`,
  );
  return data;
}

export function connectGeminiAccount(): string {
  const base = (import.meta.env.VITE_API_URL || "/api").replace(/\/$/, "");
  return `${base}/user-settings/google/connect`;
}

export async function disconnectGeminiAccount(): Promise<UserSettingsResponse> {
  const { data } = await api.post<UserSettingsResponse>(
    "/user-settings/google/disconnect",
  );
  return data;
}
