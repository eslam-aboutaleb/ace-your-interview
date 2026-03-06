import axios from "axios";
import type {
  LearningTrack,
  InterviewLevel,
  TopicSummary,
  TopicDetail,
  CreateCustomTopicRequest,
  GenerateQuestionsRequest,
  GenerateQuestionsResponse,
  GenerateQuestionsV2Response,
  GenerateQuizRequest,
  GenerateQuizResponse,
  GenerateQuizV2Response,
  ChatFollowUpRequest,
  ChatFollowUpResponse,
  LLMProvidersResponse,
  HealthStatus,
  OllamaModelsResponse,
  OllamaTestResponse,
  LearningAttemptRequest,
  LearningAttemptResponse,
  ReviewQueueResponse,
  WeakAreasResponse,
  TopicMasteryResponse,
  CreateInterviewSessionRequest,
  SubmitInterviewAnswerRequest,
  NextInterviewQuestionRequest,
  InterviewSessionResponse,
  InterviewTurnResponse,
  InterviewQuestionResponse,
  InterviewReportResponse,
  InterviewSessionsListResponse,
  InterviewStatsResponse,
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

// Redirect to /login on 401
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const reqUrl = String(error.config?.url || "");
    const skip401Redirect =
      reqUrl.includes("/learning/mastery") ||
      reqUrl.includes("/learning/weak-areas") ||
      reqUrl.includes("/learning/review-queue");
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

export async function createCustomTopic(
  req: CreateCustomTopicRequest,
): Promise<TopicDetail> {
  const { data } = await api.post<TopicDetail>("/topics/custom", req);
  return data;
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
