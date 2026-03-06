import axios from "axios";
import type {
  TopicSummary,
  TopicDetail,
  GenerateQuestionsRequest,
  GenerateQuestionsResponse,
  GenerateQuizRequest,
  GenerateQuizResponse,
  ChatFollowUpRequest,
  ChatFollowUpResponse,
  LLMProvidersResponse,
  HealthStatus,
  OllamaModelsResponse,
  OllamaTestResponse,
} from "@/types";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "/api",
  timeout: 120_000, // LLM calls can be slow
  headers: { "Content-Type": "application/json" },
});

// ── Topics ──────────────────────────────────────────────────
export async function fetchTopics(): Promise<TopicSummary[]> {
  const { data } = await api.get<TopicSummary[]>("/topics");
  return data;
}

export async function fetchTopic(topicId: string): Promise<TopicDetail> {
  const { data } = await api.get<TopicDetail>(`/topics/${topicId}`);
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

// ── Chat Follow-Up ──────────────────────────────────────────
export async function chatFollowUp(
  req: ChatFollowUpRequest,
): Promise<ChatFollowUpResponse> {
  const { data } = await api.post<ChatFollowUpResponse>("/chat/follow-up", req);
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
