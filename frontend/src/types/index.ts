/* ── API types matching the backend schemas ──────────────── */

export interface TopicSummary {
  id: string;
  title: string;
  description: string;
  section_count: number;
  estimated_questions: number;
}

export interface TopicSection {
  heading: string;
  content: string;
}

export interface TopicDetail {
  id: string;
  title: string;
  description: string;
  sections: TopicSection[];
  raw_content: string;
}

export interface QuestionAnswer {
  question: string;
  answer: string;
  difficulty: "easy" | "medium" | "hard";
}

export interface LLMConfig {
  provider: string;
  model: string;
  temperature: number;
  max_tokens: number;
}

export interface GenerateQuestionsRequest {
  topic_id: string;
  count: number;
  difficulty?: string;
  llm_config?: LLMConfig;
  section_title?: string;
  section_content?: string;
}

export interface GenerateQuestionsResponse {
  topic_id: string;
  topic_title: string;
  questions: QuestionAnswer[];
  provider_used: string;
  model_used: string;
}

export interface ProviderStatus {
  name: string;
  available: boolean;
  models: string[];
  backend: string;
}

export interface LLMProvidersResponse {
  providers: ProviderStatus[];
}

export interface HealthStatus {
  llm_chain: boolean;
  cli_agent: boolean;
  llm_chain_version: string;
  cli_agent_version: string;
}

/* ── Chat Follow-Up Types ────────────────────────────────── */

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatFollowUpRequest {
  word: string;
  context_question: string;
  context_answer: string;
  user_message: string;
  history: ChatMessage[];
  llm_config?: LLMConfig;
}

export interface ChatFollowUpResponse {
  reply: string;
  provider_used: string;
  model_used: string;
}

/* ── Quiz Types ──────────────────────────────────────────── */

export type QuizQuestionType = "mcq" | "true_false";

export interface QuizChoice {
  label: string;
  text: string;
}

export interface QuizQuestion {
  question: string;
  type: QuizQuestionType;
  choices: QuizChoice[];
  correct_answer: string;
  explanation: string;
  difficulty: string;
  topic_id: string;
}

export interface GenerateQuizRequest {
  topic_ids: string[];
  count: number;
  question_types: QuizQuestionType[];
  difficulty?: string;
  llm_config?: LLMConfig;
}

export interface GenerateQuizResponse {
  questions: QuizQuestion[];
  topics_used: string[];
  provider_used: string;
  model_used: string;
}

/* ── Ollama Types ────────────────────────────────────────── */

export interface OllamaModelInfo {
  name: string;
  size: string;
  modified_at: string;
}

export interface OllamaModelsResponse {
  connected: boolean;
  downloaded_models: OllamaModelInfo[];
  cloud_models: string[];
}

export interface OllamaTestResponse {
  connected: boolean;
  version: string;
}
