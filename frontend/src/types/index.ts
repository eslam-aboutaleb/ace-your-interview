/* ── API types matching the backend schemas ──────────────── */

export type LearningTrack =
  | "backend"
  | "frontend"
  | "system_design"
  | "ai_stack";
export type InterviewLevel = "junior" | "mid" | "senior";
export type ResponseDetail = "concise" | "very_detailed";
export type InterviewType =
  | "behavioral"
  | "technical"
  | "system_design"
  | "ai_fundamentals"
  | "coding"
  | "mixed";
export type InterviewerStyle = "supportive" | "neutral" | "challenging";
export type FeedbackMode = "concise" | "deep";

export interface TopicSummary {
  id: string;
  title: string;
  description: string;
  track: LearningTrack;
  levels: InterviewLevel[];
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
  track: LearningTrack;
  levels: InterviewLevel[];
  sections: TopicSection[];
  raw_content: string;
  requires_programming: boolean;
  language_options: string[];
  selected_language: string;
  response_detail: ResponseDetail;
  is_dynamic_topic: boolean;
  content_ready: boolean;
}

export type TopicVideosStatus =
  | "enabled"
  | "disabled_quota_exhausted"
  | "probe"
  | "disabled_config";

export interface VideoResource {
  video_id: string;
  title: string;
  url: string;
  channel: string;
  duration_seconds: number;
  thumbnail_url: string;
  published_at: string;
  source: string;
}

export interface TopicVideosStatusResponse {
  enabled: boolean;
  status: TopicVideosStatus;
  disabled_until: string;
  reason: string;
}

export interface TopicSectionVideosResponse extends TopicVideosStatusResponse {
  topic_id: string;
  section_index: number;
  section_heading: string;
  videos: VideoResource[];
  source: string;
  cached: boolean;
}

export interface TopicVideoMetricsResponse {
  status: TopicVideosStatus;
  disabled_until: string;
  reason: string;
  hidden_now: boolean;
  hidden_days_last_7: number;
  hidden_days_last_30: number;
  hidden_frequency_weekly: Array<{
    week_start: string;
    hidden_events: number;
  }>;
  panel_views_30d: number;
  video_clicks_30d: number;
  ctr_30d: number;
  reenabled_at: string;
  panel_views_since_reenable: number;
  clicks_since_reenable: number;
  ctr_since_reenable: number;
}

export interface CreateCustomTopicRequest {
  topic: string;
  target_sections?: number;
  llm_config?: LLMConfig;
}

export interface CustomTopicStreamStartEvent {
  type: "start";
  topic: string;
  target_sections: number;
}

export interface CustomTopicStreamProgressEvent {
  type: "progress";
  stage: "analyzing" | "batching" | "ready";
  message: string;
  elapsed_seconds?: number;
  batch_index?: number;
  batch_count?: number;
  generated_sections?: number;
  target_sections?: number;
}

export interface CustomTopicStreamSectionEvent {
  type: "section";
  index: number;
  total_sections: number;
  heading: string;
  content: string;
}

export interface CustomTopicStreamDoneEvent {
  type: "done";
  topic: TopicDetail;
}

export interface CustomTopicStreamErrorEvent {
  type: "error";
  code:
    | "llm_service_approval_required"
    | "study_app_llm_not_assigned"
    | "personal_credential_required"
    | "generation_failed";
  message: string;
}

export type CustomTopicStreamEvent =
  | CustomTopicStreamStartEvent
  | CustomTopicStreamProgressEvent
  | CustomTopicStreamSectionEvent
  | CustomTopicStreamDoneEvent
  | CustomTopicStreamErrorEvent;

export interface CustomTopicStreamHandlers {
  onStart?: (event: CustomTopicStreamStartEvent) => void;
  onProgress?: (event: CustomTopicStreamProgressEvent) => void;
  onSection?: (event: CustomTopicStreamSectionEvent) => void;
  onDone?: (event: CustomTopicStreamDoneEvent) => void;
  onError?: (event: CustomTopicStreamErrorEvent) => void;
}

export interface GenerateTopicContentRequest {
  preferred_language: string;
  target_sections?: number;
  force_regenerate?: boolean;
  llm_config?: LLMConfig;
}

export interface TopicContentStreamStartEvent {
  type: "start";
  topic_id: string;
  preferred_language: string;
  target_sections: number;
}

export interface TopicContentStreamProgressEvent {
  type: "progress";
  stage: "analyzing" | "batching" | "ready";
  message: string;
  elapsed_seconds?: number;
  batch_index?: number;
  batch_count?: number;
  generated_sections?: number;
  target_sections?: number;
}

export interface TopicContentStreamSectionEvent {
  type: "section";
  index: number;
  total_sections: number;
  heading: string;
  content: string;
}

export interface TopicContentStreamDoneEvent {
  type: "done";
  topic: TopicDetail;
}

export interface TopicContentStreamErrorEvent {
  type: "error";
  code:
    | "llm_service_approval_required"
    | "study_app_llm_not_assigned"
    | "personal_credential_required"
    | "generation_failed";
  message: string;
}

export type TopicContentStreamEvent =
  | TopicContentStreamStartEvent
  | TopicContentStreamProgressEvent
  | TopicContentStreamSectionEvent
  | TopicContentStreamDoneEvent
  | TopicContentStreamErrorEvent;

export interface TopicContentStreamHandlers {
  onStart?: (event: TopicContentStreamStartEvent) => void;
  onProgress?: (event: TopicContentStreamProgressEvent) => void;
  onSection?: (event: TopicContentStreamSectionEvent) => void;
  onDone?: (event: TopicContentStreamDoneEvent) => void;
  onError?: (event: TopicContentStreamErrorEvent) => void;
}

export interface QuestionAnswer {
  question: string;
  answer: string;
  difficulty: "easy" | "medium" | "hard";
}

export interface QuestionAnswerV2 {
  question_id: string;
  topic_id: string;
  question: string;
  answer: string;
  difficulty: "easy" | "medium" | "hard";
  learning_objective: string;
  source_section: string;
  source_quote: string;
  reasoning_summary: string;
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
  requested_total_count?: number;
  existing_questions?: string[];
  difficulty?: string;
  level?: InterviewLevel | null;
  response_detail?: ResponseDetail;
  preferred_language?: string;
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

export interface GenerateQuestionsV2Response {
  topic_id: string;
  topic_title: string;
  questions: QuestionAnswerV2[];
  provider_used: string;
  model_used: string;
  retries_used: number;
  malformed_items_dropped: number;
}

export interface GenerateQuestionsStreamStartEvent {
  type: "start";
  topic_id: string;
  topic_title: string;
  target_count: number;
}

export interface GenerateQuestionsStreamQuestionEvent {
  type: "question";
  question: QuestionAnswerV2;
}

export interface GenerateQuestionsStreamDoneEvent {
  type: "done";
  topic_id: string;
  topic_title: string;
  generated_count: number;
  provider_used: string;
  model_used: string;
  retries_used: number;
  malformed_items_dropped: number;
}

export interface GenerateQuestionsStreamErrorEvent {
  type: "error";
  code:
    | "llm_service_approval_required"
    | "study_app_llm_not_assigned"
    | "personal_credential_required"
    | "generation_failed";
  message: string;
}

export type GenerateQuestionsStreamEvent =
  | GenerateQuestionsStreamStartEvent
  | GenerateQuestionsStreamQuestionEvent
  | GenerateQuestionsStreamDoneEvent
  | GenerateQuestionsStreamErrorEvent;

export interface GenerateQuestionsStreamHandlers {
  onStart?: (event: GenerateQuestionsStreamStartEvent) => void;
  onQuestion?: (event: GenerateQuestionsStreamQuestionEvent) => void;
  onDone?: (event: GenerateQuestionsStreamDoneEvent) => void;
  onError?: (event: GenerateQuestionsStreamErrorEvent) => void;
}

export interface GenerateQuizStreamStartEvent {
  type: "start";
  target_count: number;
  topics_used: string[];
  question_types: QuizQuestionType[];
}

export interface GenerateQuizStreamProgressEvent {
  type: "progress";
  stage: "generating" | "recovering";
  message: string;
  generated_count: number;
  target_count: number;
}

export interface GenerateQuizStreamQuestionEvent {
  type: "question";
  question: QuizQuestionV2;
}

export interface GenerateQuizStreamDoneEvent {
  type: "done";
  generated_count: number;
  target_count: number;
  topics_used: string[];
  provider_used: string;
  model_used: string;
  retries_used: number;
  malformed_items_dropped: number;
}

export interface GenerateQuizStreamErrorEvent {
  type: "error";
  code:
    | "llm_service_approval_required"
    | "study_app_llm_not_assigned"
    | "personal_credential_required"
    | "generation_failed";
  message: string;
}

export type GenerateQuizStreamEvent =
  | GenerateQuizStreamStartEvent
  | GenerateQuizStreamProgressEvent
  | GenerateQuizStreamQuestionEvent
  | GenerateQuizStreamDoneEvent
  | GenerateQuizStreamErrorEvent;

export interface GenerateQuizStreamHandlers {
  onStart?: (event: GenerateQuizStreamStartEvent) => void;
  onProgress?: (event: GenerateQuizStreamProgressEvent) => void;
  onQuestion?: (event: GenerateQuizStreamQuestionEvent) => void;
  onDone?: (event: GenerateQuizStreamDoneEvent) => void;
  onError?: (event: GenerateQuizStreamErrorEvent) => void;
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

export type UserAuthMode = "api_key" | "account";
export type UserLLMSource = "personal" | "study_app";
export type UserSettingsProvider = "google" | "openai" | "anthropic" | "groq";

export interface UserPreferences {
  provider: UserSettingsProvider;
  model: string;
  temperature: number;
  max_tokens: number;
  auth_mode: UserAuthMode;
  llm_source: UserLLMSource;
  require_answer_reveal: boolean;
}

export interface ProviderConnectionStatus {
  provider: UserSettingsProvider;
  models: string[];
  supports_account_connect: boolean;
  api_key_connected: boolean;
  account_connected: boolean;
  using_backend_fallback: boolean;
  backend_fallback_eligible: boolean;
}

export interface UserSettingsResponse {
  has_saved_preferences: boolean;
  preferences: UserPreferences;
  providers: ProviderConnectionStatus[];
  study_app_available: boolean;
  study_app_assignment: StudyAppAssignment | null;
}

export interface StudyAppAssignment {
  provider: UserSettingsProvider;
  model: string;
  updated_at: string;
}

export interface LLMAssignmentUserItem {
  login_provider: "google" | "github";
  identifier: string;
  identity_key: string;
  is_backend_approved: boolean;
  assignment: StudyAppAssignment | null;
}

export interface LLMAssignmentsUsersResponse {
  users: LLMAssignmentUserItem[];
  provider_models: Record<UserSettingsProvider, string[]>;
}

export interface LLMMyAssignmentResponse {
  login_provider: "google" | "github";
  identifier: string;
  identity_key: string;
  is_backend_approved: boolean;
  assignment: StudyAppAssignment | null;
  provider_models: Record<UserSettingsProvider, string[]>;
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
  topic_id?: string;
  topic_title?: string;
  topic_track?: string;
  section_title?: string;
  mode?: "study" | "quiz";
  response_detail?: ResponseDetail;
  preferred_language?: string;
  requires_programming?: boolean;
  user_message: string;
  history: ChatMessage[];
  conversation_id?: string;
  use_memory?: boolean;
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

export interface QuizQuestionV2 {
  question_id: string;
  question: string;
  type: QuizQuestionType;
  choices: QuizChoice[];
  correct_answer: string;
  explanation: string;
  difficulty: string;
  topic_id: string;
  source_quote: string;
  reasoning_summary: string;
}

export interface GenerateQuizRequest {
  topic_ids: string[];
  count: number;
  question_types: QuizQuestionType[];
  difficulty?: string;
  level?: InterviewLevel | null;
  response_detail?: ResponseDetail;
  preferred_language?: string;
  llm_config?: LLMConfig;
}

export interface TopicPreferencesResponse {
  topic_id: string;
  response_detail: ResponseDetail;
  preferred_language: string;
  requires_programming: boolean;
  language_options: string[];
}

export interface TopicPreferencesUpdateRequest {
  response_detail?: ResponseDetail;
  preferred_language?: string;
}

export interface GenerateQuizResponse {
  questions: QuizQuestion[];
  topics_used: string[];
  provider_used: string;
  model_used: string;
}

export interface GenerateQuizV2Response {
  questions: QuizQuestionV2[];
  topics_used: string[];
  provider_used: string;
  model_used: string;
  retries_used: number;
  malformed_items_dropped: number;
}

export type LearningMode = "study" | "quiz";

export interface LearningAttemptRequest {
  question_id: string;
  topic_id: string;
  user_answer: string;
  is_correct: boolean;
  confidence: number;
  response_time_ms: number;
  mode: LearningMode;
}

export interface LearningAttemptResponse {
  attempt_id: number;
  topic_id: string;
  question_id: string;
  mastery_score: number;
  due_at: string;
  review_bucket: number;
}

export interface ReviewQueueItem {
  topic_id: string;
  question_id: string;
  due_at: string;
  mastery_score: number;
  last_confidence: number;
  review_bucket: number;
  attempts: number;
  state?: string;
  lapses?: number;
  suspended?: number;
  leech?: boolean;
}

export interface ReviewQueueResponse {
  items: ReviewQueueItem[];
  total_due: number;
}

export interface WeakAreaItem {
  topic_id: string;
  attempts: number;
  accuracy: number;
  avg_confidence: number;
  mastery_score: number;
  due_count: number;
}

export interface WeakAreasResponse {
  weak_areas: WeakAreaItem[];
}

export interface TopicMasteryItem {
  topic_id: string;
  mastery_score: number;
  attempts: number;
}

export interface TopicMasteryResponse {
  topics: TopicMasteryItem[];
}

export interface ProgressQuestion {
  question_id?: string;
  question: string;
  difficulty?: string;
  revealed?: boolean;
  is_correct?: boolean | null;
  confidence?: number;
}

export interface SaveProgressPayload {
  topic_title?: string;
  questions: ProgressQuestion[];
  sections?: string[];
  preferred_language?: string;
}

export interface TopicProgressDocument {
  user_id: string;
  topic_id: string;
  topic_title: string;
  summary_text: string;
  summary_status: "empty" | "pending" | "ready" | "failed" | string;
  summary_error: string;
  sections: string[];
  attempt_stats: Record<string, unknown>;
  preferred_language: string;
  question_count: number;
  revision: number;
  provider_used: string;
  model_used: string;
  generation_source: string;
  created_at: string;
  updated_at: string;
  questions_asked: string[];
}

export interface TopicProgressListResponse {
  topics: TopicProgressDocument[];
  total: number;
}

export interface LearnerProfile {
  target_role: string;
  target_date: string;
  weekly_minutes: number;
  preferred_session_minutes: number;
  current_level: InterviewLevel;
  target_level: InterviewLevel;
  primary_track: LearningTrack;
  focus_topic_ids: string[];
  target_companies: string[];
  preferred_modalities: Array<"study" | "quiz" | "interview" | "voice">;
  confidence_by_track: Partial<Record<LearningTrack, number>>;
  created_at: string;
  updated_at: string;
  diagnostic_updated_at: string;
}

export interface LearnerProfileUpdateRequest {
  target_role?: string;
  target_date?: string;
  weekly_minutes?: number;
  preferred_session_minutes?: number;
  current_level?: InterviewLevel;
  target_level?: InterviewLevel;
  primary_track?: LearningTrack;
  focus_topic_ids?: string[];
  target_companies?: string[];
  preferred_modalities?: Array<"study" | "quiz" | "interview" | "voice">;
  confidence_by_track?: Partial<Record<LearningTrack, number>>;
}

export interface CompetencyScore {
  competency_id: string;
  competency_type: "topic" | "track" | "interview_dimension" | string;
  label: string;
  track: string;
  score: number;
  confidence_gap: number;
  evidence_count: number;
  priority: number;
  source: string;
  metadata: Record<string, unknown>;
  updated_at: string;
}

export interface ProfileDiagnosticResponse {
  generated_at: string;
  profile: LearnerProfile;
  readiness_score: number;
  strongest_competencies: string[];
  urgent_competencies: string[];
  competencies: CompetencyScore[];
}

export type RecommendationType =
  | "review"
  | "study"
  | "quiz"
  | "interview"
  | "voice";

export interface RecommendationItem {
  recommendation_id: string;
  recommendation_type: RecommendationType;
  topic_id: string;
  title: string;
  reason: string;
  estimated_minutes: number;
  cta_route: string;
  priority: number;
  reason_codes: string[];
  track: string;
}

export interface RecommendationsResponse {
  generated_at: string;
  profile: LearnerProfile;
  days_until_target: number | null;
  items: RecommendationItem[];
}

export type StudyPlanTaskType =
  | "review"
  | "topic_study"
  | "quiz"
  | "interview";

export interface StudyPlanTask {
  task_type: StudyPlanTaskType;
  topic_id: string;
  title: string;
  reason: string;
  estimated_minutes: number;
  cta_route: string;
}

export interface StudyPlanDay {
  day_index: number;
  label: string;
  date: string;
  tasks: StudyPlanTask[];
}

export interface StudyPlanResponse {
  generated_at: string;
  days: number;
  daily_items: number;
  total_tasks: number;
  days_plan: StudyPlanDay[];
}

/* ── Mock Interview Types ────────────────────────────────── */

export interface InterviewRubricScore {
  technical_accuracy: number;
  reasoning_depth: number;
  communication_clarity: number;
  completeness: number;
  confidence_signal: number;
  overall: number;
  degraded?: boolean;
}

export interface InterviewRubricAverages {
  technical_accuracy: number;
  reasoning_depth: number;
  communication_clarity: number;
  completeness: number;
  confidence_signal: number;
  overall: number;
}

export interface InterviewSession {
  session_id: string;
  track: LearningTrack;
  level: InterviewLevel;
  interview_type: InterviewType;
  turn_count: number;
  turns_completed: number;
  status: "active" | "completed";
  target_role: string;
  interviewer_style: InterviewerStyle;
  feedback_mode: FeedbackMode;
  focus_areas: string[];
  created_at: string;
  updated_at: string;
  current_question: string;
  report_ready: boolean;
}

export interface InterviewTurn {
  session_id: string;
  turn_index: number;
  question: string;
  user_answer: string;
  rubric: InterviewRubricScore;
  strengths: string[];
  improvements: string[];
  follow_up_note: string;
  response_time_ms: number;
  created_at: string;
  degraded?: boolean;
}

export interface InterviewReport {
  session_id: string;
  overall_score: number;
  readiness_label: string;
  completed_turns: number;
  rubric_averages: InterviewRubricAverages;
  weak_competencies: string[];
  strengths: string[];
  recommended_topic_ids: string[];
  next_steps: string[];
  summary: string;
  created_at: string;
  updated_at: string;
}

export interface CreateInterviewSessionRequest {
  track: LearningTrack;
  level: InterviewLevel;
  interview_type: InterviewType;
  turn_count: number;
  target_role: string;
  interviewer_style?: InterviewerStyle;
  feedback_mode?: FeedbackMode;
  job_description_text: string;
  resume_summary_text: string;
  focus_areas: string[];
  llm_config?: LLMConfig;
}

export interface SubmitInterviewAnswerRequest {
  user_answer: string;
  response_time_ms: number;
  llm_config?: LLMConfig;
}

export interface NextInterviewQuestionRequest {
  llm_config?: LLMConfig;
}

export interface InterviewSessionResponse {
  session: InterviewSession;
  turns: InterviewTurn[];
}

export interface InterviewTurnResponse {
  session: InterviewSession;
  turn: InterviewTurn;
  report_ready: boolean;
  degraded?: boolean;
}

export interface InterviewQuestionResponse {
  session_id: string;
  turn_index: number;
  question: string;
  competency_focus: string;
  expected_signals: string[];
}

export interface InterviewReportResponse {
  session: InterviewSession;
  report: InterviewReport;
}

export interface InterviewSessionsListResponse {
  sessions: InterviewSession[];
  total: number;
}

export interface InterviewStatsResponse {
  total_sessions: number;
  completed_sessions: number;
  interview_readiness_score: number;
}

export interface InterviewTrendPoint {
  session_id: string;
  completed_at: string;
  overall_score: number;
  rubric_averages: InterviewRubricAverages;
  track: LearningTrack;
  level: InterviewLevel;
  interview_type: InterviewType;
  readiness_label: string;
}

export interface InterviewTrendsSummary {
  latest_score: number;
  previous_score: number;
  delta: number;
  session_count: number;
}

export interface InterviewTrendsResponse {
  points: InterviewTrendPoint[];
  summary: InterviewTrendsSummary;
}

export interface InterviewStreamStartEvent {
  type: "start";
  stage: string;
  message: string;
  session_id?: string;
}

export interface InterviewStreamProgressEvent {
  type: "progress";
  stage: string;
  message: string;
  elapsed_seconds: number;
  session_id?: string;
}

export interface InterviewStreamErrorEvent {
  type: "error";
  code:
    | "llm_service_approval_required"
    | "study_app_llm_not_assigned"
    | "personal_credential_required"
    | "generation_failed";
  message: string;
}

export interface InterviewSessionStreamDoneEvent
  extends InterviewSessionResponse {
  type: "done";
}

export interface InterviewQuestionStreamDoneEvent
  extends InterviewQuestionResponse {
  type: "done";
}

export interface InterviewTurnStreamDoneEvent extends InterviewTurnResponse {
  type: "done";
}

export type InterviewSessionStreamEvent =
  | InterviewStreamStartEvent
  | InterviewStreamProgressEvent
  | InterviewSessionStreamDoneEvent
  | InterviewStreamErrorEvent;

export type InterviewQuestionStreamEvent =
  | InterviewStreamStartEvent
  | InterviewStreamProgressEvent
  | InterviewQuestionStreamDoneEvent
  | InterviewStreamErrorEvent;

export type InterviewTurnStreamEvent =
  | InterviewStreamStartEvent
  | InterviewStreamProgressEvent
  | InterviewTurnStreamDoneEvent
  | InterviewStreamErrorEvent;

export interface InterviewSessionStreamHandlers {
  onStart?: (event: InterviewStreamStartEvent) => void;
  onProgress?: (event: InterviewStreamProgressEvent) => void;
  onDone?: (event: InterviewSessionStreamDoneEvent) => void;
  onError?: (event: InterviewStreamErrorEvent) => void;
}

export interface InterviewQuestionStreamHandlers {
  onStart?: (event: InterviewStreamStartEvent) => void;
  onProgress?: (event: InterviewStreamProgressEvent) => void;
  onDone?: (event: InterviewQuestionStreamDoneEvent) => void;
  onError?: (event: InterviewStreamErrorEvent) => void;
}

export interface InterviewTurnStreamHandlers {
  onStart?: (event: InterviewStreamStartEvent) => void;
  onProgress?: (event: InterviewStreamProgressEvent) => void;
  onDone?: (event: InterviewTurnStreamDoneEvent) => void;
  onError?: (event: InterviewStreamErrorEvent) => void;
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

/* ── Voice Types ─────────────────────────────────────────── */

export type VoiceTier = "browser" | "cloud";

export interface VoiceConfig {
  enabled: boolean;
  available_tiers: VoiceTier[];
  default_tier: VoiceTier;
  user_tier: VoiceTier;
  stt_provider: string;
  tts_provider: string;
}

export type VoiceSessionType = "chat" | "interview" | "qa";

export interface VoiceLatency {
  stt_ms: number;
  llm_ms: number;
  tts_ms: number;
  total_ms: number;
}

/** Messages FROM the server over WebSocket */
export type VoiceServerMessage =
  | { type: "ready"; session_id: string; tier: VoiceTier }
  | { type: "transcript"; text: string; final: boolean }
  | { type: "response"; text: string; audio: string; latency: VoiceLatency }
  | { type: "audio"; audio: string; text: string }
  | { type: "stopped"; session_id: string }
  | { type: "error"; message: string };

/** Messages TO the server over WebSocket */
export type VoiceClientMessage =
  | {
      type: "start";
      tier: VoiceTier;
      session_type: VoiceSessionType;
      session_id?: string;
      llm_config?: { provider: string; model: string };
    }
  | { type: "audio"; data: string; mime: string }
  | { type: "text"; content: string; speak?: boolean }
  | { type: "synthesize"; text: string }
  | { type: "stop" };

// ── FSRS Spaced Repetition ──────────────────────────
export type LearningReviewRating = "again" | "hard" | "good" | "easy";

export type LearningReviewSourceType =
  | "question"
  | "flashcard"
  | "interview"
  | "exam";

export interface LearningReviewRequest {
  card_id: string;
  topic_id: string;
  rating: LearningReviewRating;
  response_time_ms?: number;
  source_type?: LearningReviewSourceType;
}

export interface FSRSCardResponse {
  card_id: string;
  topic_id: string;
  source_type: string;
  state: string;
  stability: number | null;
  difficulty: number | null;
  due_at: string;
  last_review_at: string | null;
  reps: number;
  lapses: number;
  scheduled_days: number;
  elapsed_days: number;
  suspended: number;
  created_at: string;
  updated_at: string;
}

export interface FSRSReviewLogResponse {
  id: number;
  card_id: string;
  rating: string;
  state: string;
  review_duration_ms: number;
  scheduled_days: number | null;
  elapsed_days: number | null;
  created_at: string;
}

export interface LearningReviewResponse {
  card: FSRSCardResponse;
  log: FSRSReviewLogResponse;
  leech: boolean;
}

export interface ForecastDayItem {
  date: string;
  count: number;
}

export interface ForecastResponse {
  due_counts: ForecastDayItem[];
  total_due: number;
}

export interface CalibrationResponse {
  eligible_reviews: number;
  successful_reviews: number;
  true_retention: number;
  target_band_low: number;
  target_band_high: number;
  within_band: boolean;
  good_rating_pct: number;
  low_signal: boolean;
}

/* ── Documents + RAG Chat Types ────────────────────── */

export type DocumentStatus = "uploaded" | "processing" | "ready" | "failed";

export interface DocumentUploadResponse {
  document_id: string;
  status: string;
}

export interface DocumentDetail {
  document_id: string;
  filename: string;
  mime_type: string;
  title: string;
  status: DocumentStatus;
  error: string;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  documents: DocumentDetail[];
}

export interface Citation {
  document_id: string;
  chunk_index: number;
  quote: string;
}

export interface ChatAskRequest {
  message: string;
  document_ids?: string[];
  topic_id?: string;
  conversation_id?: string;
  llm_config?: LLMConfig;
}

export interface ChatAskResponse {
  answer: string;
  citations: Citation[];
  conversation_id: string;
}

/* ── Flashcard Decks + Anki Interop ────────── */

export interface DeckCreateRequest {
  name: string;
  description?: string;
}

export interface DeckUpdateRequest {
  name: string;
  description?: string;
}

export interface DeckResponse {
  deck_id: string;
  name: string;
  description: string;
  card_count: number;
  due_count: number;
  created_at: string;
  updated_at: string;
}

export interface DeckListResponse {
  decks: DeckResponse[];
}

export interface CardCreateRequest {
  front: string;
  back: string;
  tags?: string[];
}

export interface CardUpdateRequest {
  front: string;
  back: string;
  tags?: string[];
}

export interface FSRSCardState {
  state: string;
  stability: number | null;
  difficulty: number | null;
  due_at: string;
  last_review_at: string | null;
  reps: number;
  lapses: number;
  scheduled_days: number;
  elapsed_days: number;
  suspended: number;
}

export interface FlashcardResponse {
  card_id: string;
  deck_id: string;
  front: string;
  back: string;
  source_ref: string;
  tags: string[];
  fsrs_card_id: string;
  suspended: number;
  created_at: string;
  updated_at: string;
  fsrs: FSRSCardState | null;
}

export interface FlashcardListResponse {
  cards: FlashcardResponse[];
  total: number;
}

export type GenerateCardsSourceType = "topic" | "section" | "document";

export interface GenerateCardsRequest {
  source_type: GenerateCardsSourceType;
  topic_id?: string;
  section_title?: string;
  document_id?: string;
  count: number;
  llm_config?: LLMConfig;
}

export interface GenerateCardsResponse {
  job_id: string;
  status: string;
}

export interface GeneratedCardItem {
  card_id: string;
  front: string;
  back: string;
  source_section: string;
  source_quote: string;
}

export interface GenerateCardsJobResponse {
  job_id: string;
  status: string;
  cards: GeneratedCardItem[];
  retries_used: number;
  malformed_items_dropped: number;
  error: string;
}

export interface ImportDeckResponse {
  deck_id: string;
  deck_name: string;
  imported: number;
  skipped_duplicates: number;
  errors: string[];
}

export interface FindDuplicatesRequest {
  deck_id?: string;
  threshold?: number;
}

export interface DuplicateCardRef {
  card_id: string;
  deck_id: string;
  front: string;
}

export interface DuplicatePair {
  card_a: DuplicateCardRef;
  card_b: DuplicateCardRef;
  similarity: number;
  jaccard: number;
  cosine: number | null;
}

export interface FindDuplicatesResponse {
  pairs: DuplicatePair[];
  threshold: number;
}

export interface DeckReviewRequest {
  card_id: string;
  rating: LearningReviewRating;
  response_time_ms?: number;
}

export interface DeckReviewResponse {
  card: FlashcardResponse;
  log: FSRSReviewLogResponse;
  leech: boolean;
}

export interface StudySessionResponse {
  deck_id: string;
  cards: FlashcardResponse[];
  total_due: number;
}
