import { useState, useEffect, useMemo, useCallback } from "react";
import { useParams, useLocation, Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronLeft,
  ChevronRight,
  RotateCcw,
  Trophy,
  X,
  Loader2,
  CheckCircle2,
  XCircle,
  Sparkles,
  ListChecks,
  ChevronDown,
  AlertTriangle,
} from "lucide-react";
import {
  scaleInVariants,
  pageVariants,
  pageTransition,
  containerVariants,
  cardVariants,
  expandVariants,
} from "@/utils/animations";
import {
  fetchTopics,
  fetchTopic,
  generateQuiz,
  generateQuizV2,
  recordLearningAttempt,
  fetchWeakAreas,
} from "@/services/api";
import { useSettingsStore } from "@/store/settingsStore";
import { useProgressStore } from "@/store/progressStore";
import DifficultyBadge from "@/components/common/DifficultyBadge";
import WordHighlightChat from "@/components/common/WordHighlightChat";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import type {
  InterviewLevel,
  LearningTrack,
  ResponseDetail,
  TopicSummary,
  QuizQuestionType,
  QuizQuestionV2,
  WeakAreaItem,
} from "@/types";

type QuizState = "setup" | "playing" | "results";

const TRACK_LABELS: Record<LearningTrack, string> = {
  backend: "Backend",
  frontend: "Frontend",
  system_design: "System Design",
  ai_stack: "AI Stack",
};

export default function QuizMode() {
  const { topicId } = useParams<{ topicId: string }>();
  const location = useLocation();
  const preselectedTopic = topicId || "";

  const [quizState, setQuizState] = useState<QuizState>("setup");
  const [topics, setTopics] = useState<TopicSummary[]>([]);
  const [selectedTopics, setSelectedTopics] = useState<Set<string>>(
    preselectedTopic ? new Set([preselectedTopic]) : new Set(),
  );
  const [questionCount, setQuestionCount] = useState(10);
  const [questionTypes, setQuestionTypes] = useState<Set<QuizQuestionType>>(
    new Set(["mcq", "true_false"]),
  );
  const [difficulty, setDifficulty] = useState("");
  const [level, setLevel] = useState<InterviewLevel>("mid");
  const [generating, setGenerating] = useState(false);
  const [loadingTopics, setLoadingTopics] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");

  const [questions, setQuestions] = useState<QuizQuestionV2[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [answered, setAnswered] = useState(false);
  const [answerConfidence, setAnswerConfidence] = useState(3);
  const [answers, setAnswers] = useState<
    Record<number, { selected: string; correct: boolean; confidence: number }>
  >({});
  const [questionStartMs, setQuestionStartMs] = useState<Record<number, number>>(
    {},
  );
  const [providerInfo, setProviderInfo] = useState({
    provider: "",
    model: "",
    retries: 0,
    malformed: 0,
  });

  const [showReview, setShowReview] = useState(false);
  const [expandedReview, setExpandedReview] = useState<number | null>(null);
  const [weakAreas, setWeakAreas] = useState<WeakAreaItem[]>([]);
  const [loadingWeakAreas, setLoadingWeakAreas] = useState(false);
  const [topicAiById, setTopicAiById] = useState<
    Record<
      string,
      {
        responseDetail: ResponseDetail;
        preferredLanguage: string;
        requiresProgramming: boolean;
      }
    >
  >({});

  const settings = useSettingsStore();
  const { recordAttempt, setMastery } = useProgressStore();

  const current = questions[currentIdx];
  const topicsById = useMemo(
    () => new Map(topics.map((topic) => [topic.id, topic])),
    [topics],
  );
  const currentTopic = current ? topicsById.get(current.topic_id) : undefined;
  const currentTopicAi = current ? topicAiById[current.topic_id] : undefined;
  const currentContextAnswer = current
    ? [
        current.explanation,
        current.source_quote ? `Source quote: ${current.source_quote}` : "",
      ]
        .filter(Boolean)
        .join("\n\n")
    : "";
  const total = questions.length;
  const correctCount = useMemo(
    () => Object.values(answers).filter((a) => a.correct).length,
    [answers],
  );

  useEffect(() => {
    fetchTopics()
      .then(setTopics)
      .catch(() => setErrorMsg("Failed to load topics."))
      .finally(() => setLoadingTopics(false));
  }, []);

  const passedQuestions = (location.state as { questions?: unknown[] })
    ?.questions;
  useEffect(() => {
    if (preselectedTopic && !passedQuestions) {
      setSelectedTopics(new Set([preselectedTopic]));
    }
  }, [preselectedTopic, passedQuestions]);

  useEffect(() => {
    if (quizState !== "results") return;
    setLoadingWeakAreas(true);
    fetchWeakAreas(5)
      .then((res) => setWeakAreas(res.weak_areas))
      .catch(() => setWeakAreas([]))
      .finally(() => setLoadingWeakAreas(false));
  }, [quizState]);

  const toggleTopic = (id: string) => {
    setSelectedTopics((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleQuestionType = (type: QuizQuestionType) => {
    setQuestionTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        if (next.size > 1) next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  };

  const handleStartQuiz = useCallback(async () => {
    if (selectedTopics.size === 0) return;
    setGenerating(true);
    setErrorMsg("");
    try {
      const selectedTopicIds = Array.from(selectedTopics);
      const topicDetails = await Promise.all(
        selectedTopicIds.map(async (id) => {
          try {
            const detail = await fetchTopic(id);
            return [id, detail] as const;
          } catch {
            return null;
          }
        }),
      );
      const nextTopicAiById: Record<
        string,
        {
          responseDetail: ResponseDetail;
          preferredLanguage: string;
          requiresProgramming: boolean;
        }
      > = {};
      for (const entry of topicDetails) {
        if (!entry) continue;
        const [id, detail] = entry;
        nextTopicAiById[id] = {
          responseDetail: detail.response_detail || "concise",
          preferredLanguage: detail.selected_language || "",
          requiresProgramming: !!detail.requires_programming,
        };
      }
      setTopicAiById(nextTopicAiById);

      const singleTopicSettings =
        selectedTopicIds.length === 1 ? nextTopicAiById[selectedTopicIds[0]] : undefined;
      const req = {
        topic_ids: selectedTopicIds,
        count: questionCount,
        question_types: Array.from(questionTypes),
        difficulty: difficulty || undefined,
        level,
        response_detail: singleTopicSettings?.responseDetail,
        preferred_language:
          singleTopicSettings?.requiresProgramming && singleTopicSettings?.preferredLanguage
            ? singleTopicSettings.preferredLanguage
            : undefined,
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
      };

      try {
        const res = await generateQuizV2(req);
        if (res.questions.length === 0) {
          setErrorMsg("No valid quiz questions generated. Try again.");
          return;
        }
        setQuestions(res.questions);
        setProviderInfo({
          provider: res.provider_used,
          model: res.model_used,
          retries: res.retries_used,
          malformed: res.malformed_items_dropped,
        });
      } catch {
        const legacy = await generateQuiz(req);
        const upgraded: QuizQuestionV2[] = legacy.questions.map((q, idx) => ({
          question_id: `quiz-legacy-${idx}`,
          question: q.question,
          type: q.type,
          choices: q.choices,
          correct_answer: q.correct_answer,
          explanation: q.explanation,
          difficulty: q.difficulty,
          topic_id: q.topic_id || req.topic_ids[0] || "",
          source_quote: "Legacy mode did not return source quote.",
          reasoning_summary: "Review the explanation to reinforce reasoning.",
        }));
        if (upgraded.length === 0) {
          setErrorMsg("No quiz questions generated. Try again.");
          return;
        }
        setQuestions(upgraded);
        setProviderInfo({
          provider: legacy.provider_used,
          model: legacy.model_used,
          retries: 0,
          malformed: 0,
        });
      }

      setCurrentIdx(0);
      setAnswers({});
      setSelectedAnswer(null);
      setAnswered(false);
      setAnswerConfidence(3);
      setQuestionStartMs({ 0: Date.now() });
      setQuizState("playing");
    } catch (err) {
      console.error(err);
      setErrorMsg("Quiz generation failed. Check settings and retry.");
    } finally {
      setGenerating(false);
    }
  }, [selectedTopics, questionCount, questionTypes, difficulty, level, settings]);

  const handleSubmitAnswer = async () => {
    if (!selectedAnswer || !current) return;
    const correct = selectedAnswer === current.correct_answer;
    setAnswers((prev) => ({
      ...prev,
      [currentIdx]: {
        selected: selectedAnswer,
        correct,
        confidence: answerConfidence,
      },
    }));
    setAnswered(true);

    try {
      const responseTime = Math.max(
        0,
        Date.now() - (questionStartMs[currentIdx] || Date.now()),
      );
      const topicForAttempt = current.topic_id || Array.from(selectedTopics)[0] || "";
      const res = await recordLearningAttempt({
        question_id: current.question_id,
        topic_id: topicForAttempt,
        user_answer: selectedAnswer,
        is_correct: correct,
        confidence: answerConfidence,
        response_time_ms: responseTime,
        mode: "quiz",
      });
      if (topicForAttempt) {
        recordAttempt(topicForAttempt, correct, answerConfidence);
        setMastery(topicForAttempt, res.mastery_score);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg("Answer recorded locally, but syncing attempt failed.");
    }
  };

  const handleNext = () => {
    if (currentIdx < total - 1) {
      const nextIdx = currentIdx + 1;
      setCurrentIdx(nextIdx);
      setSelectedAnswer(answers[nextIdx]?.selected || null);
      setAnswered(!!answers[nextIdx]);
      setAnswerConfidence(answers[nextIdx]?.confidence || 3);
      setQuestionStartMs((prev) =>
        prev[nextIdx] ? prev : { ...prev, [nextIdx]: Date.now() },
      );
    } else {
      setQuizState("results");
    }
  };

  const handleRestart = () => {
    setQuizState("setup");
    setQuestions([]);
    setAnswers({});
    setCurrentIdx(0);
    setSelectedAnswer(null);
    setAnswered(false);
    setAnswerConfidence(3);
    setShowReview(false);
    setExpandedReview(null);
    setWeakAreas([]);
    setErrorMsg("");
  };

  const percentage = total > 0 ? Math.round((correctCount / total) * 100) : 0;

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
      className="min-h-[80vh] flex flex-col"
    >
      {quizState === "setup" && (
        <>
          <div className="bg-udemy-dark text-white">
            <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-8">
              <Link
                to={preselectedTopic ? `/topics/${preselectedTopic}` : "/"}
                className="inline-flex items-center gap-1 text-sm text-gray-400 hover:text-white mb-3 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Back
              </Link>
              <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
                <ListChecks className="w-7 h-7 text-udemy-purple-light" />
                Quiz Mode
              </h1>
              <p className="text-gray-400 mt-2">
                Adaptive interview quiz with attempt tracking and weak-area feedback
              </p>
            </div>
          </div>

          <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-8 flex-1">
            {errorMsg && (
              <div className="mb-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <div>{errorMsg}</div>
              </div>
            )}
            <motion.div
              className="grid grid-cols-1 lg:grid-cols-3 gap-6"
              variants={containerVariants}
              initial="hidden"
              animate="show"
            >
              <motion.div variants={cardVariants} className="lg:col-span-2">
                <div className="udemy-card p-6">
                  <h2 className="text-lg font-bold mb-4">Select Topics</h2>
                  <p className="text-sm text-udemy-text-muted mb-4">
                    Choose one or more topics for this quiz.
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-[400px] overflow-y-auto pr-1">
                    {loadingTopics
                      ? Array.from({ length: 6 }).map((_, i) => (
                          <div key={i} className="skeleton h-16 rounded-lg" />
                        ))
                      : topics.map((t) => (
                          <button
                            key={t.id}
                            onClick={() => toggleTopic(t.id)}
                            className={`p-3 rounded-lg border-2 text-left transition-all flex items-start gap-3 ${
                              selectedTopics.has(t.id)
                                ? "border-udemy-purple bg-udemy-purple/5"
                                : "border-udemy-border hover:border-gray-400"
                            }`}
                          >
                            <div
                              className={`w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors ${
                                selectedTopics.has(t.id)
                                  ? "bg-udemy-purple border-udemy-purple"
                                  : "border-gray-300"
                              }`}
                            >
                              {selectedTopics.has(t.id) && (
                                <CheckCircle2 className="w-3.5 h-3.5 text-white" />
                              )}
                            </div>
                            <div className="min-w-0">
                              <p className="text-sm font-medium line-clamp-1">
                                {t.title}
                              </p>
                              <div className="text-xs text-udemy-text-muted line-clamp-1">
                                {t.section_count} sections ·{" "}
                                {TRACK_LABELS[t.track] || t.track}
                              </div>
                            </div>
                          </button>
                        ))}
                  </div>
                </div>
              </motion.div>

              <motion.div variants={cardVariants}>
                <div className="udemy-card p-6 space-y-6">
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Number of Questions
                    </label>
                    <input
                      type="range"
                      min="5"
                      max="30"
                      step="5"
                      value={questionCount}
                      onChange={(e) => setQuestionCount(Number(e.target.value))}
                      className="w-full accent-udemy-purple"
                    />
                    <div className="flex justify-between text-xs text-udemy-text-muted mt-1">
                      <span>5</span>
                      <span className="font-bold text-udemy-purple text-sm">
                        {questionCount}
                      </span>
                      <span>30</span>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Question Types
                    </label>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <button
                        onClick={() => toggleQuestionType("mcq")}
                        className={`flex-1 py-2 px-3 rounded-lg border-2 text-sm font-medium transition-all ${
                          questionTypes.has("mcq")
                            ? "border-udemy-purple bg-udemy-purple/10 text-udemy-purple"
                            : "border-udemy-border text-udemy-text-muted"
                        }`}
                      >
                        Multiple Choice
                      </button>
                      <button
                        onClick={() => toggleQuestionType("true_false")}
                        className={`flex-1 py-2 px-3 rounded-lg border-2 text-sm font-medium transition-all ${
                          questionTypes.has("true_false")
                            ? "border-udemy-purple bg-udemy-purple/10 text-udemy-purple"
                            : "border-udemy-border text-udemy-text-muted"
                        }`}
                      >
                        True / False
                      </button>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Difficulty
                    </label>
                    <select
                      value={difficulty}
                      onChange={(e) => setDifficulty(e.target.value)}
                      className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                    >
                      <option value="">Mixed levels</option>
                      <option value="easy">Easy</option>
                      <option value="medium">Medium</option>
                      <option value="hard">Hard</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Interview Level
                    </label>
                    <select
                      value={level}
                      onChange={(e) => setLevel(e.target.value as InterviewLevel)}
                      className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                    >
                      <option value="junior">Junior</option>
                      <option value="mid">Mid</option>
                      <option value="senior">Senior</option>
                    </select>
                  </div>

                  <button
                    onClick={handleStartQuiz}
                    disabled={selectedTopics.size === 0 || generating}
                    className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {generating ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Sparkles className="w-4 h-4" />
                    )}
                    {generating
                      ? "Generating Quiz..."
                      : `Start Quiz (${selectedTopics.size} topic${selectedTopics.size !== 1 ? "s" : ""})`}
                  </button>

                  <p className="text-xs text-center text-udemy-text-muted">
                    Powered by {providerInfo.provider || settings.provider}
                  </p>
                </div>
              </motion.div>
            </motion.div>
          </div>
        </>
      )}

      {quizState === "playing" && current && (
        <>
          <div className="bg-udemy-dark text-white">
            <div className="max-w-3xl mx-auto px-4 sm:px-6 py-4 flex flex-wrap items-center justify-between gap-2">
              <button
                onClick={() => {
                  if (confirm("Exit quiz? Your progress will be lost.")) {
                    handleRestart();
                  }
                }}
                className="w-full sm:w-auto flex items-center gap-1 text-sm text-gray-400 hover:text-white transition-colors"
              >
                <X className="w-4 h-4" />
                Exit Quiz
              </button>
              <span className="text-sm font-medium">
                {currentIdx + 1} / {total}
              </span>
              <span className="text-sm text-gray-400">{correctCount} correct</span>
            </div>
            <div className="h-1 bg-gray-700">
              <motion.div
                className="h-full bg-udemy-purple"
                animate={{
                  width: `${((currentIdx + (answered ? 1 : 0)) / total) * 100}%`,
                }}
                transition={{ type: "spring", stiffness: 200, damping: 20 }}
              />
            </div>
          </div>

          <div className="flex-1 flex items-center justify-center px-4 sm:px-6 py-10">
            <AnimatePresence mode="wait">
              <motion.div
                key={currentIdx}
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ type: "spring", stiffness: 300, damping: 25 }}
                className="udemy-card max-w-2xl w-full"
              >
                <div className="p-6 md:p-8">
                  {errorMsg && (
                    <div className="mb-4 text-xs text-amber-900 bg-amber-50 border border-amber-300 rounded p-2">
                      {errorMsg}
                    </div>
                  )}
                  <div className="flex items-center gap-3 mb-2">
                    <span className="w-8 h-8 bg-udemy-purple rounded-full flex items-center justify-center text-white text-sm font-bold">
                      {currentIdx + 1}
                    </span>
                    <DifficultyBadge
                      difficulty={current.difficulty as "easy" | "medium" | "hard"}
                    />
                    <span
                      className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                        current.type === "true_false"
                          ? "bg-blue-100 text-blue-700"
                          : "bg-purple-100 text-purple-700"
                      }`}
                    >
                      {current.type === "true_false"
                        ? "True / False"
                        : "Multiple Choice"}
                    </span>
                  </div>

                  <WordHighlightChat
                    contextQuestion={current.question}
                    contextAnswer={currentContextAnswer}
                    sessionKey={`quiz:${current.topic_id || "unknown"}`}
                    topicId={current.topic_id}
                    topicTitle={currentTopic?.title || current.topic_id}
                    topicTrack={currentTopic?.track || ""}
                    mode="quiz"
                    showFloatingTrigger
                    floatingTriggerWord={currentTopic?.title || current.topic_id}
                    floatingTriggerPrompt="Explain this quiz concept in an organized way and include practical tradeoffs."
                    responseDetail={currentTopicAi?.responseDetail || "concise"}
                    preferredLanguage={currentTopicAi?.preferredLanguage || ""}
                    requiresProgramming={!!currentTopicAi?.requiresProgramming}
                    selectionTargetSelector='[data-word-chat-target="true"]'
                    qaKey={`quiz:${current.question_id}`}
                  >
                    <h2
                      data-word-chat-target="true"
                      className="text-lg md:text-xl font-bold leading-relaxed mb-6 mt-4"
                    >
                      {current.question}
                    </h2>

                    <div className="space-y-3 mb-6">
                      {current.choices.map((choice) => {
                        const isSelected = selectedAnswer === choice.label;
                        const isCorrect =
                          answered && choice.label === current.correct_answer;
                        const isWrong =
                          answered &&
                          isSelected &&
                          choice.label !== current.correct_answer;

                        return (
                          <button
                            key={choice.label}
                            onClick={() => {
                              if (!answered) setSelectedAnswer(choice.label);
                            }}
                            disabled={answered}
                            className={`w-full p-4 rounded-lg border-2 text-left transition-all flex items-start gap-3 ${
                              isCorrect
                                ? "border-green-500 bg-green-50"
                                : isWrong
                                  ? "border-red-500 bg-red-50"
                                  : isSelected
                                    ? "border-udemy-purple bg-udemy-purple/5"
                                    : "border-udemy-border hover:border-gray-400"
                            } ${answered ? "cursor-default" : "cursor-pointer"}`}
                          >
                            <span
                              className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold border-2 transition-colors ${
                                isCorrect
                                  ? "bg-green-500 border-green-500 text-white"
                                  : isWrong
                                    ? "bg-red-500 border-red-500 text-white"
                                    : isSelected
                                      ? "bg-udemy-purple border-udemy-purple text-white"
                                      : "border-gray-300 text-gray-500"
                              }`}
                            >
                              {isCorrect ? (
                                <CheckCircle2 className="w-4 h-4" />
                              ) : isWrong ? (
                                <XCircle className="w-4 h-4" />
                              ) : (
                                choice.label
                              )}
                            </span>
                            <span className="text-[15px] leading-relaxed pt-1">
                              {choice.text}
                            </span>
                          </button>
                        );
                      })}
                    </div>

                    {!answered && (
                      <div className="mb-5">
                        <label className="text-xs font-medium text-udemy-text-muted block mb-1">
                          Confidence: {answerConfidence}/5
                        </label>
                        <input
                          type="range"
                          min={1}
                          max={5}
                          step={1}
                          value={answerConfidence}
                          onChange={(e) => setAnswerConfidence(Number(e.target.value))}
                          className="w-full accent-udemy-purple"
                        />
                      </div>
                    )}

                    <AnimatePresence>
                      {answered && current.explanation && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          className="overflow-hidden mb-6"
                        >
                          <div
                            className={`p-4 rounded-lg border-l-4 ${
                              answers[currentIdx]?.correct
                                ? "bg-green-50 border-green-500"
                                : "bg-amber-50 border-amber-500"
                            }`}
                          >
                            <h4 className="text-xs font-bold uppercase text-udemy-text-muted mb-1">
                              Explanation
                            </h4>
                            <div data-word-chat-target="true">
                              <MarkdownRenderer
                                content={current.explanation}
                                className="text-[14px]"
                              />
                            </div>
                            {current.source_quote && (
                              <p
                                data-word-chat-target="true"
                                className="text-xs text-udemy-text-muted mt-2 italic"
                              >
                                Source: &ldquo;{current.source_quote}&rdquo;
                              </p>
                            )}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </WordHighlightChat>

                  <div className="flex items-center justify-between">
                    {!answered ? (
                      <button
                        onClick={handleSubmitAnswer}
                        disabled={!selectedAnswer}
                        className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Submit Answer
                      </button>
                    ) : (
                      <button
                        onClick={handleNext}
                        className="btn-primary flex items-center gap-2"
                      >
                        {currentIdx < total - 1 ? (
                          <>
                            Next <ChevronRight className="w-4 h-4" />
                          </>
                        ) : (
                          <>
                            <Trophy className="w-4 h-4" />
                            See Results
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </>
      )}

      {quizState === "results" && (
        <div className="flex-1 flex flex-col items-center px-4 sm:px-6 py-10">
          <motion.div
            variants={scaleInVariants}
            initial="hidden"
            animate="visible"
            className="udemy-card p-8 max-w-lg w-full text-center mb-8"
          >
            <div className="relative w-32 h-32 mx-auto mb-6">
              <svg className="w-32 h-32 -rotate-90" viewBox="0 0 120 120">
                <circle
                  cx="60"
                  cy="60"
                  r="52"
                  fill="none"
                  stroke="#d1d7dc"
                  strokeWidth="8"
                />
                <motion.circle
                  cx="60"
                  cy="60"
                  r="52"
                  fill="none"
                  stroke={
                    percentage >= 70
                      ? "#1e6055"
                      : percentage >= 40
                        ? "#b4690e"
                        : "#b32d0f"
                  }
                  strokeWidth="8"
                  strokeLinecap="round"
                  strokeDasharray={2 * Math.PI * 52}
                  initial={{ strokeDashoffset: 2 * Math.PI * 52 }}
                  animate={{
                    strokeDashoffset: 2 * Math.PI * 52 * (1 - percentage / 100),
                  }}
                  transition={{ duration: 1.2, ease: "easeOut", delay: 0.2 }}
                />
              </svg>
              <span className="absolute inset-0 flex items-center justify-center text-3xl font-bold">
                {percentage}%
              </span>
            </div>

            <h2 className="text-2xl font-bold mb-2">
              {percentage >= 80
                ? "Outstanding"
                : percentage >= 60
                  ? "Great work"
                  : percentage >= 40
                    ? "Good effort"
                    : "Keep studying"}
            </h2>
            <p className="text-udemy-text-muted mb-2">
              You got <span className="font-bold text-udemy-text">{correctCount}</span>{" "}
              out of <span className="font-bold text-udemy-text">{total}</span>{" "}
              correct.
            </p>
            <p className="text-xs text-udemy-text-muted mb-3">
              Powered by {providerInfo.provider}
              {providerInfo.model && ` / ${providerInfo.model}`}
              {` · retries: ${providerInfo.retries} · dropped malformed: ${providerInfo.malformed}`}
            </p>

            <div className="flex flex-wrap items-center justify-center gap-3">
              <button
                onClick={handleRestart}
                className="btn-secondary flex items-center gap-2"
              >
                <RotateCcw className="w-4 h-4" />
                New Quiz
              </button>
              <button
                onClick={() => setShowReview(!showReview)}
                className="btn-primary flex items-center gap-2"
              >
                <ListChecks className="w-4 h-4" />
                {showReview ? "Hide" : "Review"} Answers
              </button>
            </div>
          </motion.div>

          <div className="max-w-2xl w-full mb-6">
            <div className="udemy-card p-4">
              <h3 className="font-bold mb-2">Personalized Next Step</h3>
              {loadingWeakAreas ? (
                <p className="text-sm text-udemy-text-muted">Loading weak areas...</p>
              ) : weakAreas.length === 0 ? (
                <p className="text-sm text-udemy-text-muted">
                  No weak-area data yet. Keep practicing to build your adaptive queue.
                </p>
              ) : (
                <div className="space-y-2">
                  {weakAreas.slice(0, 3).map((w) => (
                    <div
                      key={w.topic_id}
                      className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-sm bg-udemy-bg rounded p-2"
                    >
                      <div className="min-w-0">
                        <span className="font-medium">{w.topic_id}</span>
                        <span className="text-udemy-text-muted ml-2">
                          mastery {Math.round(w.mastery_score * 100)}% · due {w.due_count}
                        </span>
                      </div>
                      <Link to={`/topics/${w.topic_id}`} className="text-udemy-purple">
                        Review
                      </Link>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <AnimatePresence>
            {showReview && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                className="max-w-2xl w-full space-y-3"
              >
                <h3 className="text-lg font-bold mb-2 text-center">Question Review</h3>
                {questions.map((q, idx) => {
                  const ans = answers[idx];
                  const isCorrect = ans?.correct;
                  const reviewTopic = topicsById.get(q.topic_id);
                  const reviewTopicAi = topicAiById[q.topic_id];
                  const reviewContextAnswer = [
                    q.explanation,
                    q.source_quote ? `Source quote: ${q.source_quote}` : "",
                  ]
                    .filter(Boolean)
                    .join("\n\n");
                  return (
                    <motion.div
                      key={q.question_id}
                      variants={cardVariants}
                      initial="hidden"
                      animate="show"
                    >
                      <WordHighlightChat
                        contextQuestion={q.question}
                        contextAnswer={reviewContextAnswer}
                        sessionKey={`quiz:${q.topic_id || "unknown"}`}
                        topicId={q.topic_id}
                        topicTitle={reviewTopic?.title || q.topic_id}
                        topicTrack={reviewTopic?.track || ""}
                        mode="quiz"
                        responseDetail={reviewTopicAi?.responseDetail || "concise"}
                        preferredLanguage={reviewTopicAi?.preferredLanguage || ""}
                        requiresProgramming={!!reviewTopicAi?.requiresProgramming}
                        selectionTargetSelector='[data-word-chat-target="true"]'
                        qaKey={`quiz-review:${q.question_id}`}
                      >
                        <div className="udemy-card overflow-hidden">
                          <button
                            onClick={() => {
                              const highlighted = window.getSelection()?.toString().trim();
                              if (highlighted) return;
                              setExpandedReview(expandedReview === idx ? null : idx);
                            }}
                            className="w-full p-4 text-left flex items-start gap-3 hover:bg-gray-50 transition-colors"
                          >
                            <span
                              className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold text-white ${
                                isCorrect ? "bg-green-500" : "bg-red-500"
                              }`}
                            >
                              {idx + 1}
                            </span>
                            <div className="flex-1 min-w-0">
                              <p
                                data-word-chat-target="true"
                                className="text-sm font-medium line-clamp-2"
                              >
                                {q.question}
                              </p>
                              <div className="flex items-center gap-2 mt-1">
                                <span
                                  className={`text-xs font-bold ${
                                    isCorrect ? "text-green-600" : "text-red-500"
                                  }`}
                                >
                                  {isCorrect ? "Correct" : "Incorrect"}
                                </span>
                                <span className="text-xs text-udemy-text-muted">
                                  Your answer: {ans?.selected}
                                  {!isCorrect && ` · Correct: ${q.correct_answer}`}
                                </span>
                              </div>
                            </div>
                            <motion.div
                              animate={{ rotate: expandedReview === idx ? 180 : 0 }}
                              className="flex-shrink-0 mt-1"
                            >
                              <ChevronDown className="w-5 h-5 text-udemy-text-muted" />
                            </motion.div>
                          </button>
                          <AnimatePresence>
                            {expandedReview === idx && (
                              <motion.div
                                variants={expandVariants}
                                initial="collapsed"
                                animate="expanded"
                                exit="collapsed"
                                className="overflow-hidden"
                              >
                                <div className="px-4 pb-4 border-t border-udemy-border pt-3">
                                  <div className="space-y-2 mb-3">
                                    {q.choices.map((c) => (
                                      <div
                                        key={c.label}
                                        className={`text-sm px-3 py-2 rounded-lg flex items-center gap-2 ${
                                          c.label === q.correct_answer
                                            ? "bg-green-50 text-green-800 font-medium"
                                            : c.label === ans?.selected && !ans?.correct
                                              ? "bg-red-50 text-red-800"
                                              : "text-gray-600"
                                        }`}
                                      >
                                        <span className="font-bold">{c.label}.</span>
                                        <span>{c.text}</span>
                                      </div>
                                    ))}
                                  </div>
                                  {q.explanation && (
                                    <div className="bg-udemy-bg rounded-lg p-3">
                                      <h4 className="text-xs font-bold text-udemy-text-muted uppercase mb-1">
                                        Explanation
                                      </h4>
                                      <div data-word-chat-target="true">
                                        <MarkdownRenderer
                                          content={q.explanation}
                                          className="text-[13px]"
                                        />
                                      </div>
                                      <p
                                        data-word-chat-target="true"
                                        className="text-xs text-udemy-text-muted mt-2 italic"
                                      >
                                        Source: &ldquo;{q.source_quote}&rdquo;
                                      </p>
                                    </div>
                                  )}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      </WordHighlightChat>
                    </motion.div>
                  );
                })}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  );
}
