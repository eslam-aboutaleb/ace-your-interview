import { useState, useEffect, useMemo, useCallback, useRef } from "react";
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
  Lightbulb,
  Timer,
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
  generateQuizV2Stream,
  isQuizStreamError,
  recordLearningAttempt,
  fetchWeakAreas,
  fetchHint,
  fetchNextObjectives,
} from "@/services/api";
import { useSettingsStore } from "@/store/settingsStore";
import { useProgressStore } from "@/store/progressStore";
import { useQuizStore } from "@/store/quizStore";
import { useStreamGeneration } from "@/hooks/useStreamGeneration";
import DifficultyBadge from "@/components/common/DifficultyBadge";
import WordHighlightChat from "@/components/common/WordHighlightChat";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import {
  normalizeEscapedMultilineText,
  normalizeEscapedSingleLineText,
} from "@/utils/textNormalization";
import type {
  GenerateQuizStreamDoneEvent,
  HintResponse,
  InterviewLevel,
  LearningTrack,
  ResponseDetail,
  TopicSummary,
  QuizQuestionType,
  QuizQuestionV2,
  QuizMode,
  NextObjectiveItem,
  WeakAreaItem,
} from "@/types";

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

  const quiz = useQuizStore();
  const {
    phase,
    questions,
    currentIdx,
    selectedAnswer,
    answered,
    answerConfidence,
    answers,
    questionStartMs,
    providerInfo,
    generating,
    generationProgress,
    generatedPreviewCount,
    degradedQuiz,
    errorMsg,
    examFinalized,
    remainingSeconds,
  } = quiz;

  // Local UI state: setup configuration, modals,
  // per-question hint UI, and topic metadata.
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
  const [quizMode, setQuizMode] = useState<QuizMode>("practice");
  const [timeLimitSeconds, setTimeLimitSeconds] = useState<number | null>(
    null,
  );
  const [targetObjectives, setTargetObjectives] = useState<NextObjectiveItem[]>(
    [],
  );
  const [loadingTopics, setLoadingTopics] = useState(true);

  const [showReview, setShowReview] = useState(false);
  const [expandedReview, setExpandedReview] = useState<number | null>(null);
  const [weakAreas, setWeakAreas] = useState<WeakAreaItem[]>([]);
  const [loadingWeakAreas, setLoadingWeakAreas] = useState(false);

  // Per-question hint state
  const [quizHintLevel, setQuizHintLevel] = useState(1);
  const [quizHintResult, setQuizHintResult] = useState<HintResponse | null>(
    null,
  );
  const [quizHintLoading, setQuizHintLoading] = useState(false);
  const [quizHintError, setQuizHintError] = useState("");
  const [quizHintLevelUsed, setQuizHintLevelUsed] = useState(0);
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
  const { recordAttempt } = useProgressStore();
  const { run: runQuizStream } =
    useStreamGeneration<GenerateQuizStreamDoneEvent>();

  const questionHeadingRef = useRef<HTMLHeadingElement>(null);
  const firstQuestionRef = useRef(true);

  // Reset the quiz store when leaving the page so a
  // later visit starts fresh (mirrors local useState).
  useEffect(() => {
    return () => {
      useQuizStore.getState().reset();
    };
  }, []);

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
        current.source_quote
          ? `Source quote: ${normalizeEscapedMultilineText(current.source_quote)}`
          : "",
      ]
        .filter(Boolean)
        .join("\n\n")
    : "";
  const total = questions.length;
  const correctCount = useMemo(
    () => Object.values(answers).filter((a) => a.correct).length,
    [answers],
  );

  // Move focus to newly displayed questions (skip the
  // first question so the page header keeps focus).
  useEffect(() => {
    if (phase !== "playing" || !current) return;
    if (firstQuestionRef.current) {
      firstQuestionRef.current = false;
      return;
    }
    questionHeadingRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, currentIdx]);

  useEffect(() => {
    fetchTopics()
      .then((loaded) =>
        setTopics(
          loaded.map((topic) => ({
            ...topic,
            title: normalizeEscapedSingleLineText(topic.title),
            description: normalizeEscapedSingleLineText(topic.description),
          })),
        ),
      )
      .catch(() =>
        useQuizStore.getState().setErrorMsg("Failed to load topics."),
      )
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
    if (phase !== "results") return;
    setLoadingWeakAreas(true);
    fetchWeakAreas(5)
      .then((res) => setWeakAreas(res.weak_areas))
      .catch(() => setWeakAreas([]))
      .finally(() => setLoadingWeakAreas(false));
  }, [phase]);

  // Reset per-question hint state when moving between questions
  useEffect(() => {
    setQuizHintResult(null);
    setQuizHintError("");
    setQuizHintLevelUsed(0);
    setQuizHintLevel(1);
  }, [currentIdx]);

  // Adaptive targeting: rank the learner's weakest objectives
  // for the selected topics so the next quiz targets them.
  useEffect(() => {
    if (phase !== "setup" || selectedTopics.size === 0) {
      setTargetObjectives([]);
      return;
    }
    let cancelled = false;
    fetchNextObjectives(Array.from(selectedTopics), 5)
      .then((res) => {
        if (!cancelled) setTargetObjectives(res.objectives);
      })
      .catch(() => {
        if (!cancelled) setTargetObjectives([]);
      });
    return () => {
      cancelled = true;
    };
  }, [phase, selectedTopics]);

  // Exam countdown: auto-submit when the timer expires.
  useEffect(() => {
    if (
      phase !== "playing" ||
      quizMode !== "exam" ||
      examFinalized ||
      timeLimitSeconds === null
    ) {
      return;
    }
    if (remainingSeconds === null) {
      quiz.setRemainingSeconds(timeLimitSeconds);
      return;
    }
    if (remainingSeconds <= 0) {
      void handleExamSubmit();
      return;
    }
    const tick = window.setTimeout(() => {
      quiz.setRemainingSeconds((prev) => (prev === null ? null : prev - 1));
    }, 1000);
    return () => window.clearTimeout(tick);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, quizMode, examFinalized, timeLimitSeconds, remainingSeconds]);

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
    const quizState = useQuizStore.getState();
    if (selectedTopics.size === 0) return;
    quizState.setGenerating(true);
    quizState.setErrorMsg("");
    quizState.setGenerationProgress("Preparing quiz generation...");
    quizState.setGeneratedPreviewCount(0);
    quizState.setDegradedQuiz(null);
    quizState.setQuestions([]);
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
      const mergeUniqueQuestions = (
        currentList: QuizQuestionV2[],
        incoming: QuizQuestionV2[],
      ): QuizQuestionV2[] => {
        const seenIds = new Set<string>();
        const seenText = new Set<string>();
        const merged: QuizQuestionV2[] = [];
        for (const item of [...currentList, ...incoming]) {
          const id = (item.question_id || "").trim();
          const key = (item.question || "").trim().toLowerCase();
          if (id && seenIds.has(id)) continue;
          if (!id && key && seenText.has(key)) continue;
          if (id) seenIds.add(id);
          if (key) seenText.add(key);
          merged.push(item);
        }
        return merged;
      };
      for (const entry of topicDetails) {
        if (!entry) continue;
        const [id, detail] = entry;
        nextTopicAiById[id] = {
          responseDetail: detail.response_detail || "very_detailed",
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
        mode: quizMode,
        time_limit_seconds: quizMode === "exam" ? timeLimitSeconds : null,
        target_objectives: targetObjectives.map((o) => o.learning_objective),
      };

      let latestProviderInfo = {
        provider: "",
        model: "",
        retries: 0,
        malformed: 0,
      };

      const streamedQuestions: QuizQuestionV2[] = [];
      const result = await runQuizStream({
        stream: (signal) =>
          generateQuizV2Stream(req, {
            onStart: (event) => {
              quizState.setGenerationProgress(
                `Starting quiz generation (${event.target_count} questions).`,
              );
            },
            onProgress: (event) => {
              quizState.setGenerationProgress(event.message || "Generating quiz questions...");
            },
            onQuestion: (event) => {
              const merged = mergeUniqueQuestions(streamedQuestions, [event.question]);
              streamedQuestions.splice(0, streamedQuestions.length, ...merged);
              quizState.setQuestions([...streamedQuestions]);
              quizState.setGeneratedPreviewCount(streamedQuestions.length);
            },
            onDone: (event) => {
              latestProviderInfo = {
                provider: event.provider_used,
                model: event.model_used,
                retries: event.retries_used,
                malformed: event.malformed_items_dropped,
              };
              quizState.setProviderInfo(latestProviderInfo);
              if (event.degraded) {
                quizState.setDegradedQuiz({
                  requested: event.requested_count,
                  returned: event.returned_count,
                });
              }
              quizState.setGenerationProgress("Quiz generation completed.");
            },
            onWarning: (event) => {
              quizState.setGenerationProgress(event.message || "Some topics could not be resolved.");
            },
            onError: (event) => {
              quizState.setGenerationProgress(event.message || "Quiz generation failed.");
            },
          }, signal),
        isFallbackEligible: (err) => isQuizStreamError(err) && !!err.fallbackEligible,
      });
      if (result.error) {
        throw result.error;
      }
      if (result.usedFallback) {
        quizState.setGenerationProgress(
          "Streaming unavailable, retrying with compatibility mode...",
        );
      }

      const finalQuestions = [...streamedQuestions];
      if (finalQuestions.length === 0) {
        quizState.setErrorMsg("No quiz questions generated. Try again.");
        return;
      }
      quizState.setQuestions(finalQuestions);
      quizState.setProviderInfo(latestProviderInfo);
      quizState.beginPlaying();
    } catch (err) {
      console.error(err);
      useQuizStore.getState().setErrorMsg(
        "Quiz generation failed. Check settings and retry.",
      );
    } finally {
      useQuizStore.getState().setGenerationProgress("");
      useQuizStore.getState().setGeneratedPreviewCount(0);
      useQuizStore.getState().setGenerating(false);
    }
  }, [selectedTopics, questionCount, questionTypes, difficulty, level, quizMode, timeLimitSeconds, targetObjectives, settings, runQuizStream]);

  const handleSubmitAnswer = async () => {
    if (!selectedAnswer || !current) return;
    const correct = selectedAnswer === current.correct_answer;
    quiz.recordAnswer(currentIdx, {
      selected: selectedAnswer,
      correct,
      confidence: answerConfidence,
    });
    quiz.setAnswered(true);

    try {
      const responseTime = Math.max(
        0,
        Date.now() - (questionStartMs[currentIdx] || Date.now()),
      );
      const topicForAttempt = current.topic_id || Array.from(selectedTopics)[0] || "";
      await recordLearningAttempt({
        question_id: current.question_id,
        topic_id: topicForAttempt,
        user_answer: selectedAnswer,
        is_correct: correct,
        confidence: answerConfidence,
        response_time_ms: responseTime,
        mode: quizMode === "exam" ? "exam" : "quiz",
        learning_objective: current.learning_objective,
        question_text: current.question,
        correct_answer: current.correct_answer,
        level,
      });
      if (topicForAttempt) {
        // The server now returns the FSRS card state; the
        // display-only client-side EWMA mirror in the
        // progress store keeps the local mastery estimate.
        recordAttempt(topicForAttempt, correct, answerConfidence);
      }
    } catch (err) {
      console.error(err);
      quiz.setErrorMsg("Answer recorded locally, but syncing attempt failed.");
    }
  };

  // Exam mode: answers are withheld until every question is
  // answered (or the timer expires), then all attempts are
  // recorded in one pass.
  const handleExamSubmit = async () => {
    if (examFinalized) return;
    quiz.setExamFinalized(true);
    const topicFallback = Array.from(selectedTopics)[0] || "";
    await Promise.all(
      questions.map(async (q, idx) => {
        const ans = answers[idx];
        if (!ans) return;
        const topicForAttempt = q.topic_id || topicFallback;
        try {
          await recordLearningAttempt({
            question_id: q.question_id,
            topic_id: topicForAttempt,
            user_answer: ans.selected,
            is_correct: ans.correct,
            confidence: ans.confidence,
            response_time_ms: 0,
            mode: "exam",
            learning_objective: q.learning_objective,
          });
          if (topicForAttempt) {
            recordAttempt(topicForAttempt, ans.correct, ans.confidence);
          }
        } catch (err) {
          console.error(err);
        }
      }),
    );
    quiz.setPhase("results");
  };

  const handleQuizHint = async () => {
    if (!current) return;
    setQuizHintLoading(true);
    setQuizHintError("");
    try {
      const res = await fetchHint(current.question_id, quizHintLevel, {
        question_text: current.question,
      });
      setQuizHintResult(res);
      setQuizHintLevelUsed((prev) => Math.max(prev, res.level));
    } catch (err) {
      const detail = (
        err as { response?: { data?: { detail?: unknown } } }
      )?.response?.data?.detail;
      setQuizHintError(
        typeof detail === "string"
          ? detail
          : "Could not load a hint for this question.",
      );
    } finally {
      setQuizHintLoading(false);
    }
  };

  const handleNext = () => {
    if (currentIdx < total - 1) {
      const nextIdx = currentIdx + 1;
      quiz.setCurrentIdx(nextIdx);
      quiz.setSelectedAnswer(answers[nextIdx]?.selected || null);
      quiz.setAnswered(!!answers[nextIdx]);
      quiz.setAnswerConfidence(answers[nextIdx]?.confidence || 3);
      quiz.touchQuestionStart(nextIdx);
    } else if (quizMode === "exam") {
      void handleExamSubmit();
    } else {
      quiz.setPhase("results");
    }
  };

  const handlePrev = () => {
    if (currentIdx > 0) {
      const prevIdx = currentIdx - 1;
      quiz.setCurrentIdx(prevIdx);
      quiz.setSelectedAnswer(answers[prevIdx]?.selected || null);
      quiz.setAnswered(!!answers[prevIdx]);
      quiz.setAnswerConfidence(answers[prevIdx]?.confidence || 3);
    }
  };

  const handleRestart = () => {
    quiz.resetToSetup();
    setShowReview(false);
    setExpandedReview(null);
    setWeakAreas([]);
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
      {phase === "setup" && (
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
              <div className="mb-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2" role="alert">
                <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  {errorMsg}
                  <button
                    onClick={() => void handleStartQuiz()}
                    disabled={generating || selectedTopics.size === 0}
                    className="ml-2 font-bold underline disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Retry generation
                  </button>
                </div>
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

                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Quiz Mode
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => setQuizMode("practice")}
                        className={`py-2 px-3 rounded-lg border-2 text-sm font-medium transition-all ${
                          quizMode === "practice"
                            ? "border-udemy-purple bg-udemy-purple/10 text-udemy-purple"
                            : "border-udemy-border text-udemy-text-muted"
                        }`}
                      >
                        Practice
                      </button>
                      <button
                        onClick={() => setQuizMode("exam")}
                        className={`py-2 px-3 rounded-lg border-2 text-sm font-medium transition-all ${
                          quizMode === "exam"
                            ? "border-udemy-purple bg-udemy-purple/10 text-udemy-purple"
                            : "border-udemy-border text-udemy-text-muted"
                        }`}
                      >
                        Exam
                      </button>
                    </div>
                    <p className="text-xs text-udemy-text-muted mt-1.5">
                      {quizMode === "exam"
                        ? "Timed. Answers and explanations are revealed after submission."
                        : "Immediate feedback with explanations after each answer."}
                    </p>
                  </div>

                  {quizMode === "exam" && (
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Time Limit
                      </label>
                      <select
                        value={timeLimitSeconds === null ? "" : String(timeLimitSeconds)}
                        onChange={(e) =>
                          setTimeLimitSeconds(
                            e.target.value === "" ? null : Number(e.target.value),
                          )
                        }
                        className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                      >
                        <option value="">Untimed</option>
                        <option value="120">2 minutes</option>
                        <option value="300">5 minutes</option>
                        <option value="600">10 minutes</option>
                        <option value="900">15 minutes</option>
                        <option value="1800">30 minutes</option>
                      </select>
                    </div>
                  )}

                  {targetObjectives.length > 0 && (
                    <div className="rounded-lg border border-udemy-border bg-udemy-bg p-3">
                      <h3 className="text-xs font-bold uppercase text-udemy-text-muted mb-1.5">
                        Adaptive Targets
                      </h3>
                      <ul className="space-y-1">
                        {targetObjectives.slice(0, 3).map((objective) => (
                          <li
                            key={`${objective.topic_id}:${objective.learning_objective}`}
                            className="text-xs text-udemy-text-muted"
                          >
                            <span className="font-medium text-udemy-text">
                              {normalizeEscapedSingleLineText(
                                objective.topic_title || objective.topic_id,
                              )}
                            </span>
                            <span className="block truncate">
                              {normalizeEscapedSingleLineText(
                                objective.learning_objective,
                              )}
                            </span>
                          </li>
                        ))}
                      </ul>
                      <p className="text-[11px] text-udemy-text-muted mt-1.5">
                        Next quiz prioritizes your weakest objectives.
                      </p>
                    </div>
                  )}

                  <button
                    onClick={() => void handleStartQuiz()}
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
                  {generating && (
                    <p className="mt-2 text-xs text-udemy-text-muted" aria-live="polite" role="status">
                      {generationProgress || "Generating quiz questions..."}{" "}
                      {generatedPreviewCount > 0
                        ? `(${generatedPreviewCount}/${questionCount})`
                        : ""}
                    </p>
                  )}

                  <p className="text-xs text-center text-udemy-text-muted">
                    Powered by {providerInfo.provider || settings.provider}
                  </p>
                </div>
              </motion.div>
            </motion.div>
          </div>
        </>
      )}

      {phase === "playing" && current && (
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
              {quizMode === "exam" && timeLimitSeconds !== null && (
                <span
                  className="text-sm font-medium inline-flex items-center gap-1.5"
                  role="timer"
                  aria-live="polite"
                  aria-label={`Time remaining: ${remainingSeconds ?? timeLimitSeconds} seconds`}
                >
                  <Timer className="w-4 h-4" />
                  {Math.floor((remainingSeconds ?? timeLimitSeconds) / 60)}:
                  {String((remainingSeconds ?? timeLimitSeconds) % 60).padStart(
                    2,
                    "0",
                  )}
                </span>
              )}
              <span className="text-sm text-gray-400">
                {quizMode === "exam" && !examFinalized
                  ? `${Object.keys(answers).length} answered`
                  : `${correctCount} correct`}
              </span>
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

          {degradedQuiz && (
            <div className="max-w-2xl mx-auto px-4 sm:px-6 py-4">
              <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2" role="alert">
                <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  Only {degradedQuiz.returned} of {degradedQuiz.requested}{" "}
                  questions could be generated. The quiz continues with the
                  available questions.
                  <button
                    onClick={() => {
                      quiz.resetToSetup();
                      void handleStartQuiz();
                    }}
                    disabled={generating}
                    className="ml-2 font-bold underline disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Retry generation
                  </button>
                </div>
              </div>
            </div>
          )}

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
                    contextQuestion={normalizeEscapedSingleLineText(current.question)}
                    contextAnswer={currentContextAnswer}
                    sessionKey={`quiz:${current.topic_id || "unknown"}`}
                    topicId={current.topic_id}
                    topicTitle={
                      normalizeEscapedSingleLineText(currentTopic?.title || current.topic_id)
                    }
                    topicTrack={currentTopic?.track || ""}
                    mode="quiz"
                    showFloatingTrigger
                    floatingTriggerWord={
                      normalizeEscapedSingleLineText(currentTopic?.title || current.topic_id)
                    }
                    floatingTriggerPrompt="Explain this quiz concept in an organized way and include practical tradeoffs."
                    responseDetail={currentTopicAi?.responseDetail || "very_detailed"}
                    preferredLanguage={currentTopicAi?.preferredLanguage || ""}
                    requiresProgramming={!!currentTopicAi?.requiresProgramming}
                    selectionTargetSelector='[data-word-chat-target="true"]'
                    qaKey={`quiz:${current.question_id}`}
                  >
                    <h2
                      ref={questionHeadingRef}
                      tabIndex={-1}
                      data-word-chat-target="true"
                      className="text-lg md:text-xl font-bold leading-relaxed mb-6 mt-4 outline-none"
                    >
                      {normalizeEscapedSingleLineText(current.question)}
                    </h2>

                    <div className="space-y-3 mb-6">
                      {current.choices.map((choice) => {
                        const isSelected = selectedAnswer === choice.label;
                        // Exam mode withholds correctness until the
                        // whole exam is submitted; practice mode
                        // reveals it per answer.
                        const revealCorrectness =
                          quizMode === "practice" ? answered : examFinalized;
                        const isCorrect =
                          revealCorrectness &&
                          choice.label === current.correct_answer;
                        const isWrong =
                          revealCorrectness &&
                          isSelected &&
                          choice.label !== current.correct_answer;

                        return (
                          <button
                            key={choice.label}
                            onClick={() => {
                              if (quizMode === "exam") {
                                quiz.setSelectedAnswer(choice.label);
                                quiz.recordAnswer(currentIdx, {
                                  selected: choice.label,
                                  correct:
                                    choice.label === current.correct_answer,
                                  confidence: answerConfidence,
                                });
                                quiz.setAnswered(true);
                                return;
                              }
                              if (!answered) quiz.setSelectedAnswer(choice.label);
                            }}
                            disabled={quizMode === "practice" && answered}
                            className={`w-full p-4 rounded-lg border-2 text-left transition-all flex items-start gap-3 ${
                              isCorrect
                                ? "border-green-500 bg-green-50"
                                : isWrong
                                  ? "border-red-500 bg-red-50"
                                  : isSelected
                                    ? "border-udemy-purple bg-udemy-purple/5"
                                    : "border-udemy-border hover:border-gray-400"
                            } ${
                              quizMode === "practice" && answered
                                ? "cursor-default"
                                : "cursor-pointer"
                            }`}
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
                              {normalizeEscapedSingleLineText(choice.text)}
                            </span>
                          </button>
                        );
                      })}
                    </div>

                    {(quizMode === "practice"
                      ? !answered
                      : !examFinalized) && (
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
                          onChange={(e) => quiz.setAnswerConfidence(Number(e.target.value))}
                          className="w-full accent-udemy-purple"
                        />
                      </div>
                    )}

                    {quizMode === "practice" && (
                      <div className="mb-5 rounded-lg border border-udemy-border p-3">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="inline-flex items-center gap-1.5 text-xs font-medium">
                            <Lightbulb className="w-3.5 h-3.5 text-udemy-purple" />
                            Hint
                          </span>
                          <select
                            value={quizHintLevel}
                            onChange={(e) => setQuizHintLevel(Number(e.target.value))}
                            disabled={quizHintLoading}
                            className="border border-udemy-border rounded px-2 py-1 text-xs"
                          >
                            <option value={1}>Level 1 — nudge</option>
                            <option value={2}>Level 2 — approach</option>
                            <option value={3}>Level 3 — partial</option>
                            <option value={4}>Level 4 — answer</option>
                          </select>
                          <button
                            onClick={handleQuizHint}
                            disabled={quizHintLoading}
                            className="btn-secondary inline-flex items-center gap-1 text-xs disabled:opacity-50"
                          >
                            {quizHintLoading ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              <Lightbulb className="w-3 h-3" />
                            )}
                            Get hint
                          </button>
                          {quizHintResult && (
                            <span className="text-[11px] text-udemy-text-muted">
                              {quizHintResult.hints_used}/{quizHintResult.hints_remaining === 0 ? 0 : quizHintResult.hints_used + quizHintResult.hints_remaining} used
                            </span>
                          )}
                        </div>
                        {quizHintError && (
                          <p className="text-[11px] text-red-600 mt-1.5 flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3" />
                            {quizHintError}
                          </p>
                        )}
                        {quizHintResult && (
                          <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded mt-2 p-2 whitespace-pre-line">
                            {quizHintResult.hint}
                          </p>
                        )}
                      </div>
                    )}

                    <AnimatePresence>
                      {(quizMode === "practice"
                        ? answered
                        : examFinalized) &&
                        current.explanation && (
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
                                  className="text-xs text-udemy-text-muted mt-2 italic whitespace-pre-line"
                                >
                                  Source: &ldquo;{normalizeEscapedMultilineText(current.source_quote)}&rdquo;
                                </p>
                              )}
                            </div>
                          </motion.div>
                        )}
                    </AnimatePresence>
                  </WordHighlightChat>

                  <div className="flex items-center justify-between">
                    {quizMode === "exam" ? (
                      <>
                        <button
                          onClick={handlePrev}
                          disabled={currentIdx === 0}
                          className="btn-secondary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          <ChevronLeft className="w-4 h-4" />
                          Previous
                        </button>
                        {currentIdx < total - 1 ? (
                          <button
                            onClick={handleNext}
                            className="btn-primary flex items-center gap-2"
                          >
                            Next <ChevronRight className="w-4 h-4" />
                          </button>
                        ) : (
                          <button
                            onClick={() => void handleExamSubmit()}
                            disabled={examFinalized}
                            className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            <Trophy className="w-4 h-4" />
                            Submit Exam
                          </button>
                        )}
                      </>
                    ) : !answered ? (
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

      {phase === "results" && (
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
                        <span className="font-medium">
                          {normalizeEscapedSingleLineText(w.topic_id)}
                        </span>
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
                    q.source_quote
                      ? `Source quote: ${normalizeEscapedMultilineText(q.source_quote)}`
                      : "",
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
                        contextQuestion={normalizeEscapedSingleLineText(q.question)}
                        contextAnswer={reviewContextAnswer}
                        sessionKey={`quiz:${q.topic_id || "unknown"}`}
                        topicId={q.topic_id}
                        topicTitle={
                          normalizeEscapedSingleLineText(reviewTopic?.title || q.topic_id)
                        }
                        topicTrack={reviewTopic?.track || ""}
                        mode="quiz"
                        responseDetail={reviewTopicAi?.responseDetail || "very_detailed"}
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
                                {normalizeEscapedSingleLineText(q.question)}
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
                                        className="text-xs text-udemy-text-muted mt-2 italic whitespace-pre-line"
                                      >
                                        Source: &ldquo;{normalizeEscapedMultilineText(q.source_quote)}&rdquo;
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
