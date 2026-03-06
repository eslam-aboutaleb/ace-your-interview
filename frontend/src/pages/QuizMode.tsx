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
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import {
  scaleInVariants,
  pageVariants,
  pageTransition,
  containerVariants,
  cardVariants,
  expandVariants,
} from "@/utils/animations";
import { fetchTopics, generateQuiz } from "@/services/api";
import { useSettingsStore } from "@/store/settingsStore";
import DifficultyBadge from "@/components/common/DifficultyBadge";
import type { TopicSummary, QuizQuestion, QuizQuestionType } from "@/types";

type QuizState = "setup" | "playing" | "results";

export default function QuizMode() {
  const { topicId } = useParams<{ topicId: string }>();
  const location = useLocation();

  // If navigated from TopicStudy with pre-filled topic
  const preselectedTopic = topicId || "";

  // ── Setup state ──────────────────────────────────────────
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
  const [generating, setGenerating] = useState(false);
  const [loadingTopics, setLoadingTopics] = useState(true);

  // ── Playing state ────────────────────────────────────────
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [answered, setAnswered] = useState(false);
  const [answers, setAnswers] = useState<
    Record<number, { selected: string; correct: boolean }>
  >({});
  const [providerInfo, setProviderInfo] = useState({ provider: "", model: "" });

  // ── Results state ────────────────────────────────────────
  const [showReview, setShowReview] = useState(false);
  const [expandedReview, setExpandedReview] = useState<number | null>(null);

  const settings = useSettingsStore();

  const current = questions[currentIdx];
  const total = questions.length;
  const correctCount = useMemo(
    () => Object.values(answers).filter((a) => a.correct).length,
    [answers],
  );

  // Load topics on mount
  useEffect(() => {
    fetchTopics()
      .then(setTopics)
      .catch(console.error)
      .finally(() => setLoadingTopics(false));
  }, []);

  // If passed from TopicStudy with old-style questions, go to setup
  const passedQuestions = (location.state as { questions?: unknown[] })
    ?.questions;

  useEffect(() => {
    if (preselectedTopic && !passedQuestions) {
      setSelectedTopics(new Set([preselectedTopic]));
    }
  }, [preselectedTopic, passedQuestions]);

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
        if (next.size > 1) next.delete(type); // Keep at least one
      } else {
        next.add(type);
      }
      return next;
    });
  };

  const handleStartQuiz = useCallback(async () => {
    if (selectedTopics.size === 0) return;
    setGenerating(true);
    try {
      const res = await generateQuiz({
        topic_ids: Array.from(selectedTopics),
        count: questionCount,
        question_types: Array.from(questionTypes),
        difficulty: difficulty || undefined,
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
      });
      if (res.questions.length > 0) {
        setQuestions(res.questions);
        setProviderInfo({
          provider: res.provider_used,
          model: res.model_used,
        });
        setCurrentIdx(0);
        setAnswers({});
        setSelectedAnswer(null);
        setAnswered(false);
        setQuizState("playing");
      }
    } catch (err) {
      console.error(err);
    } finally {
      setGenerating(false);
    }
  }, [selectedTopics, questionCount, questionTypes, difficulty, settings]);

  const handleSubmitAnswer = () => {
    if (!selectedAnswer || !current) return;
    const correct = selectedAnswer === current.correct_answer;
    setAnswers((prev) => ({
      ...prev,
      [currentIdx]: { selected: selectedAnswer, correct },
    }));
    setAnswered(true);
  };

  const handleNext = () => {
    if (currentIdx < total - 1) {
      setCurrentIdx((p) => p + 1);
      setSelectedAnswer(null);
      setAnswered(false);
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
    setShowReview(false);
    setExpandedReview(null);
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
      {/* ═══════════ SETUP SCREEN ═══════════ */}
      {quizState === "setup" && (
        <>
          <div className="bg-udemy-dark text-white">
            <div className="max-w-[1340px] mx-auto px-6 py-8">
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
                Test your knowledge with multiple choice and true/false
                questions
              </p>
            </div>
          </div>

          <div className="max-w-[1340px] mx-auto px-6 py-8 flex-1">
            <motion.div
              className="grid grid-cols-1 lg:grid-cols-3 gap-6"
              variants={containerVariants}
              initial="hidden"
              animate="show"
            >
              {/* Topic selection */}
              <motion.div variants={cardVariants} className="lg:col-span-2">
                <div className="udemy-card p-6">
                  <h2 className="text-lg font-bold mb-4">Select Topics</h2>
                  <p className="text-sm text-udemy-text-muted mb-4">
                    Choose one or more topics to be quizzed on
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
                              <p className="text-xs text-udemy-text-muted line-clamp-1">
                                {t.section_count} sections
                              </p>
                            </div>
                          </button>
                        ))}
                  </div>
                  <div className="flex gap-2 mt-4">
                    <button
                      onClick={() =>
                        setSelectedTopics(new Set(topics.map((t) => t.id)))
                      }
                      className="text-xs text-udemy-purple hover:underline"
                    >
                      Select all
                    </button>
                    <span className="text-xs text-udemy-text-muted">·</span>
                    <button
                      onClick={() => setSelectedTopics(new Set())}
                      className="text-xs text-udemy-text-muted hover:underline"
                    >
                      Clear
                    </button>
                  </div>
                </div>
              </motion.div>

              {/* Quiz settings panel */}
              <motion.div variants={cardVariants}>
                <div className="udemy-card p-6 space-y-6">
                  {/* Question count */}
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Number of Questions
                    </label>
                    <input
                      type="range"
                      min="5"
                      max="100"
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
                      <span>100</span>
                    </div>
                  </div>

                  {/* Question types */}
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Question Types
                    </label>
                    <div className="flex gap-2">
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

                  {/* Difficulty */}
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

                  {/* Start button */}
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
                    Powered by{" "}
                    <span className="capitalize">{settings.provider}</span>
                  </p>
                </div>
              </motion.div>
            </motion.div>
          </div>
        </>
      )}

      {/* ═══════════ PLAYING SCREEN ═══════════ */}
      {quizState === "playing" && current && (
        <>
          {/* Top bar */}
          <div className="bg-udemy-dark text-white">
            <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
              <button
                onClick={() => {
                  if (confirm("Exit quiz? Your progress will be lost.")) {
                    handleRestart();
                  }
                }}
                className="flex items-center gap-1 text-sm text-gray-400 hover:text-white transition-colors"
              >
                <X className="w-4 h-4" />
                Exit Quiz
              </button>
              <span className="text-sm font-medium">
                {currentIdx + 1} / {total}
              </span>
              <span className="text-sm text-gray-400">
                {correctCount} correct
              </span>
            </div>
            {/* Progress */}
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

          <div className="flex-1 flex items-center justify-center px-6 py-10">
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
                  {/* Question header */}
                  <div className="flex items-center gap-3 mb-2">
                    <span className="w-8 h-8 bg-udemy-purple rounded-full flex items-center justify-center text-white text-sm font-bold">
                      {currentIdx + 1}
                    </span>
                    <DifficultyBadge
                      difficulty={
                        current.difficulty as "easy" | "medium" | "hard"
                      }
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

                  {/* Question text */}
                  <h2 className="text-lg md:text-xl font-bold leading-relaxed mb-6 mt-4">
                    {current.question}
                  </h2>

                  {/* Answer choices */}
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
                          <span
                            className={`text-[15px] leading-relaxed pt-1 ${
                              isCorrect
                                ? "font-medium text-green-800"
                                : isWrong
                                  ? "text-red-800"
                                  : ""
                            }`}
                          >
                            {choice.text}
                          </span>
                        </button>
                      );
                    })}
                  </div>

                  {/* Explanation after answering */}
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
                          <div className="markdown-content text-[14px] leading-relaxed">
                            <ReactMarkdown>{current.explanation}</ReactMarkdown>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Actions */}
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

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => {
                          setCurrentIdx((p) => Math.max(0, p - 1));
                          setSelectedAnswer(
                            answers[currentIdx - 1]?.selected || null,
                          );
                          setAnswered(!!answers[currentIdx - 1]);
                        }}
                        disabled={currentIdx === 0}
                        className="p-2 rounded hover:bg-gray-100 disabled:opacity-30 transition-colors"
                      >
                        <ChevronLeft className="w-5 h-5" />
                      </button>
                      <span className="text-sm text-udemy-text-muted">
                        {currentIdx + 1} / {total}
                      </span>
                      <button
                        onClick={() => {
                          if (answers[currentIdx]) {
                            setCurrentIdx((p) => Math.min(total - 1, p + 1));
                            setSelectedAnswer(
                              answers[currentIdx + 1]?.selected || null,
                            );
                            setAnswered(!!answers[currentIdx + 1]);
                          }
                        }}
                        disabled={
                          currentIdx === total - 1 || !answers[currentIdx]
                        }
                        className="p-2 rounded hover:bg-gray-100 disabled:opacity-30 transition-colors"
                      >
                        <ChevronRight className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </>
      )}

      {/* ═══════════ RESULTS SCREEN ═══════════ */}
      {quizState === "results" && (
        <div className="flex-1 flex flex-col items-center px-6 py-10">
          <motion.div
            variants={scaleInVariants}
            initial="hidden"
            animate="visible"
            className="udemy-card p-8 max-w-lg w-full text-center mb-8"
          >
            {/* Score ring */}
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
                  transition={{ duration: 1.5, ease: "easeOut", delay: 0.3 }}
                />
              </svg>
              <motion.span
                className="absolute inset-0 flex items-center justify-center text-3xl font-bold"
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.8, type: "spring" }}
              >
                {percentage}%
              </motion.span>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1 }}
            >
              <h2 className="text-2xl font-bold mb-2">
                {percentage >= 80
                  ? "Outstanding! \u{1F3C6}"
                  : percentage >= 60
                    ? "Great work! \u{1F4AA}"
                    : percentage >= 40
                      ? "Good effort! \u{1F4DA}"
                      : "Keep studying! \u{1F3AF}"}
              </h2>
              <p className="text-udemy-text-muted mb-2">
                You got{" "}
                <span className="font-bold text-udemy-text">
                  {correctCount}
                </span>{" "}
                out of{" "}
                <span className="font-bold text-udemy-text">{total}</span>{" "}
                questions correct.
              </p>

              {/* Breakdown */}
              <div className="flex items-center justify-center gap-4 mb-6 text-sm">
                <span className="text-green-600 font-medium flex items-center gap-1">
                  <CheckCircle2 className="w-4 h-4" />
                  {correctCount} correct
                </span>
                <span className="text-red-500 font-medium flex items-center gap-1">
                  <XCircle className="w-4 h-4" />
                  {total - correctCount} incorrect
                </span>
              </div>

              {providerInfo.provider && (
                <p className="text-xs text-udemy-text-muted mb-4">
                  Powered by {providerInfo.provider}
                  {providerInfo.model && ` / ${providerInfo.model}`}
                </p>
              )}

              <div className="flex items-center justify-center gap-3">
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
          </motion.div>

          {/* Review section */}
          <AnimatePresence>
            {showReview && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                className="max-w-2xl w-full space-y-3"
              >
                <h3 className="text-lg font-bold mb-2 text-center">
                  Question Review
                </h3>
                {questions.map((q, idx) => {
                  const ans = answers[idx];
                  const isCorrect = ans?.correct;
                  return (
                    <motion.div
                      key={idx}
                      variants={cardVariants}
                      initial="hidden"
                      animate="show"
                      className="udemy-card overflow-hidden"
                    >
                      <button
                        onClick={() =>
                          setExpandedReview(expandedReview === idx ? null : idx)
                        }
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
                          <p className="text-sm font-medium line-clamp-2">
                            {q.question}
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            <span
                              className={`text-xs font-bold ${isCorrect ? "text-green-600" : "text-red-500"}`}
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
                          animate={{
                            rotate: expandedReview === idx ? 180 : 0,
                          }}
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
                              {/* Show choices with highlighting */}
                              <div className="space-y-2 mb-3">
                                {q.choices.map((c) => (
                                  <div
                                    key={c.label}
                                    className={`text-sm px-3 py-2 rounded-lg flex items-center gap-2 ${
                                      c.label === q.correct_answer
                                        ? "bg-green-50 text-green-800 font-medium"
                                        : c.label === ans?.selected &&
                                            !ans.correct
                                          ? "bg-red-50 text-red-800"
                                          : "text-gray-600"
                                    }`}
                                  >
                                    <span className="font-bold">
                                      {c.label}.
                                    </span>
                                    <span>{c.text}</span>
                                    {c.label === q.correct_answer && (
                                      <CheckCircle2 className="w-4 h-4 text-green-600 ml-auto flex-shrink-0" />
                                    )}
                                  </div>
                                ))}
                              </div>
                              {q.explanation && (
                                <div className="bg-udemy-bg rounded-lg p-3">
                                  <h4 className="text-xs font-bold text-udemy-text-muted uppercase mb-1">
                                    Explanation
                                  </h4>
                                  <div className="markdown-content text-[13px] leading-relaxed">
                                    <ReactMarkdown>
                                      {q.explanation}
                                    </ReactMarkdown>
                                  </div>
                                </div>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
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
