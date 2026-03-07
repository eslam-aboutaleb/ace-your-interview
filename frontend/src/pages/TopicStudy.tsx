import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { useParams, Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronDown,
  ChevronLeft,
  Sparkles,
  BookOpen,
  CheckCircle2,
  Loader2,
  Play,
  AlertTriangle,
  Search,
  X,
} from "lucide-react";
import {
  pageVariants,
  pageTransition,
  expandVariants,
  containerVariants,
  cardVariants,
} from "@/utils/animations";
import {
  fetchTopic,
  generateQuestions,
  generateQuestionsV2,
  generateQuestionsV2Stream,
  isQuestionsStreamError,
  recordLearningAttempt,
} from "@/services/api";
import { useSettingsStore } from "@/store/settingsStore";
import { useProgressStore } from "@/store/progressStore";
import DifficultyBadge from "@/components/common/DifficultyBadge";
import ProgressBar from "@/components/common/ProgressBar";
import WordHighlightChat from "@/components/common/WordHighlightChat";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import type {
  TopicDetail,
  QuestionAnswerV2,
  InterviewLevel,
  LearningTrack,
} from "@/types";

const TRACK_LABELS: Record<LearningTrack, string> = {
  backend: "Backend",
  frontend: "Frontend",
  system_design: "System Design",
  ai_stack: "AI Stack",
};

type ProviderInfo = {
  provider: string;
  model: string;
  retries: number;
  malformed: number;
};

type IndexedSection = {
  index: number;
  heading: string;
  content: string;
  searchable: string;
};

export default function TopicStudy() {
  const { topicId } = useParams<{ topicId: string }>();
  const [topic, setTopic] = useState<TopicDetail | null>(null);
  const [questions, setQuestions] = useState<QuestionAnswerV2[]>([]);
  const [expandedQ, setExpandedQ] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [revealedAnswers, setRevealedAnswers] = useState<Set<number>>(new Set());
  const [submittedAttempts, setSubmittedAttempts] = useState<Set<number>>(
    new Set(),
  );
  const [answerDrafts, setAnswerDrafts] = useState<Record<number, string>>({});
  const [confidenceByIdx, setConfidenceByIdx] = useState<Record<number, number>>({});
  const [questionStartMs, setQuestionStartMs] = useState<Record<number, number>>(
    {},
  );
  const [submittingIdx, setSubmittingIdx] = useState<number | null>(null);
  const [activeSection, setActiveSection] = useState(0);
  const [courseSearchInput, setCourseSearchInput] = useState("");
  const [courseSearchQuery, setCourseSearchQuery] = useState("");
  const [navScrollTop, setNavScrollTop] = useState(0);
  const [navViewportHeight, setNavViewportHeight] = useState(0);
  const [questionCount, setQuestionCount] = useState(5);
  const [difficulty, setDifficulty] = useState<string>("");
  const [level, setLevel] = useState<InterviewLevel>("mid");
  const [providerInfo, setProviderInfo] = useState<ProviderInfo>({
    provider: "",
    model: "",
    retries: 0,
    malformed: 0,
  });
  const [errorMsg, setErrorMsg] = useState("");
  const navRef = useRef<HTMLElement | null>(null);
  const generationAbortRef = useRef<AbortController | null>(null);

  const settings = useSettingsStore();
  const {
    getTopicProgress,
    markTopicComplete,
    recordAttempt,
    setMastery,
    addAnswered,
  } = useProgressStore();

  useEffect(() => {
    if (!topicId) return;
    setLoading(true);
    setActiveSection(0);
    setCourseSearchInput("");
    setCourseSearchQuery("");
    setNavScrollTop(0);
    fetchTopic(topicId)
      .then(setTopic)
      .catch(() => setErrorMsg("Failed to load this topic. Please retry."))
      .finally(() => setLoading(false));
  }, [topicId]);

  useEffect(() => {
    return () => {
      generationAbortRef.current?.abort();
      generationAbortRef.current = null;
    };
  }, []);

  useEffect(() => {
    const handle = window.setTimeout(
      () => setCourseSearchQuery(courseSearchInput.trim().toLowerCase()),
      150,
    );
    return () => window.clearTimeout(handle);
  }, [courseSearchInput]);

  const handleGenerate = useCallback(async () => {
    if (!topicId || !topic) return;
    generationAbortRef.current?.abort();
    const controller = new AbortController();
    generationAbortRef.current = controller;
    setGenerating(true);
    setErrorMsg("");
    setQuestions([]);
    setExpandedQ(null);
    setRevealedAnswers(new Set());
    setSubmittedAttempts(new Set());
    setAnswerDrafts({});
    setConfidenceByIdx({});
    setQuestionStartMs({});
    try {
      const activeS = topic.sections[activeSection];
      const payload = {
        topic_id: topicId,
        count: questionCount,
        difficulty: difficulty || undefined,
        level,
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
        ...(activeS
          ? {
              section_title: activeS.heading,
              section_content: activeS.content,
            }
          : {}),
      };
      const applyV2Result = (res: Awaited<ReturnType<typeof generateQuestionsV2>>) => {
        setQuestions(res.questions);
        setProviderInfo({
          provider: res.provider_used,
          model: res.model_used,
          retries: res.retries_used,
          malformed: res.malformed_items_dropped,
        });
      };
      const applyLegacyResult = (
        legacy: Awaited<ReturnType<typeof generateQuestions>>,
      ) => {
        const upgraded: QuestionAnswerV2[] = legacy.questions.map((q, idx) => ({
          question_id: `${topicId}:legacy:${idx}`,
          topic_id: topicId,
          question: q.question,
          answer: q.answer,
          difficulty: q.difficulty,
          learning_objective: "Understand the concept and apply it in context.",
          source_section:
            activeS?.heading || topic.sections[activeSection]?.heading || "Topic",
          source_quote: "Legacy mode response did not include source quote.",
          misconception_trap: "Confusing terms without checking documentation context.",
          reasoning_summary: "Review the answer and compare with your own reasoning.",
        }));
        setQuestions(upgraded);
        setProviderInfo({
          provider: legacy.provider_used,
          model: legacy.model_used,
          retries: 0,
          malformed: 0,
        });
      };

      const runFallbackGeneration = async () => {
        try {
          const res = await generateQuestionsV2(payload);
          applyV2Result(res);
        } catch {
          const legacy = await generateQuestions(payload);
          applyLegacyResult(legacy);
        }
      };

      let streamedCount = 0;
      try {
        await generateQuestionsV2Stream(
          payload,
          {
            onQuestion: (event) => {
              streamedCount += 1;
              setQuestions((prev) => [...prev, event.question]);
            },
            onDone: (event) => {
              setProviderInfo({
                provider: event.provider_used,
                model: event.model_used,
                retries: event.retries_used,
                malformed: event.malformed_items_dropped,
              });
            },
            onError: (event) => {
              setErrorMsg(event.message || "Question generation failed.");
            },
          },
          controller.signal,
        );
      } catch (streamErr) {
        if (controller.signal.aborted) return;
        if (
          isQuestionsStreamError(streamErr)
          && streamErr.fallbackEligible
          && streamedCount === 0
        ) {
          await runFallbackGeneration();
          return;
        }
        throw streamErr;
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      console.error(err);
      if (isQuestionsStreamError(err)) {
        setErrorMsg(err.message || "Question generation failed.");
        return;
      }
      setErrorMsg(
        "Question generation failed. Check LLM settings/connection and try again.",
      );
    } finally {
      if (generationAbortRef.current === controller) {
        generationAbortRef.current = null;
        setGenerating(false);
      }
    }
  }, [topicId, topic, activeSection, questionCount, difficulty, level, settings]);

  const toggleQuestion = (idx: number) => {
    if (expandedQ === idx) {
      setExpandedQ(null);
      return;
    }
    setExpandedQ(idx);
    setQuestionStartMs((prev) =>
      prev[idx] ? prev : { ...prev, [idx]: Date.now() },
    );
    setConfidenceByIdx((prev) => ({ ...prev, [idx]: prev[idx] || 3 }));
  };

  const handleReveal = (idx: number) => {
    setRevealedAnswers((prev) => new Set(prev).add(idx));
  };

  const submitAttempt = async (idx: number, isCorrect: boolean) => {
    if (!topicId || submittedAttempts.has(idx)) return;
    const q = questions[idx];
    if (!q) return;
    setSubmittingIdx(idx);
    setErrorMsg("");

    const confidence = confidenceByIdx[idx] || 3;
    const responseTime = Math.max(0, Date.now() - (questionStartMs[idx] || Date.now()));
    const learnerAnswer = answerDrafts[idx] || "";

    try {
      const res = await recordLearningAttempt({
        question_id: q.question_id,
        topic_id: topicId,
        user_answer: learnerAnswer,
        is_correct: isCorrect,
        confidence,
        response_time_ms: responseTime,
        mode: "study",
      });
      setSubmittedAttempts((prev) => new Set(prev).add(idx));
      addAnswered(topicId, 1);
      recordAttempt(topicId, isCorrect, confidence);
      setMastery(topicId, res.mastery_score);
    } catch (err) {
      console.error(err);
      setErrorMsg("Could not save your attempt. Please retry.");
    } finally {
      setSubmittingIdx(null);
    }
  };

  const allSubmitted =
    questions.length > 0 && submittedAttempts.size === questions.length;

  useEffect(() => {
    if (allSubmitted && topicId) {
      markTopicComplete(topicId);
    }
  }, [allSubmitted, topicId, markTopicComplete]);

  const progress = topicId ? getTopicProgress(topicId) : 0;
  const indexedSections = useMemo<IndexedSection[]>(
    () =>
      (topic?.sections || []).map((sec, index) => ({
        index,
        heading: sec.heading,
        content: sec.content,
        searchable: `${sec.heading} ${sec.content}`.toLowerCase(),
      })),
    [topic],
  );

  const filteredSections = useMemo(
    () =>
      !courseSearchQuery
        ? indexedSections
        : indexedSections.filter((sec) => sec.searchable.includes(courseSearchQuery)),
    [indexedSections, courseSearchQuery],
  );

  useEffect(() => {
    if (!filteredSections.length) return;
    if (filteredSections.some((sec) => sec.index === activeSection)) return;
    setActiveSection(filteredSections[0].index);
  }, [filteredSections, activeSection]);

  useEffect(() => {
    const navEl = navRef.current;
    if (!navEl) return;
    const updateViewport = () => setNavViewportHeight(navEl.clientHeight);
    updateViewport();
    window.addEventListener("resize", updateViewport);
    return () => window.removeEventListener("resize", updateViewport);
  }, [filteredSections.length]);

  useEffect(() => {
    const navEl = navRef.current;
    if (!navEl) return;
    navEl.scrollTop = 0;
    setNavScrollTop(0);
  }, [courseSearchQuery]);

  const totalSections = indexedSections.length;
  const matchedSections = filteredSections.length;
  const useVirtualizedSections = matchedSections > 250;
  const sectionRowHeight = 46;
  const sectionOverscan = 12;
  const virtualStart = useVirtualizedSections
    ? Math.max(0, Math.floor(navScrollTop / sectionRowHeight) - sectionOverscan)
    : 0;
  const virtualVisibleCount = useVirtualizedSections
    ? Math.ceil(Math.max(navViewportHeight, sectionRowHeight) / sectionRowHeight) +
      sectionOverscan * 2
    : matchedSections;
  const virtualEnd = useVirtualizedSections
    ? Math.min(matchedSections, virtualStart + virtualVisibleCount)
    : matchedSections;
  const renderedSections = filteredSections.slice(virtualStart, virtualEnd);
  const topSpacerHeight = useVirtualizedSections ? virtualStart * sectionRowHeight : 0;
  const bottomSpacerHeight = useVirtualizedSections
    ? Math.max(0, (matchedSections - virtualEnd) * sectionRowHeight)
    : 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-udemy-purple" />
      </div>
    );
  }

  if (!topic) {
    return (
      <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-12 text-center">
        <p className="text-udemy-text-muted text-lg">Topic not found.</p>
        <Link to="/" className="btn-primary inline-block mt-4">
          Back to Dashboard
        </Link>
      </div>
    );
  }

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
    >
      <div className="bg-udemy-dark text-white">
        <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-6">
          <Link
            to="/"
            className="inline-flex items-center gap-1 text-sm text-gray-400 hover:text-white mb-3 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
            Back to all topics
          </Link>
          <h1 className="text-2xl md:text-3xl font-bold mb-2">{topic.title}</h1>
          <div className="mb-2">
            <span className="inline-flex items-center rounded-full bg-udemy-purple/20 px-2.5 py-1 text-xs font-semibold text-udemy-purple-light">
              {TRACK_LABELS[topic.track] || topic.track || "Topic"}
            </span>
          </div>
          <p className="text-gray-400 text-sm max-w-2xl">{topic.description}</p>
          <div className="mt-4">
            <ProgressBar percent={progress} className="max-w-md" />
          </div>
        </div>
      </div>

      <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-8">
        <div className="flex flex-col lg:flex-row gap-8">
          <aside className="lg:w-72 flex-shrink-0">
            <div className="udemy-card p-4 lg:sticky lg:top-20">
              <h3 className="text-sm font-bold text-udemy-text-muted uppercase tracking-wide mb-3">
                Course Content
              </h3>
              <div className="relative mb-2">
                <Search className="w-4 h-4 text-udemy-text-muted absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  value={courseSearchInput}
                  onChange={(e) => setCourseSearchInput(e.target.value)}
                  placeholder="Search sections..."
                  className="w-full border border-udemy-border rounded-lg pl-9 pr-8 py-2 text-sm focus:outline-none focus:border-udemy-purple"
                />
                {courseSearchInput && (
                  <button
                    onClick={() => setCourseSearchInput("")}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-udemy-text-muted hover:text-udemy-text"
                    aria-label="Clear section search"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
              <p className="text-xs text-udemy-text-muted mb-2">
                {matchedSections} of {totalSections} sections
              </p>
              <nav
                ref={navRef}
                onScroll={(e) => setNavScrollTop(e.currentTarget.scrollTop)}
                className="space-y-1 max-h-[70vh] overflow-y-auto pr-1 friendly-scrollbar"
              >
                {matchedSections === 0 && (
                  <div className="px-3 py-3 rounded text-sm text-udemy-text-muted bg-udemy-bg">
                    No matching sections.
                  </div>
                )}
                {topSpacerHeight > 0 && <div style={{ height: topSpacerHeight }} aria-hidden />}
                {renderedSections.map((sec) => (
                  <button
                    key={sec.index}
                    onClick={() => setActiveSection(sec.index)}
                    className={`w-full h-[46px] text-left px-3 rounded text-sm transition-colors flex items-center gap-2 ${
                      activeSection === sec.index
                        ? "bg-udemy-purple/10 text-udemy-purple font-medium"
                        : "text-udemy-text-muted hover:bg-gray-50"
                    }`}
                  >
                    <span className="w-5 h-5 flex items-center justify-center rounded-full bg-udemy-bg text-xs font-bold flex-shrink-0">
                      {sec.index + 1}
                    </span>
                    <span className="line-clamp-1">{sec.heading}</span>
                  </button>
                ))}
                {bottomSpacerHeight > 0 && (
                  <div style={{ height: bottomSpacerHeight }} aria-hidden />
                )}
              </nav>

              {questions.length > 0 && (
                <Link
                  to={`/quiz/${topicId}`}
                  className="btn-secondary w-full text-center mt-4 text-sm flex items-center justify-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Quiz Mode
                </Link>
              )}
            </div>
          </aside>

          <div className="flex-1 min-w-0">
            {topic.sections[activeSection] && (
              <motion.div
                key={activeSection}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="udemy-card p-4 sm:p-6 mb-6"
              >
                <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-udemy-purple" />
                  {topic.sections[activeSection].heading}
                </h2>
                <MarkdownRenderer
                  content={topic.sections[activeSection].content}
                  className="text-[14px]"
                />
              </motion.div>
            )}

            <div className="udemy-card p-4 sm:p-6 mb-6">
              <div className="flex flex-wrap items-end gap-4 mb-4">
                <div>
                  <label className="block text-xs font-medium text-udemy-text-muted mb-1">
                    Number of Questions
                  </label>
                  <select
                    value={questionCount}
                    onChange={(e) => setQuestionCount(Number(e.target.value))}
                    className="border border-udemy-border rounded px-3 py-2 text-sm"
                  >
                    {[3, 5, 10, 15, 20].map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-udemy-text-muted mb-1">
                    Difficulty
                  </label>
                  <select
                    value={difficulty}
                    onChange={(e) => setDifficulty(e.target.value)}
                    className="border border-udemy-border rounded px-3 py-2 text-sm"
                  >
                    <option value="">All levels</option>
                    <option value="easy">Easy</option>
                    <option value="medium">Medium</option>
                    <option value="hard">Hard</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-udemy-text-muted mb-1">
                    Interview Level
                  </label>
                  <select
                    value={level}
                    onChange={(e) => setLevel(e.target.value as InterviewLevel)}
                    className="border border-udemy-border rounded px-3 py-2 text-sm"
                  >
                    <option value="junior">Junior</option>
                    <option value="mid">Mid</option>
                    <option value="senior">Senior</option>
                  </select>
                </div>

                <button
                  onClick={handleGenerate}
                  disabled={generating}
                  className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {generating ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Sparkles className="w-4 h-4" />
                  )}
                  {generating ? "Generating..." : "Generate Questions"}
                </button>
              </div>

              {providerInfo.provider && (
                <p className="text-xs text-udemy-text-muted">
                  Powered by <span className="font-medium">{providerInfo.provider}</span>
                  {providerInfo.model && ` / ${providerInfo.model}`}
                  {` · retries: ${providerInfo.retries} · dropped malformed: ${providerInfo.malformed}`}
                </p>
              )}
            </div>

            {errorMsg && (
              <div className="mb-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <p>{errorMsg}</p>
                  {!generating && (
                    <button
                      onClick={handleGenerate}
                      className="text-xs font-medium underline mt-1"
                    >
                      Retry generation
                    </button>
                  )}
                </div>
              </div>
            )}

            {generating && Math.max(0, questionCount - questions.length) > 0 && (
              <div className="space-y-3">
                {Array.from({ length: Math.max(0, questionCount - questions.length) }).map((_, i) => (
                  <motion.div
                    key={`skeleton-${i}`}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="udemy-card p-5"
                  >
                    <div className="skeleton h-4 w-3/4 mb-2" />
                    <div className="skeleton h-3 w-1/4" />
                  </motion.div>
                ))}
              </div>
            )}

            {questions.length > 0 && (
              <motion.div
                className="space-y-3"
                variants={containerVariants}
                initial="hidden"
                animate="show"
              >
                {questions.map((qa, idx) => {
                  const isAnswerVisible =
                    !settings.requireAnswerReveal || revealedAnswers.has(idx);
                  return (
                    <motion.div key={qa.question_id} variants={cardVariants}>
                      <WordHighlightChat
                        contextQuestion={qa.question}
                        contextAnswer={qa.answer}
                        sessionKey={`study:${topic.id}`}
                        topicId={topic.id}
                        topicTitle={topic.title}
                        topicTrack={topic.track}
                        sectionTitle={qa.source_section || topic.sections[activeSection]?.heading}
                        mode="study"
                        selectionTargetSelector='[data-word-chat-target="true"]'
                        qaKey={`${topicId}-${qa.question_id}`}
                      >
                        <div className="udemy-card overflow-hidden">
                        <button
                          onClick={() => {
                            const highlighted = window.getSelection()?.toString().trim();
                            if (highlighted) return;
                            toggleQuestion(idx);
                          }}
                          className="w-full p-5 text-left flex items-start gap-3 hover:bg-gray-50 transition-colors"
                        >
                          <span className="flex-shrink-0 w-7 h-7 bg-udemy-purple/10 rounded-full flex items-center justify-center text-sm font-bold text-udemy-purple">
                            {idx + 1}
                          </span>
                          <div className="flex-1 min-w-0">
                            <p
                              data-word-chat-target="true"
                              className="font-medium text-[15px] leading-relaxed"
                            >
                              {qa.question}
                            </p>
                            <div className="flex items-center gap-2 mt-2">
                              <DifficultyBadge difficulty={qa.difficulty} />
                              {submittedAttempts.has(idx) && (
                                <span className="text-udemy-success text-xs flex items-center gap-1">
                                  <CheckCircle2 className="w-3.5 h-3.5" />
                                  Attempt Saved
                                </span>
                              )}
                            </div>
                          </div>
                          <motion.div
                            animate={{ rotate: expandedQ === idx ? 180 : 0 }}
                            transition={{ duration: 0.2 }}
                            className="flex-shrink-0 mt-1"
                          >
                            <ChevronDown className="w-5 h-5 text-udemy-text-muted" />
                          </motion.div>
                        </button>

                        <AnimatePresence>
                          {expandedQ === idx && (
                            <motion.div
                              variants={expandVariants}
                              initial="collapsed"
                              animate="expanded"
                              exit="collapsed"
                              className="overflow-hidden"
                            >
                              <div className="px-5 pb-5 pt-0 border-t border-udemy-border">
                                {settings.requireAnswerReveal && (
                                  <div className="mt-4 bg-white rounded-lg p-4 border border-udemy-border">
                                    <h4 className="text-xs font-bold text-udemy-text-muted uppercase tracking-wide mb-2">
                                      Your Attempt (Before Reveal)
                                    </h4>
                                    <textarea
                                      value={answerDrafts[idx] || ""}
                                      onChange={(e) =>
                                        setAnswerDrafts((prev) => ({
                                          ...prev,
                                          [idx]: e.target.value,
                                        }))
                                      }
                                      rows={4}
                                      placeholder="Type your answer in your own words..."
                                      className="w-full border border-udemy-border rounded p-3 text-sm"
                                    />
                                    <div className="mt-3">
                                      <label className="text-xs font-medium text-udemy-text-muted block mb-1">
                                        Confidence: {confidenceByIdx[idx] || 3}/5
                                      </label>
                                      <input
                                        type="range"
                                        min={1}
                                        max={5}
                                        step={1}
                                        value={confidenceByIdx[idx] || 3}
                                        onChange={(e) =>
                                          setConfidenceByIdx((prev) => ({
                                            ...prev,
                                            [idx]: Number(e.target.value),
                                          }))
                                        }
                                        className="w-full accent-udemy-purple"
                                      />
                                    </div>
                                  </div>
                                )}

                                {!isAnswerVisible && (
                                  <button
                                    onClick={() => handleReveal(idx)}
                                    className="btn-secondary mt-3"
                                  >
                                    Reveal Official Answer
                                  </button>
                                )}

                                {isAnswerVisible && (
                                  <>
                                    <div className="mt-3 bg-udemy-bg rounded-lg p-4">
                                      <h4 className="text-xs font-bold text-udemy-text-muted uppercase tracking-wide mb-2">
                                        Official Answer
                                      </h4>
                                      <div data-word-chat-target="true">
                                        <MarkdownRenderer
                                          content={qa.answer}
                                          className="text-[14px]"
                                        />
                                      </div>
                                    </div>

                                    <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                                      <div className="bg-white border border-udemy-border rounded-lg p-3">
                                        <h4 className="text-xs font-bold text-udemy-text-muted uppercase mb-1">
                                          Learning Objective
                                        </h4>
                                        <p className="text-sm">{qa.learning_objective}</p>
                                      </div>
                                      <div className="bg-white border border-udemy-border rounded-lg p-3">
                                        <h4 className="text-xs font-bold text-udemy-text-muted uppercase mb-1">
                                          Misconception Trap
                                        </h4>
                                        <p className="text-sm">{qa.misconception_trap}</p>
                                      </div>
                                      <div className="bg-white border border-udemy-border rounded-lg p-3 md:col-span-2">
                                        <h4 className="text-xs font-bold text-udemy-text-muted uppercase mb-1">
                                          Source Grounding
                                        </h4>
                                        <p className="text-xs text-udemy-text-muted mb-1">
                                          Section: {qa.source_section}
                                        </p>
                                        <p className="text-sm italic">&ldquo;{qa.source_quote}&rdquo;</p>
                                      </div>
                                    </div>
                                  </>
                                )}

                                {isAnswerVisible && !submittedAttempts.has(idx) && (
                                  <div className="mt-3 flex flex-wrap gap-2">
                                    <button
                                      onClick={() => submitAttempt(idx, true)}
                                      disabled={submittingIdx === idx}
                                      className="btn-primary disabled:opacity-60"
                                    >
                                      {submittingIdx === idx
                                        ? "Saving..."
                                        : "I got this mostly right"}
                                    </button>
                                    <button
                                      onClick={() => submitAttempt(idx, false)}
                                      disabled={submittingIdx === idx}
                                      className="btn-secondary disabled:opacity-60"
                                    >
                                      I need more practice
                                    </button>
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

                <AnimatePresence>
                  {allSubmitted && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0 }}
                      className="udemy-card p-6 text-center border-2 border-udemy-success"
                    >
                      <div className="w-16 h-16 bg-udemy-success-bg rounded-full flex items-center justify-center mx-auto mb-3">
                        <CheckCircle2 className="w-8 h-8 text-udemy-success" />
                      </div>
                      <h3 className="text-lg font-bold mb-1">Topic Complete</h3>
                      <p className="text-sm text-udemy-text-muted mb-4">
                        You submitted learning attempts for all {questions.length} questions.
                      </p>
                      <div className="flex items-center justify-center gap-3">
                        <button onClick={handleGenerate} className="btn-primary">
                          Generate More
                        </button>
                        <Link to={`/quiz/${topicId}`} className="btn-secondary">
                          Take Quiz
                        </Link>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
