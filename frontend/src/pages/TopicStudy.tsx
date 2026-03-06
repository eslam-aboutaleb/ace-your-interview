import { useEffect, useState, useCallback } from "react";
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
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import {
  pageVariants,
  pageTransition,
  expandVariants,
  containerVariants,
  cardVariants,
} from "@/utils/animations";
import { fetchTopic, generateQuestions } from "@/services/api";
import { useSettingsStore } from "@/store/settingsStore";
import { useProgressStore } from "@/store/progressStore";
import DifficultyBadge from "@/components/common/DifficultyBadge";
import ProgressBar from "@/components/common/ProgressBar";
import WordHighlightChat from "@/components/common/WordHighlightChat";
import type { TopicDetail, QuestionAnswer } from "@/types";

export default function TopicStudy() {
  const { topicId } = useParams<{ topicId: string }>();
  const [topic, setTopic] = useState<TopicDetail | null>(null);
  const [questions, setQuestions] = useState<QuestionAnswer[]>([]);
  const [expandedQ, setExpandedQ] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [revealedAnswers, setRevealedAnswers] = useState<Set<number>>(
    new Set(),
  );
  const [activeSection, setActiveSection] = useState(0);
  const [questionCount, setQuestionCount] = useState(5);
  const [difficulty, setDifficulty] = useState<string>("");
  const [providerInfo, setProviderInfo] = useState({ provider: "", model: "" });

  const settings = useSettingsStore();
  const { addAnswered, getTopicProgress, markTopicComplete } =
    useProgressStore();

  useEffect(() => {
    if (!topicId) return;
    setLoading(true);
    fetchTopic(topicId)
      .then(setTopic)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [topicId]);

  const handleGenerate = useCallback(async () => {
    if (!topicId || !topic) return;
    setGenerating(true);
    setQuestions([]);
    setExpandedQ(null);
    setRevealedAnswers(new Set());
    try {
      const activeS = topic.sections[activeSection];
      const res = await generateQuestions({
        topic_id: topicId,
        count: questionCount,
        difficulty: difficulty || undefined,
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
        // Section-scoped: pass active section content
        ...(activeS
          ? {
              section_title: activeS.heading,
              section_content: activeS.content,
            }
          : {}),
      });
      setQuestions(res.questions);
      setProviderInfo({ provider: res.provider_used, model: res.model_used });
    } catch (err) {
      console.error(err);
    } finally {
      setGenerating(false);
    }
  }, [topicId, topic, activeSection, questionCount, difficulty, settings]);

  const toggleQuestion = (idx: number) => {
    if (expandedQ === idx) {
      setExpandedQ(null);
    } else {
      setExpandedQ(idx);
      if (!revealedAnswers.has(idx)) {
        setRevealedAnswers((prev) => new Set(prev).add(idx));
        if (topicId) addAnswered(topicId, 1);
      }
    }
  };

  const allRevealed =
    questions.length > 0 && revealedAnswers.size === questions.length;

  useEffect(() => {
    if (allRevealed && topicId) {
      markTopicComplete(topicId);
    }
  }, [allRevealed, topicId, markTopicComplete]);

  const progress = topicId ? getTopicProgress(topicId) : 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-udemy-purple" />
      </div>
    );
  }

  if (!topic) {
    return (
      <div className="max-w-[1340px] mx-auto px-6 py-12 text-center">
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
      {/* Breadcrumb + title bar */}
      <div className="bg-udemy-dark text-white">
        <div className="max-w-[1340px] mx-auto px-6 py-6">
          <Link
            to="/"
            className="inline-flex items-center gap-1 text-sm text-gray-400 hover:text-white mb-3 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
            Back to all topics
          </Link>
          <h1 className="text-2xl md:text-3xl font-bold mb-2">{topic.title}</h1>
          <p className="text-gray-400 text-sm max-w-2xl">{topic.description}</p>
          <div className="mt-4">
            <ProgressBar percent={progress} className="max-w-md" />
          </div>
        </div>
      </div>

      <div className="max-w-[1340px] mx-auto px-6 py-8">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Sidebar — sections */}
          <aside className="lg:w-72 flex-shrink-0">
            <div className="udemy-card p-4 sticky top-20">
              <h3 className="text-sm font-bold text-udemy-text-muted uppercase tracking-wide mb-3">
                Course Content
              </h3>
              <nav className="space-y-1">
                {topic.sections.map((sec, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveSection(i)}
                    className={`w-full text-left px-3 py-2 rounded text-sm transition-colors flex items-center gap-2
                      ${
                        activeSection === i
                          ? "bg-udemy-purple/10 text-udemy-purple font-medium"
                          : "text-udemy-text-muted hover:bg-gray-50"
                      }`}
                  >
                    <span className="w-5 h-5 flex items-center justify-center rounded-full bg-udemy-bg text-xs font-bold flex-shrink-0">
                      {i + 1}
                    </span>
                    <span className="line-clamp-1">{sec.heading}</span>
                  </button>
                ))}
              </nav>

              {/* Quiz link */}
              {questions.length > 0 && (
                <Link
                  to={`/quiz/${topicId}`}
                  state={{ questions }}
                  className="btn-secondary w-full text-center mt-4 text-sm flex items-center justify-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Quiz Mode
                </Link>
              )}
            </div>
          </aside>

          {/* Main content */}
          <div className="flex-1 min-w-0">
            {/* Action section content */}
            {topic.sections[activeSection] && (
              <motion.div
                key={activeSection}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="udemy-card p-6 mb-6"
              >
                <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-udemy-purple" />
                  {topic.sections[activeSection].heading}
                </h2>
                <div className="markdown-content prose prose-sm max-w-none">
                  <ReactMarkdown>
                    {topic.sections[activeSection].content}
                  </ReactMarkdown>
                </div>
              </motion.div>
            )}

            {/* Generate questions controls */}
            <div className="udemy-card p-6 mb-6">
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
                    {[3, 5, 10, 15, 20, 30, 50, 75, 100].map((n) => (
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
                  Powered by{" "}
                  <span className="font-medium">{providerInfo.provider}</span>
                  {providerInfo.model && ` / ${providerInfo.model}`}
                </p>
              )}
            </div>

            {/* Loading skeleton */}
            {generating && (
              <div className="space-y-3">
                {Array.from({ length: questionCount }).map((_, i) => (
                  <motion.div
                    key={i}
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

            {/* Question list */}
            {!generating && questions.length > 0 && (
              <motion.div
                className="space-y-3"
                variants={containerVariants}
                initial="hidden"
                animate="show"
              >
                {questions.map((qa, idx) => (
                  <motion.div key={idx} variants={cardVariants}>
                    <div className="udemy-card overflow-hidden">
                      <button
                        onClick={() => toggleQuestion(idx)}
                        className="w-full p-5 text-left flex items-start gap-3 hover:bg-gray-50 transition-colors"
                      >
                        <span className="flex-shrink-0 w-7 h-7 bg-udemy-purple/10 rounded-full flex items-center justify-center text-sm font-bold text-udemy-purple">
                          {idx + 1}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-[15px] leading-relaxed">
                            {qa.question}
                          </p>
                          <div className="flex items-center gap-2 mt-2">
                            <DifficultyBadge difficulty={qa.difficulty} />
                            {revealedAnswers.has(idx) && (
                              <span className="text-udemy-success text-xs flex items-center gap-1">
                                <CheckCircle2 className="w-3.5 h-3.5" />
                                Reviewed
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
                              <WordHighlightChat
                                contextQuestion={qa.question}
                                contextAnswer={qa.answer}
                                qaKey={`${topicId}-${idx}`}
                              >
                                {/* Question — selectable for highlight chat */}
                                <div className="mt-4 bg-udemy-purple/5 rounded-lg p-4 border border-udemy-purple/10">
                                  <h4 className="text-xs font-bold text-udemy-text-muted uppercase tracking-wide mb-2">
                                    Question
                                  </h4>
                                  <p className="text-[14px] leading-relaxed font-medium select-text cursor-text">
                                    {qa.question}
                                  </p>
                                </div>

                                {/* Answer */}
                                <div className="mt-3 bg-udemy-bg rounded-lg p-4">
                                  <h4 className="text-xs font-bold text-udemy-text-muted uppercase tracking-wide mb-2">
                                    Answer
                                  </h4>
                                  <div className="markdown-content text-[14px] leading-relaxed">
                                    <ReactMarkdown>{qa.answer}</ReactMarkdown>
                                  </div>
                                </div>
                              </WordHighlightChat>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </motion.div>
                ))}

                {/* Completion banner */}
                <AnimatePresence>
                  {allRevealed && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0 }}
                      className="udemy-card p-6 text-center border-2 border-udemy-success"
                    >
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{
                          type: "spring",
                          stiffness: 400,
                          damping: 15,
                          delay: 0.2,
                        }}
                        className="w-16 h-16 bg-udemy-success-bg rounded-full flex items-center justify-center mx-auto mb-3"
                      >
                        <CheckCircle2 className="w-8 h-8 text-udemy-success" />
                      </motion.div>
                      <h3 className="text-lg font-bold mb-1">
                        Topic Complete! 🎉
                      </h3>
                      <p className="text-sm text-udemy-text-muted mb-4">
                        You've reviewed all {questions.length} questions.
                      </p>
                      <div className="flex items-center justify-center gap-3">
                        <button
                          onClick={handleGenerate}
                          className="btn-primary"
                        >
                          Generate More
                        </button>
                        <Link to="/" className="btn-secondary">
                          Back to Dashboard
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
