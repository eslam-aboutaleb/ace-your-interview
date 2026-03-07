import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
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
  ExternalLink,
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
  fetchTopicVideosStatus,
  fetchTopicSectionVideos,
  recordTopicVideoEvent,
  updateTopicPreferences,
  generateTopicContentStream,
  isTopicContentStreamError,
  generateQuestions,
  generateQuestionsV2,
  generateQuestionsV2Stream,
  isQuestionsStreamError,
  recordLearningAttempt,
} from "@/services/api";
import { useSettingsStore } from "@/store/settingsStore";
import { useProgressStore } from "@/store/progressStore";
import { useAuthStore } from "@/store/authStore";
import DifficultyBadge from "@/components/common/DifficultyBadge";
import ProgressBar from "@/components/common/ProgressBar";
import WordHighlightChat from "@/components/common/WordHighlightChat";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import {
  normalizeEscapedMultilineText,
  normalizeEscapedSingleLineText,
} from "@/utils/textNormalization";
import type {
  TopicDetail,
  QuestionAnswerV2,
  InterviewLevel,
  LearningTrack,
  ResponseDetail,
  VideoResource,
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
  const navigate = useNavigate();
  const [topic, setTopic] = useState<TopicDetail | null>(null);
  const [questions, setQuestions] = useState<QuestionAnswerV2[]>([]);
  const [expandedQ, setExpandedQ] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [revealedAnswers, setRevealedAnswers] = useState<Set<number>>(
    new Set(),
  );
  const [submittedAttempts, setSubmittedAttempts] = useState<Set<number>>(
    new Set(),
  );
  const [answerDrafts, setAnswerDrafts] = useState<Record<number, string>>({});
  const [confidenceByIdx, setConfidenceByIdx] = useState<
    Record<number, number>
  >({});
  const [questionStartMs, setQuestionStartMs] = useState<
    Record<number, number>
  >({});
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
  const [responseDetail, setResponseDetail] =
    useState<ResponseDetail>("concise");
  const [requiresProgramming, setRequiresProgramming] = useState(false);
  const [languageOptions, setLanguageOptions] = useState<string[]>([]);
  const [preferredLanguage, setPreferredLanguage] = useState("");
  const [savingTopicPrefs, setSavingTopicPrefs] = useState(false);
  const [generatingCurriculum, setGeneratingCurriculum] = useState(false);
  const [curriculumProgress, setCurriculumProgress] = useState("");
  const [curriculumError, setCurriculumError] = useState("");
  const [curriculumSections, setCurriculumSections] = useState<
    { index: number; heading: string }[]
  >([]);
  const [topicVideosEnabled, setTopicVideosEnabled] = useState(false);
  const [topicVideosStatusLoaded, setTopicVideosStatusLoaded] = useState(false);
  const [sectionVideos, setSectionVideos] = useState<VideoResource[]>([]);
  const [videosLoading, setVideosLoading] = useState(false);
  const [videosError, setVideosError] = useState("");
  const navRef = useRef<HTMLElement | null>(null);
  const generationAbortRef = useRef<AbortController | null>(null);
  const curriculumAbortRef = useRef<AbortController | null>(null);

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
      .then((loaded) => {
        setTopic(loaded);
        setResponseDetail(loaded.response_detail || "concise");
        setRequiresProgramming(!!loaded.requires_programming);
        setLanguageOptions(loaded.language_options || []);
        setPreferredLanguage(loaded.selected_language || "");
      })
      .catch(() => setErrorMsg("Failed to load this topic. Please retry."))
      .finally(() => setLoading(false));
  }, [topicId]);

  useEffect(() => {
    if (!topicId) return;
    let active = true;
    setTopicVideosStatusLoaded(false);
    setTopicVideosEnabled(false);
    setSectionVideos([]);
    setVideosError("");
    fetchTopicVideosStatus()
      .then((status) => {
        if (!active) return;
        setTopicVideosEnabled(!!status.enabled);
      })
      .catch(() => {
        if (!active) return;
        setTopicVideosEnabled(false);
      })
      .finally(() => {
        if (!active) return;
        setTopicVideosStatusLoaded(true);
      });
    return () => {
      active = false;
    };
  }, [topicId]);

  useEffect(() => {
    return () => {
      generationAbortRef.current?.abort();
      generationAbortRef.current = null;
      curriculumAbortRef.current?.abort();
      curriculumAbortRef.current = null;
    };
  }, []);

  useEffect(() => {
    const handle = window.setTimeout(
      () => setCourseSearchQuery(courseSearchInput.trim().toLowerCase()),
      150,
    );
    return () => window.clearTimeout(handle);
  }, [courseSearchInput]);

  useEffect(() => {
    if (!topicId || !topic || !topicVideosEnabled || !topicVideosStatusLoaded) {
      setSectionVideos([]);
      setVideosError("");
      setVideosLoading(false);
      return;
    }
    const section = topic.sections[activeSection];
    if (!section) {
      setSectionVideos([]);
      setVideosError("");
      setVideosLoading(false);
      return;
    }

    let active = true;
    setVideosLoading(true);
    setVideosError("");
    fetchTopicSectionVideos(topicId, {
      section_index: activeSection,
      preferred_language: preferredLanguage || undefined,
      limit: 3,
    })
      .then((res) => {
        if (!active) return;
        if (!res.enabled || res.status === "disabled_quota_exhausted") {
          setTopicVideosEnabled(false);
          setSectionVideos([]);
          return;
        }
        setSectionVideos(Array.isArray(res.videos) ? res.videos : []);
      })
      .catch(() => {
        if (!active) return;
        setSectionVideos([]);
        setVideosError("Could not load video recommendations right now.");
      })
      .finally(() => {
        if (!active) return;
        setVideosLoading(false);
      });

    return () => {
      active = false;
    };
  }, [
    topicId,
    topic,
    activeSection,
    preferredLanguage,
    topicVideosEnabled,
    topicVideosStatusLoaded,
  ]);

  const saveTopicPreferences = useCallback(
    async (next: {
      response_detail?: ResponseDetail;
      preferred_language?: string;
    }) => {
      if (!topicId) return;
      setSavingTopicPrefs(true);
      try {
        const saved = await updateTopicPreferences(topicId, next);
        setResponseDetail(saved.response_detail);
        setRequiresProgramming(saved.requires_programming);
        setLanguageOptions(saved.language_options || []);
        setPreferredLanguage(saved.preferred_language || "");
        if (topic?.is_dynamic_topic && next.preferred_language !== undefined) {
          const refreshed = await fetchTopic(topicId);
          setTopic(refreshed);
          setResponseDetail(refreshed.response_detail || "concise");
          setRequiresProgramming(!!refreshed.requires_programming);
          setLanguageOptions(refreshed.language_options || []);
          setPreferredLanguage(refreshed.selected_language || "");
          setActiveSection(0);
        }
        return saved;
      } catch (err) {
        console.error(err);
        setErrorMsg("Failed to save topic preferences.");
        return null;
      } finally {
        setSavingTopicPrefs(false);
      }
    },
    [topicId, topic?.is_dynamic_topic],
  );

  const handleGenerateCurriculum = useCallback(
    async (forceRegenerate: boolean) => {
      if (!topicId || !topic?.is_dynamic_topic) return;
      if (!useAuthStore.getState().isAuthenticated) {
        navigate("/login");
        return;
      }
      curriculumAbortRef.current?.abort();
      const controller = new AbortController();
      curriculumAbortRef.current = controller;
      setGeneratingCurriculum(true);
      setCurriculumError("");
      setCurriculumProgress("Preparing roadmap generation...");
      setCurriculumSections([]);
      setErrorMsg("");

      const selected =
        (preferredLanguage || "").trim().toLowerCase() ||
        languageOptions[0] ||
        "python";

      try {
        await saveTopicPreferences({ preferred_language: selected });
        await generateTopicContentStream(
          topicId,
          {
            preferred_language: selected,
            ...(topic?.sections?.length
              ? { target_sections: topic.sections.length }
              : {}),
            force_regenerate: forceRegenerate,
            llm_config: {
              provider: settings.provider,
              model: settings.model,
              temperature: settings.temperature,
              max_tokens: settings.maxTokens,
            },
          },
          {
            onStart: () => {
              setCurriculumProgress(
                "Starting language-specific roadmap generation...",
              );
            },
            onProgress: (event) => {
              setCurriculumProgress(event.message || "Generating roadmap...");
            },
            onSection: (event) => {
              setCurriculumSections((prev) => [
                ...prev,
                { index: event.index, heading: event.heading },
              ]);
            },
            onDone: (event) => {
              setCurriculumProgress("Roadmap generated.");
              setTopic(event.topic);
              setResponseDetail(event.topic.response_detail || "concise");
              setRequiresProgramming(!!event.topic.requires_programming);
              setLanguageOptions(event.topic.language_options || []);
              setPreferredLanguage(event.topic.selected_language || selected);
              setActiveSection(0);
            },
            onError: (event) => {
              setCurriculumError(event.message || "Roadmap generation failed.");
            },
          },
          controller.signal,
        );
      } catch (err) {
        if (controller.signal.aborted) return;
        console.error(err);
        if (isTopicContentStreamError(err)) {
          setCurriculumError(err.message || "Roadmap generation failed.");
        } else {
          setCurriculumError("Roadmap generation failed.");
        }
      } finally {
        if (curriculumAbortRef.current === controller) {
          curriculumAbortRef.current = null;
        }
        setGeneratingCurriculum(false);
      }
    },
    [
      topicId,
      topic?.is_dynamic_topic,
      preferredLanguage,
      languageOptions,
      saveTopicPreferences,
      settings,
    ],
  );

  const handleGenerate = useCallback(async () => {
    if (!topicId || !topic) return;
    if (!useAuthStore.getState().isAuthenticated) {
      navigate("/login");
      return;
    }
    if (topic.is_dynamic_topic && !topic.content_ready) {
      setErrorMsg("Generate the problem-solving roadmap first.");
      return;
    }
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
      const basePayload = {
        topic_id: topicId,
        requested_total_count: questionCount,
        difficulty: difficulty || undefined,
        level,
        response_detail: responseDetail,
        preferred_language: requiresProgramming
          ? preferredLanguage || undefined
          : undefined,
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
      const payloadWithCount = (
        count: number,
        opts?: { includeSection?: boolean; existingQuestions?: string[] },
      ) => {
        const includeSection = opts?.includeSection !== false;
        const existingQuestions = opts?.existingQuestions || [];
        if (!includeSection) {
          const { section_title, section_content, ...rest } = basePayload;
          return { ...rest, count, existing_questions: existingQuestions };
        }
        return { ...basePayload, count, existing_questions: existingQuestions };
      };
      const questionKey = (q: Pick<QuestionAnswerV2, "question">) =>
        q.question.trim().toLowerCase().replace(/\s+/g, " ");
      const mergeUniqueQuestions = (
        current: QuestionAnswerV2[],
        incoming: QuestionAnswerV2[],
      ) => {
        const seen = new Set(current.map((q) => questionKey(q)));
        const merged = [...current];
        for (const item of incoming) {
          const key = questionKey(item);
          if (!key || seen.has(key)) continue;
          seen.add(key);
          merged.push(item);
          if (merged.length >= questionCount) break;
        }
        return merged;
      };
      const mapLegacyQuestions = (
        legacy: Awaited<ReturnType<typeof generateQuestions>>,
        startAt: number,
      ): QuestionAnswerV2[] =>
        legacy.questions.map((q, idx) => ({
          question_id: `${topicId}:legacy:${startAt + idx}`,
          topic_id: topicId,
          question: q.question,
          answer: q.answer,
          difficulty: q.difficulty,
          learning_objective: "Understand the concept and apply it in context.",
          source_section:
            activeS?.heading ||
            topic.sections[activeSection]?.heading ||
            "Topic",
          source_quote: "Legacy mode response did not include source quote.",
          misconception_trap:
            "Confusing terms without checking documentation context.",
          reasoning_summary:
            "Review the answer and compare with your own reasoning.",
        }));

      const topUpMissingQuestions = async (
        seed: QuestionAnswerV2[],
      ): Promise<QuestionAnswerV2[]> => {
        let merged = mergeUniqueQuestions([], seed).slice(0, questionCount);
        const retryModes: Array<"section" | "topic"> = activeS
          ? ["section", "topic"]
          : ["topic"];

        for (const mode of retryModes) {
          if (controller.signal.aborted || merged.length >= questionCount)
            break;
          const includeSection = mode === "section";

          for (let attempt = 0; attempt < 4; attempt += 1) {
            if (controller.signal.aborted || merged.length >= questionCount)
              break;
            const remaining = questionCount - merged.length;
            const existingQuestions = merged.map((q) => q.question);

            try {
              const res = await generateQuestionsV2(
                payloadWithCount(remaining, {
                  includeSection,
                  existingQuestions,
                }),
              );
              merged = mergeUniqueQuestions(merged, res.questions).slice(
                0,
                questionCount,
              );
              setProviderInfo({
                provider: res.provider_used,
                model: res.model_used,
                retries: res.retries_used,
                malformed: res.malformed_items_dropped,
              });
            } catch {
              // Best-effort v2 top-up. Legacy fallback below.
            }

            if (controller.signal.aborted || merged.length >= questionCount)
              break;

            try {
              const legacy = await generateQuestions(
                payloadWithCount(questionCount - merged.length, {
                  includeSection,
                  existingQuestions: merged.map((q) => q.question),
                }),
              );
              const upgraded = mapLegacyQuestions(legacy, merged.length);
              merged = mergeUniqueQuestions(merged, upgraded).slice(
                0,
                questionCount,
              );
              setProviderInfo({
                provider: legacy.provider_used,
                model: legacy.model_used,
                retries: 0,
                malformed: 0,
              });
            } catch {
              // Legacy fallback is best effort; caller handles partial results.
            }
          }
        }

        return merged;
      };

      const streamedQuestions: QuestionAnswerV2[] = [];
      try {
        await generateQuestionsV2Stream(
          payloadWithCount(questionCount, { existingQuestions: [] }),
          {
            onQuestion: (event) => {
              const merged = mergeUniqueQuestions(streamedQuestions, [
                event.question,
              ]);
              streamedQuestions.splice(0, streamedQuestions.length, ...merged);
              setQuestions([...streamedQuestions]);
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

        if (controller.signal.aborted) return;

        let finalQuestions = [...streamedQuestions];
        if (finalQuestions.length < questionCount) {
          finalQuestions = await topUpMissingQuestions(finalQuestions);
          if (controller.signal.aborted) return;
          setQuestions(finalQuestions);
        }
        if (finalQuestions.length < questionCount) {
          setErrorMsg(
            `Generated ${finalQuestions.length} of ${questionCount} questions. Try retrying or selecting another section.`,
          );
        }
      } catch (streamErr) {
        if (controller.signal.aborted) return;
        if (isQuestionsStreamError(streamErr) && streamErr.fallbackEligible) {
          const recovered = await topUpMissingQuestions(streamedQuestions);
          if (controller.signal.aborted) return;
          setQuestions(recovered);
          if (recovered.length < questionCount) {
            setErrorMsg(
              `Generated ${recovered.length} of ${questionCount} questions. Try retrying or selecting another section.`,
            );
          }
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
  }, [
    topicId,
    topic,
    activeSection,
    questionCount,
    difficulty,
    level,
    settings,
    responseDetail,
    preferredLanguage,
    requiresProgramming,
  ]);

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
    const responseTime = Math.max(
      0,
      Date.now() - (questionStartMs[idx] || Date.now()),
    );
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
  const normalizedTopicTitle = normalizeEscapedSingleLineText(
    topic?.title || "",
  );
  const normalizedTopicDescription = normalizeEscapedMultilineText(
    topic?.description || "",
  );
  const formatVideoDuration = (seconds: number): string => {
    const safe = Math.max(0, Number(seconds || 0));
    const mins = Math.floor(safe / 60);
    const hrs = Math.floor(mins / 60);
    const remMins = mins % 60;
    if (hrs > 0) {
      return `${hrs}h ${remMins}m`;
    }
    return `${mins}m`;
  };
  const handleVideoClick = useCallback(
    (video: VideoResource) => {
      if (!topicId || !topic?.sections[activeSection]) return;
      void recordTopicVideoEvent({
        event_name: "video_click",
        topic_id: topicId,
        section_index: activeSection,
        section_heading: topic.sections[activeSection].heading,
        video_id: video.video_id,
        metadata: {
          url: video.url,
          source: video.source,
        },
      });
    },
    [topicId, topic, activeSection],
  );
  const indexedSections = useMemo<IndexedSection[]>(
    () =>
      (topic?.sections || []).map((sec, index) => ({
        index,
        heading: normalizeEscapedSingleLineText(sec.heading),
        content: normalizeEscapedMultilineText(sec.content),
        searchable:
          `${normalizeEscapedSingleLineText(sec.heading)} ${normalizeEscapedMultilineText(sec.content)}`.toLowerCase(),
      })),
    [topic],
  );

  const filteredSections = useMemo(
    () =>
      !courseSearchQuery
        ? indexedSections
        : indexedSections.filter((sec) =>
            sec.searchable.includes(courseSearchQuery),
          ),
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
    ? Math.ceil(
        Math.max(navViewportHeight, sectionRowHeight) / sectionRowHeight,
      ) +
      sectionOverscan * 2
    : matchedSections;
  const virtualEnd = useVirtualizedSections
    ? Math.min(matchedSections, virtualStart + virtualVisibleCount)
    : matchedSections;
  const renderedSections = filteredSections.slice(virtualStart, virtualEnd);
  const topSpacerHeight = useVirtualizedSections
    ? virtualStart * sectionRowHeight
    : 0;
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
          <h1 className="text-2xl md:text-3xl font-bold mb-2">
            {normalizedTopicTitle}
          </h1>
          <div className="mb-2">
            <span className="inline-flex items-center rounded-full bg-udemy-purple/20 px-2.5 py-1 text-xs font-semibold text-udemy-purple-light">
              {TRACK_LABELS[topic.track] || topic.track || "Topic"}
            </span>
          </div>
          <p className="text-gray-400 text-sm max-w-2xl whitespace-pre-line">
            {normalizedTopicDescription}
          </p>
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
                    {topic.is_dynamic_topic && !topic.content_ready
                      ? "Generate the roadmap to load course sections."
                      : "No matching sections."}
                  </div>
                )}
                {topSpacerHeight > 0 && (
                  <div style={{ height: topSpacerHeight }} aria-hidden />
                )}
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
            {topic.is_dynamic_topic && (
              <div className="udemy-card p-4 sm:p-6 mb-6">
                <h2 className="text-lg font-bold mb-2">
                  Problem Solving Roadmap
                </h2>
                <p className="text-sm text-udemy-text-muted mb-4">
                  Generate a language-specific curriculum with 120 sections from
                  beginner to senior.
                </p>
                <div className="flex flex-wrap items-end gap-3">
                  <div>
                    <label className="block text-xs font-medium text-udemy-text-muted mb-1">
                      Programming Language
                    </label>
                    <select
                      value={preferredLanguage}
                      onChange={(e) => {
                        const next = e.target.value;
                        setPreferredLanguage(next);
                        void saveTopicPreferences({ preferred_language: next });
                      }}
                      className="border border-udemy-border rounded px-3 py-2 text-sm min-w-[180px]"
                      disabled={savingTopicPrefs || generatingCurriculum}
                    >
                      {languageOptions.map((lang) => (
                        <option key={lang} value={lang}>
                          {lang}
                        </option>
                      ))}
                    </select>
                  </div>
                  <button
                    onClick={() => void handleGenerateCurriculum(false)}
                    disabled={generatingCurriculum}
                    className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {topic.content_ready
                      ? "Load or Regenerate"
                      : "Generate Problem Solving Roadmap"}
                  </button>
                  {topic.content_ready && (
                    <button
                      onClick={() => void handleGenerateCurriculum(true)}
                      disabled={generatingCurriculum}
                      className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Force Regenerate
                    </button>
                  )}
                </div>
                {curriculumProgress && (
                  <p className="text-xs text-udemy-text-muted mt-3 whitespace-pre-line">
                    {normalizeEscapedMultilineText(curriculumProgress)}
                  </p>
                )}
                {curriculumError && (
                  <div className="mt-3 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                    <span className="whitespace-pre-line">
                      {normalizeEscapedMultilineText(curriculumError)}
                    </span>
                  </div>
                )}
                {generatingCurriculum && (
                  <div className="mt-4 max-h-52 overflow-y-auto friendly-scrollbar pr-1 space-y-1">
                    {curriculumSections.map((sec) => (
                      <div
                        key={`${sec.index}-${sec.heading}`}
                        className="text-xs border border-udemy-border rounded px-2 py-1"
                      >
                        {sec.index}.{" "}
                        {normalizeEscapedSingleLineText(sec.heading)}
                      </div>
                    ))}
                  </div>
                )}
                {!topic.content_ready && !generatingCurriculum && (
                  <p className="text-xs text-udemy-text-muted mt-3">
                    No roadmap generated yet for this language.
                  </p>
                )}
              </div>
            )}

            {topic.sections[activeSection] && (
              <motion.div
                key={activeSection}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="udemy-card p-4 sm:p-6 mb-6"
              >
                <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-udemy-purple" />
                  {normalizeEscapedSingleLineText(
                    topic.sections[activeSection].heading,
                  )}
                </h2>
                <MarkdownRenderer
                  content={topic.sections[activeSection].content}
                  className="text-[14px]"
                />
              </motion.div>
            )}

            {topicVideosStatusLoaded &&
              topicVideosEnabled &&
              topic.sections[activeSection] && (
                <div className="udemy-card p-4 sm:p-6 mb-6">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-udemy-text-muted uppercase tracking-wide">
                      Recommended Videos
                    </h3>
                    {videosLoading && (
                      <Loader2 className="w-4 h-4 animate-spin text-udemy-text-muted" />
                    )}
                  </div>

                  {videosError && (
                    <div className="mb-3 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                      {videosError}
                    </div>
                  )}

                  {!videosLoading &&
                    sectionVideos.length === 0 &&
                    !videosError && (
                      <p className="text-sm text-udemy-text-muted">
                        No video recommendations were found for this section
                        yet.
                      </p>
                    )}

                  <div className="space-y-2">
                    {sectionVideos.map((video) => (
                      <a
                        key={video.video_id}
                        href={video.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={() => handleVideoClick(video)}
                        className="block rounded-lg border border-udemy-border p-3 hover:bg-gray-50 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <p className="text-sm font-semibold line-clamp-2">
                              {normalizeEscapedSingleLineText(video.title)}
                            </p>
                            <p className="text-xs text-udemy-text-muted mt-1">
                              {normalizeEscapedSingleLineText(video.channel)} ·{" "}
                              {formatVideoDuration(video.duration_seconds)}
                            </p>
                          </div>
                          <ExternalLink className="w-4 h-4 text-udemy-text-muted flex-shrink-0 mt-0.5" />
                        </div>
                      </a>
                    ))}
                  </div>
                </div>
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
                    Response Detail
                  </label>
                  <select
                    value={responseDetail}
                    onChange={(e) => {
                      const next = e.target.value as ResponseDetail;
                      setResponseDetail(next);
                      void saveTopicPreferences({ response_detail: next });
                    }}
                    className="border border-udemy-border rounded px-3 py-2 text-sm"
                    disabled={savingTopicPrefs}
                  >
                    <option value="concise">Concise</option>
                    <option value="very_detailed">Very Detailed</option>
                  </select>
                </div>

                {requiresProgramming && !topic.is_dynamic_topic && (
                  <div>
                    <label className="block text-xs font-medium text-udemy-text-muted mb-1">
                      Code Language
                    </label>
                    <select
                      value={preferredLanguage}
                      onChange={(e) => {
                        const next = e.target.value;
                        setPreferredLanguage(next);
                        void saveTopicPreferences({ preferred_language: next });
                      }}
                      className="border border-udemy-border rounded px-3 py-2 text-sm min-w-[150px]"
                      disabled={savingTopicPrefs}
                    >
                      <option value="">Auto</option>
                      {languageOptions.map((lang) => (
                        <option key={lang} value={lang}>
                          {lang}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

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
                  disabled={
                    generating ||
                    (topic.is_dynamic_topic && !topic.content_ready)
                  }
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

            {generating &&
              Math.max(0, questionCount - questions.length) > 0 && (
                <div className="space-y-3">
                  {Array.from({
                    length: Math.max(0, questionCount - questions.length),
                  }).map((_, i) => (
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
                        contextQuestion={normalizeEscapedSingleLineText(
                          qa.question,
                        )}
                        contextAnswer={qa.answer}
                        sessionKey={`study:${topic.id}`}
                        topicId={topic.id}
                        topicTitle={normalizedTopicTitle}
                        topicTrack={topic.track}
                        sectionTitle={normalizeEscapedSingleLineText(
                          qa.source_section ||
                            topic.sections[activeSection]?.heading ||
                            "",
                        )}
                        mode="study"
                        showFloatingTrigger={
                          expandedQ === null ? idx === 0 : expandedQ === idx
                        }
                        floatingTriggerWord={normalizeEscapedSingleLineText(
                          qa.source_section ||
                            topic.sections[activeSection]?.heading ||
                            normalizedTopicTitle,
                        )}
                        floatingTriggerPrompt="Explain this section in an organized way with practical examples and key tradeoffs."
                        responseDetail={responseDetail}
                        preferredLanguage={preferredLanguage}
                        requiresProgramming={requiresProgramming}
                        selectionTargetSelector='[data-word-chat-target="true"]'
                        qaKey={`${topicId}-${qa.question_id}`}
                      >
                        <div className="udemy-card overflow-hidden">
                          <button
                            onClick={() => {
                              const highlighted = window
                                .getSelection()
                                ?.toString()
                                .trim();
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
                                {normalizeEscapedSingleLineText(qa.question)}
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
                                          Confidence:{" "}
                                          {confidenceByIdx[idx] || 3}/5
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
                                          <p className="text-sm">
                                            {normalizeEscapedSingleLineText(
                                              qa.learning_objective,
                                            )}
                                          </p>
                                        </div>
                                        <div className="bg-white border border-udemy-border rounded-lg p-3">
                                          <h4 className="text-xs font-bold text-udemy-text-muted uppercase mb-1">
                                            Misconception Trap
                                          </h4>
                                          <p className="text-sm">
                                            {normalizeEscapedSingleLineText(
                                              qa.misconception_trap,
                                            )}
                                          </p>
                                        </div>
                                      </div>
                                    </>
                                  )}

                                  {isAnswerVisible &&
                                    !submittedAttempts.has(idx) && (
                                      <div className="mt-3 flex flex-wrap gap-2">
                                        <button
                                          onClick={() =>
                                            submitAttempt(idx, true)
                                          }
                                          disabled={submittingIdx === idx}
                                          className="btn-primary disabled:opacity-60"
                                        >
                                          {submittingIdx === idx
                                            ? "Saving..."
                                            : "I got this mostly right"}
                                        </button>
                                        <button
                                          onClick={() =>
                                            submitAttempt(idx, false)
                                          }
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
                        You submitted learning attempts for all{" "}
                        {questions.length} questions.
                      </p>
                      <div className="flex items-center justify-center gap-3">
                        <button
                          onClick={handleGenerate}
                          className="btn-primary"
                        >
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
