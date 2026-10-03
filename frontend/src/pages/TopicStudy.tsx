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
  saveTopicProgress,
  autosaveTopicProgress,
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
import {
  loadTopicStudyQuestionCache,
  persistTopicStudyQuestionCache,
} from "@/utils/topicStudyQuestionCache";
import type {
  TopicDetail,
  QuestionAnswerV2,
  InterviewLevel,
  LearningTrack,
  ResponseDetail,
  VideoResource,
  SaveProgressPayload,
} from "@/types";
import type {
  TopicStudyProviderInfo,
  TopicStudyQuestionCacheEntry,
  TopicStudyQuestionSignature,
} from "@/utils/topicStudyQuestionCache";

const TRACK_LABELS: Record<LearningTrack, string> = {
  backend: "Backend",
  frontend: "Frontend",
  system_design: "System Design",
  ai_stack: "AI Stack",
};

type IndexedSection = {
  index: number;
  heading: string;
  content: string;
  searchable: string;
};

type InFlightQuestionRun = {
  controller: AbortController;
  runId: number;
};

const DEFAULT_PROVIDER_INFO: TopicStudyProviderInfo = {
  provider: "",
  model: "",
  retries: 0,
  malformed: 0,
};

// keepalive bodies are capped at 64 KB by the browser; 60 questions keeps the
// auto-save beacon comfortably inside that budget.
const AUTOSAVE_QUESTION_LIMIT = 60;

function stableHash(input: string): string {
  let hash = 5381;
  for (let idx = 0; idx < input.length; idx += 1) {
    hash = ((hash << 5) + hash) ^ input.charCodeAt(idx);
  }
  return (hash >>> 0).toString(36);
}

function makeSectionFingerprint(heading: string, content: string): string {
  return stableHash(`${heading}::${content}`);
}

function buildQuestionSignatureKey(signature: TopicStudyQuestionSignature): string {
  return [
    encodeURIComponent(signature.topicId),
    String(signature.sectionIndex),
    encodeURIComponent(signature.sectionFingerprint),
    String(signature.questionCount),
    encodeURIComponent(signature.difficulty || ""),
    signature.level,
    signature.responseDetail,
    encodeURIComponent(signature.preferredLanguage || ""),
    signature.requiresProgramming ? "1" : "0",
    encodeURIComponent(signature.llmProvider || ""),
    encodeURIComponent(signature.llmModel || ""),
    String(signature.llmTemperature),
    String(signature.llmMaxTokens),
  ].join(":");
}

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
  // Per-question self-assessment outcome. submittedAttempts only records that an
  // attempt exists, but the stored summary needs right-vs-needs-practice counts.
  const [outcomeByIdx, setOutcomeByIdx] = useState<Record<number, boolean>>({});
  const [visitedSections, setVisitedSections] = useState<string[]>([]);
  const [saveProgressStatus, setSaveProgressStatus] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");
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
  const [providerInfo, setProviderInfo] = useState<TopicStudyProviderInfo>(
    DEFAULT_PROVIDER_INFO,
  );
  const [errorMsg, setErrorMsg] = useState("");
  const [responseDetail, setResponseDetail] =
    useState<ResponseDetail>("very_detailed");
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
  const curriculumAbortRef = useRef<AbortController | null>(null);
  const entriesByKeyRef = useRef<Record<string, TopicStudyQuestionCacheEntry>>(
    {},
  );
  const inFlightByKeyRef = useRef<Map<string, InFlightQuestionRun>>(new Map());
  const runCounterRef = useRef(0);
  const activeQuestionKeyRef = useRef("");

  const settings = useSettingsStore();
  const {
    getTopicProgress,
    markTopicComplete,
    recordAttempt,
    setMastery,
    addAnswered,
  } = useProgressStore();

  const activeSectionSnapshot = topic?.sections[activeSection] || null;

  // A section counts as covered once the learner has opened it; the stored
  // summary uses the untouched sections to decide what to ask next.
  useEffect(() => {
    const heading = activeSectionSnapshot?.heading?.trim();
    if (!heading) return;
    setVisitedSections((prev) => (prev.includes(heading) ? prev : [...prev, heading]));
  }, [activeSectionSnapshot?.heading]);

  const buildProgressPayload = useCallback(
    (questionLimit?: number): SaveProgressPayload => {
      const rows = questions.map((qa, idx) => ({
        question_id: qa.question_id,
        question: qa.question,
        difficulty: qa.difficulty,
        revealed: revealedAnswers.has(idx),
        is_correct: outcomeByIdx[idx] ?? null,
        confidence: confidenceByIdx[idx] || 0,
      }));
      return {
        topic_title: topic?.title || "",
        // The beacon body is capped at 64 KB by the browser, so the auto-save
        // path sends only the most recent slice.
        questions: questionLimit ? rows.slice(-questionLimit) : rows,
        sections: visitedSections,
        preferred_language: preferredLanguage,
      };
    },
    [
      questions,
      revealedAnswers,
      outcomeByIdx,
      confidenceByIdx,
      topic,
      visitedSections,
      preferredLanguage,
    ],
  );

  const handleSaveProgress = useCallback(async () => {
    if (!topicId || saveProgressStatus === "saving") return;
    setSaveProgressStatus("saving");
    try {
      await saveTopicProgress(topicId, buildProgressPayload());
      setSaveProgressStatus("saved");
    } catch (err) {
      console.error(err);
      setSaveProgressStatus("error");
    }
  }, [topicId, saveProgressStatus, buildProgressPayload]);

  useEffect(() => {
    if (saveProgressStatus === "idle") return;
    const timer = window.setTimeout(() => setSaveProgressStatus("idle"), 2500);
    return () => window.clearTimeout(timer);
  }, [saveProgressStatus]);

  const saveProgressLabel =
    saveProgressStatus === "saving"
      ? "Saving..."
      : saveProgressStatus === "saved"
        ? "Progress saved"
        : saveProgressStatus === "error"
          ? "Save failed"
          : "Save progress";

  // Two ways a study session ends without the learner pressing the button:
  // closing/backgrounding the tab (`pagehide`) and navigating away inside the SPA
  // (component unmount — React does not fire `pagehide` for route changes).
  // Registered once per mount so a state change cannot trigger a summary call;
  // the latest payload is read from a ref at fire time.
  const progressPayloadRef = useRef<SaveProgressPayload | null>(null);
  progressPayloadRef.current = buildProgressPayload(AUTOSAVE_QUESTION_LIMIT);
  const autoSaveSentRef = useRef(false);

  useEffect(() => {
    if (!topicId) return;
    const fireAutoSave = () => {
      if (autoSaveSentRef.current) return;
      const payload = progressPayloadRef.current;
      // Nothing generated yet: saving would create an empty progress document
      // and spend a summary call on a topic the learner has not worked on.
      if (!payload || payload.questions.length === 0) return;
      autoSaveSentRef.current = true;
      void autosaveTopicProgress(topicId, payload);
    };
    window.addEventListener("pagehide", fireAutoSave);
    return () => {
      window.removeEventListener("pagehide", fireAutoSave);
      fireAutoSave();
    };
  }, [topicId]);

  const activeQuestionSignature = useMemo<TopicStudyQuestionSignature | null>(
    () => {
      if (!topicId || !topic || !activeSectionSnapshot) return null;
      return {
        topicId,
        sectionIndex: activeSection,
        sectionFingerprint: makeSectionFingerprint(
          activeSectionSnapshot.heading,
          activeSectionSnapshot.content,
        ),
        questionCount,
        difficulty: difficulty || "",
        level,
        responseDetail,
        preferredLanguage: requiresProgramming ? preferredLanguage || "" : "",
        requiresProgramming,
        llmProvider: settings.provider,
        llmModel: settings.model,
        llmTemperature: settings.temperature,
        llmMaxTokens: settings.maxTokens,
      };
    },
    [
      topicId,
      topic,
      activeSectionSnapshot,
      activeSection,
      questionCount,
      difficulty,
      level,
      responseDetail,
      preferredLanguage,
      requiresProgramming,
      settings.provider,
      settings.model,
      settings.temperature,
      settings.maxTokens,
    ],
  );
  const activeQuestionKey = useMemo(
    () =>
      activeQuestionSignature
        ? buildQuestionSignatureKey(activeQuestionSignature)
        : "",
    [activeQuestionSignature],
  );

  const applyEntryToActiveView = useCallback(
    (entry: TopicStudyQuestionCacheEntry | null) => {
      if (!entry) {
        setQuestions([]);
        setGenerating(false);
        setProviderInfo(DEFAULT_PROVIDER_INFO);
        setErrorMsg("");
        return;
      }
      setQuestions(entry.questions);
      setGenerating(entry.status === "generating");
      setProviderInfo(entry.providerInfo);
      setErrorMsg(entry.errorMsg || "");
    },
    [],
  );

  const persistEntries = useCallback(() => {
    if (!topicId) return;
    persistTopicStudyQuestionCache(topicId, entriesByKeyRef.current);
  }, [topicId]);

  const updateEntryByKey = useCallback(
    (
      key: string,
      updater: (
        previous: TopicStudyQuestionCacheEntry | undefined,
      ) => TopicStudyQuestionCacheEntry | undefined,
    ) => {
      const next = updater(entriesByKeyRef.current[key]);
      if (!next) {
        delete entriesByKeyRef.current[key];
      } else {
        entriesByKeyRef.current[key] = next;
      }
      persistEntries();
      if (activeQuestionKeyRef.current === key) {
        applyEntryToActiveView(next ?? null);
      }
    },
    [applyEntryToActiveView, persistEntries],
  );

  const isRunCurrent = useCallback((key: string, runId: number) => {
    const run = inFlightByKeyRef.current.get(key);
    return !!run && run.runId === runId;
  }, []);

  const abortAllQuestionRuns = useCallback(() => {
    for (const run of inFlightByKeyRef.current.values()) {
      run.controller.abort();
    }
    inFlightByKeyRef.current.clear();
  }, []);

  useEffect(() => {
    abortAllQuestionRuns();
    entriesByKeyRef.current = topicId ? loadTopicStudyQuestionCache(topicId) : {};
    activeQuestionKeyRef.current = "";
    applyEntryToActiveView(null);
    setExpandedQ(null);
    setRevealedAnswers(new Set());
    setSubmittedAttempts(new Set());
    setAnswerDrafts({});
    setConfidenceByIdx({});
    setQuestionStartMs({});
    setSubmittingIdx(null);
  }, [topicId, abortAllQuestionRuns, applyEntryToActiveView]);

  useEffect(() => {
    activeQuestionKeyRef.current = activeQuestionKey;
    const entry =
      activeQuestionKey && entriesByKeyRef.current[activeQuestionKey]
        ? entriesByKeyRef.current[activeQuestionKey]
        : null;
    applyEntryToActiveView(entry);
    setExpandedQ(null);
    setRevealedAnswers(new Set());
    setSubmittedAttempts(new Set());
    setAnswerDrafts({});
    setConfidenceByIdx({});
    setQuestionStartMs({});
    setSubmittingIdx(null);
  }, [activeQuestionKey, applyEntryToActiveView]);

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
        setResponseDetail(loaded.response_detail || "very_detailed");
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
      abortAllQuestionRuns();
      curriculumAbortRef.current?.abort();
      curriculumAbortRef.current = null;
    };
  }, [abortAllQuestionRuns]);

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
          setResponseDetail(refreshed.response_detail || "very_detailed");
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
              setTopic((prev) => {
                if (!prev) return prev;
                const sectionIndex = Math.max(0, event.index - 1);
                const nextSections = [...(prev.sections || [])];
                while (nextSections.length <= sectionIndex) {
                  nextSections.push({ heading: "", content: "" });
                }
                nextSections[sectionIndex] = {
                  heading: event.heading,
                  content: event.content,
                };
                return {
                  ...prev,
                  sections: nextSections,
                };
              });
            },
            onDone: (event) => {
              setCurriculumProgress("Roadmap generated.");
              setTopic(event.topic);
              setResponseDetail(event.topic.response_detail || "very_detailed");
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
    if (!topicId || !topic || !activeQuestionSignature) return;
    if (!useAuthStore.getState().isAuthenticated) {
      navigate("/login");
      return;
    }
    if (topic.is_dynamic_topic && !topic.content_ready) {
      setErrorMsg("Generate the problem-solving roadmap first.");
      return;
    }

    const activeS = topic.sections[activeSection];
    if (!activeS) {
      setErrorMsg("Pick a section before generating questions.");
      return;
    }

    const cacheKey = buildQuestionSignatureKey(activeQuestionSignature);
    const priorRun = inFlightByKeyRef.current.get(cacheKey);
    priorRun?.controller.abort();

    const runId = runCounterRef.current + 1;
    runCounterRef.current = runId;
    const controller = new AbortController();
    inFlightByKeyRef.current.set(cacheKey, { controller, runId });

    const startedAt = Date.now();
    updateEntryByKey(cacheKey, () => ({
      key: cacheKey,
      signature: activeQuestionSignature,
      status: "generating",
      questions: [],
      providerInfo: DEFAULT_PROVIDER_INFO,
      errorMsg: "",
      startedAt,
      completedAt: 0,
      updatedAt: startedAt,
    }));
    setExpandedQ(null);
    setRevealedAnswers(new Set());
    setSubmittedAttempts(new Set());
    setAnswerDrafts({});
    setConfidenceByIdx({});
    setQuestionStartMs({});

    const isCurrentRun = () => isRunCurrent(cacheKey, runId);
    const updateRunEntry = (
      producer: (
        previous: TopicStudyQuestionCacheEntry,
      ) => TopicStudyQuestionCacheEntry,
    ) => {
      if (!isCurrentRun()) return;
      updateEntryByKey(cacheKey, (previous) => {
        const base: TopicStudyQuestionCacheEntry =
          previous ??
          ({
            key: cacheKey,
            signature: activeQuestionSignature,
            status: "generating",
            questions: [],
            providerInfo: DEFAULT_PROVIDER_INFO,
            errorMsg: "",
            startedAt,
            completedAt: 0,
            updatedAt: Date.now(),
          } satisfies TopicStudyQuestionCacheEntry);
        return {
          ...producer(base),
          updatedAt: Date.now(),
        };
      });
    };
    const finalizeRun = (
      status: "ready" | "error",
      finalQuestions: QuestionAnswerV2[],
      finalProviderInfo: TopicStudyProviderInfo,
      finalErrorMsg: string,
    ) => {
      updateRunEntry((previous) => ({
        ...previous,
        status,
        questions: finalQuestions,
        providerInfo: finalProviderInfo,
        errorMsg: finalErrorMsg,
        completedAt: Date.now(),
      }));
    };

    try {
      const targetCount = questionCount;
      const basePayload = {
        topic_id: topicId,
        requested_total_count: targetCount,
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
        section_title: activeS.heading,
        section_content: activeS.content,
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
      const normalizeQuestionKey = (q: Pick<QuestionAnswerV2, "question">) =>
        q.question.trim().toLowerCase().replace(/\s+/g, " ");
      const mergeUniqueQuestions = (
        current: QuestionAnswerV2[],
        incoming: QuestionAnswerV2[],
      ) => {
        const seen = new Set(current.map((q) => normalizeQuestionKey(q)));
        const merged = [...current];
        for (const item of incoming) {
          const key = normalizeQuestionKey(item);
          if (!key || seen.has(key)) continue;
          seen.add(key);
          merged.push(item);
          if (merged.length >= targetCount) break;
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
          learning_objective:
            "After this question, the learner should be able to explain the concept and apply it in context.",
          source_section: activeS.heading || "Topic",
          source_quote: "Legacy mode response did not include source quote.",
          reasoning_summary:
            "Review the core constraint first, then compare it with your own reasoning.",
        }));

      const topUpMissingQuestions = async (
        seed: QuestionAnswerV2[],
      ): Promise<QuestionAnswerV2[]> => {
        let merged = mergeUniqueQuestions([], seed).slice(0, targetCount);
        const retryModes: Array<"section" | "topic"> = ["section", "topic"];

        for (const mode of retryModes) {
          if (!isCurrentRun() || controller.signal.aborted || merged.length >= targetCount) {
            break;
          }
          const includeSection = mode === "section";

          for (let attempt = 0; attempt < 4; attempt += 1) {
            if (!isCurrentRun() || controller.signal.aborted || merged.length >= targetCount) {
              break;
            }
            const remaining = targetCount - merged.length;
            const existingQuestions = merged.map((q) => q.question);

            try {
              const res = await generateQuestionsV2(
                payloadWithCount(remaining, {
                  includeSection,
                  existingQuestions,
                }),
              );
              if (!isCurrentRun()) break;
              merged = mergeUniqueQuestions(merged, res.questions).slice(0, targetCount);
              updateRunEntry((previous) => ({
                ...previous,
                providerInfo: {
                  provider: res.provider_used,
                  model: res.model_used,
                  retries: res.retries_used,
                  malformed: res.malformed_items_dropped,
                },
              }));
            } catch {
              // Best-effort v2 top-up. Legacy fallback below.
            }

            if (!isCurrentRun() || controller.signal.aborted || merged.length >= targetCount) {
              break;
            }

            try {
              const legacy = await generateQuestions(
                payloadWithCount(targetCount - merged.length, {
                  includeSection,
                  existingQuestions: merged.map((q) => q.question),
                }),
              );
              if (!isCurrentRun()) break;
              const upgraded = mapLegacyQuestions(legacy, merged.length);
              merged = mergeUniqueQuestions(merged, upgraded).slice(0, targetCount);
              updateRunEntry((previous) => ({
                ...previous,
                providerInfo: {
                  provider: legacy.provider_used,
                  model: legacy.model_used,
                  retries: 0,
                  malformed: 0,
                },
              }));
            } catch {
              // Legacy fallback is best effort; caller handles partial results.
            }
          }
        }

        return merged;
      };

      const streamedQuestions: QuestionAnswerV2[] = [];
      let latestProviderInfo = DEFAULT_PROVIDER_INFO;
      try {
        await generateQuestionsV2Stream(
          payloadWithCount(targetCount, { existingQuestions: [] }),
          {
            onQuestion: (event) => {
              if (!isCurrentRun()) return;
              const merged = mergeUniqueQuestions(streamedQuestions, [
                event.question,
              ]);
              streamedQuestions.splice(0, streamedQuestions.length, ...merged);
              updateRunEntry((previous) => ({
                ...previous,
                status: "generating",
                questions: [...streamedQuestions],
              }));
            },
            onDone: (event) => {
              if (!isCurrentRun()) return;
              latestProviderInfo = {
                provider: event.provider_used,
                model: event.model_used,
                retries: event.retries_used,
                malformed: event.malformed_items_dropped,
              };
              updateRunEntry((previous) => ({
                ...previous,
                providerInfo: latestProviderInfo,
              }));
            },
            onError: (event) => {
              if (!isCurrentRun()) return;
              const message = event.message || "Question generation failed.";
              updateRunEntry((previous) => ({
                ...previous,
                status: "error",
                errorMsg: message,
              }));
            },
          },
          controller.signal,
        );

        if (!isCurrentRun() || controller.signal.aborted) return;

        let finalQuestions = [...streamedQuestions];
        if (finalQuestions.length < targetCount) {
          finalQuestions = await topUpMissingQuestions(finalQuestions);
          if (!isCurrentRun() || controller.signal.aborted) return;
        }
        if (finalQuestions.length < targetCount) {
          finalizeRun(
            "error",
            finalQuestions,
            latestProviderInfo,
            `Generated ${finalQuestions.length} of ${targetCount} questions. Try retrying or selecting another section.`,
          );
          return;
        }
        finalizeRun("ready", finalQuestions, latestProviderInfo, "");
      } catch (streamErr) {
        if (!isCurrentRun() || controller.signal.aborted) return;
        if (isQuestionsStreamError(streamErr) && streamErr.fallbackEligible) {
          const recovered = await topUpMissingQuestions(streamedQuestions);
          if (!isCurrentRun() || controller.signal.aborted) return;
          if (recovered.length < targetCount) {
            finalizeRun(
              "error",
              recovered,
              latestProviderInfo,
              `Generated ${recovered.length} of ${targetCount} questions. Try retrying or selecting another section.`,
            );
            return;
          }
          finalizeRun("ready", recovered, latestProviderInfo, "");
          return;
        }
        throw streamErr;
      }
    } catch (err) {
      if (!isCurrentRun() || controller.signal.aborted) return;
      console.error(err);
      const message = isQuestionsStreamError(err)
        ? err.message || "Question generation failed."
        : "Question generation failed. Check LLM settings/connection and try again.";
      finalizeRun("error", [], DEFAULT_PROVIDER_INFO, message);
    } finally {
      const current = inFlightByKeyRef.current.get(cacheKey);
      if (current && current.runId === runId) {
        inFlightByKeyRef.current.delete(cacheKey);
      }
    }
  }, [
    topicId,
    topic,
    activeSection,
    activeQuestionSignature,
    questionCount,
    difficulty,
    level,
    responseDetail,
    preferredLanguage,
    requiresProgramming,
    settings.provider,
    settings.model,
    settings.temperature,
    settings.maxTokens,
    navigate,
    updateEntryByKey,
    isRunCurrent,
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
      setOutcomeByIdx((prev) => ({ ...prev, [idx]: isCorrect }));
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
  const isProblemSolvingTopic =
    topic?.id === "00-problem-solving-and-algorithms";
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
                    <option value="very_detailed">Interview Pass</option>
                    <option value="concise">Focused Review</option>
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

                <button
                  onClick={handleSaveProgress}
                  disabled={saveProgressStatus === "saving"}
                  className="btn-secondary flex items-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {saveProgressStatus === "saving" ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <CheckCircle2 className="w-4 h-4" />
                  )}
                  {saveProgressLabel}
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
                                        {isProblemSolvingTopic &&
                                          qa.reasoning_summary && (
                                            <div className="mb-3 rounded-lg border border-udemy-purple/20 bg-white px-3 py-2">
                                              <p className="text-[11px] font-bold uppercase tracking-wide text-udemy-purple mb-1">
                                                Key reasoning takeaway
                                              </p>
                                              <p className="text-sm text-udemy-text whitespace-pre-line">
                                                {normalizeEscapedMultilineText(
                                                  qa.reasoning_summary,
                                                )}
                                              </p>
                                            </div>
                                          )}
                                        <div data-word-chat-target="true">
                                          <MarkdownRenderer
                                            content={qa.answer}
                                            className="text-[14px]"
                                          />
                                        </div>
                                      </div>

                                      <div className="mt-3">
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
                        <button
                          onClick={handleSaveProgress}
                          disabled={saveProgressStatus === "saving"}
                          className="btn-secondary disabled:opacity-60"
                        >
                          {saveProgressLabel}
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
