import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Loader2,
  MessageSquare,
  Sparkles,
  Clock,
  ArrowRight,
  Upload,
  FileText,
  Building2,
  Trash2,
  CheckCircle2,
  AlertCircle,
  Mic,
  MicOff,
  Radio,
  PhoneOff,
  AlertTriangle,
} from "lucide-react";
import {
  pageVariants,
  pageTransition,
  containerVariants,
  cardVariants,
} from "@/utils/animations";
import {
  createInterviewSession,
  createInterviewSessionStream,
  fetchLearnerProfile,
  fetchInterviewSessions,
  isInterviewStreamError,
  uploadResume,
  deleteResume,
  analyzeJobDescription,
  fetchCompanyPacks,
} from "@/services/api";
import { useSettingsStore } from "@/store/settingsStore";
import { useVoice } from "@/hooks/useVoice";
import { normalizeEscapedMultilineText } from "@/utils/textNormalization";
import type {
  CompanyPackInfo,
  FeedbackMode,
  InterviewLevel,
  InterviewSession,
  InterviewerStyle,
  InterviewType,
  JDAnalysisResponse,
  LearningTrack,
  ResumeSkillProfile,
} from "@/types";

function interviewStartErrorMessage(error: unknown): string {
  const detail = (
    error as {
      response?: {
        data?: {
          detail?: unknown;
        };
      };
      message?: string;
    }
  )?.response?.data?.detail;

  if (typeof detail === "string" && detail.trim()) {
    return detail.trim();
  }
  if (
    typeof detail === "object" &&
    detail &&
    typeof (detail as { message?: unknown }).message === "string" &&
    (detail as { message: string }).message.trim()
  ) {
    return (detail as { message: string }).message.trim();
  }

  const message = (error as { message?: unknown })?.message;
  if (typeof message === "string" && message.trim()) {
    return message.trim();
  }

  return "Could not start mock interview.";
}

const TRACK_LABELS: Record<LearningTrack, string> = {
  backend: "Backend",
  frontend: "Frontend",
  system_design: "System Design",
  ai_stack: "AI Stack",
};

export default function InterviewSetup() {
  const navigate = useNavigate();
  const settings = useSettingsStore();
  const voice = useVoice();

  const [track, setTrack] = useState<LearningTrack>("backend");
  const [level, setLevel] = useState<InterviewLevel>("mid");
  const [interviewType, setInterviewType] = useState<InterviewType>("mixed");
  const [turnCount, setTurnCount] = useState(5);
  const [interviewerStyle, setInterviewerStyle] =
    useState<InterviewerStyle>("neutral");
  const [feedbackMode, setFeedbackMode] = useState<FeedbackMode>("concise");
  const [targetRole, setTargetRole] = useState("");
  const [jobDescriptionText, setJobDescriptionText] = useState("");
  const [resumeSummaryText, setResumeSummaryText] = useState("");
  const [focusAreasRaw, setFocusAreasRaw] = useState("");
  const [company, setCompany] = useState("");

  const [companies, setCompanies] = useState<CompanyPackInfo[]>([]);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [resumeProfile, setResumeProfile] =
    useState<ResumeSkillProfile | null>(null);
  const [uploadingResume, setUploadingResume] = useState(false);
  const [deletingResume, setDeletingResume] = useState(false);
  const [attachResume, setAttachResume] = useState(true);
  const [jdAnalyzing, setJdAnalyzing] = useState(false);
  const [jdAnalysis, setJdAnalysis] = useState<JDAnalysisResponse | null>(
    null,
  );
  const [resumeError, setResumeError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [starting, setStarting] = useState(false);
  const [startProgress, setStartProgress] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [recent, setRecent] = useState<InterviewSession[]>([]);
  const [loadingRecent, setLoadingRecent] = useState(true);
  const [voiceInterviewActive, setVoiceInterviewActive] = useState(false);
  const [voiceInterviewError, setVoiceInterviewError] = useState("");

  useEffect(() => {
    fetchInterviewSessions(6, 0)
      .then((res) => setRecent(res.sessions))
      .catch(() => setRecent([]))
      .finally(() => setLoadingRecent(false));
  }, []);

  useEffect(() => {
    fetchCompanyPacks()
      .then((res) => setCompanies(res.packs))
      .catch(() => setCompanies([]));
  }, []);

  useEffect(() => {
    fetchLearnerProfile()
      .then((profile) => {
        if (profile.primary_track) setTrack(profile.primary_track);
        if (profile.current_level) setLevel(profile.current_level);
        if (profile.target_role)
          setTargetRole((prev) => prev || profile.target_role);
      })
      .catch(() => {
        // learner profile is optional here
      });
  }, []);

  const focusAreas = focusAreasRaw
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean)
    .slice(0, 8);

  const llmConfig = {
    provider: settings.provider,
    model: settings.model,
    temperature: settings.temperature,
    max_tokens: settings.maxTokens,
  };

  const handleResumeUpload = async () => {
    if (!resumeFile) return;
    setUploadingResume(true);
    setResumeError("");
    try {
      const res = await uploadResume(resumeFile, llmConfig);
      setResumeProfile(res.profile);
      if (res.profile.skills.length && !resumeSummaryText) {
        setResumeSummaryText(
          [
            res.profile.skills.length
              ? `Skills: ${res.profile.skills.join(", ")}`
              : "",
            res.profile.roles.length
              ? `Roles: ${res.profile.roles.join(", ")}`
              : "",
            typeof res.profile.experience_years === "number"
              ? `Experience: ${res.profile.experience_years} years`
              : "",
          ]
            .filter(Boolean)
            .join("\n"),
        );
      }
    } catch (err) {
      const detail = (
        err as {
          response?: { data?: { detail?: unknown } };
          message?: string;
        }
      )?.response?.data?.detail;
      setResumeError(
        typeof detail === "string"
          ? detail
          : typeof detail === "object" && detail && "message" in detail
            ? String((detail as { message?: unknown }).message)
            : "Could not parse the resume.",
      );
    } finally {
      setUploadingResume(false);
    }
  };

  const handleResumeDelete = async () => {
    setDeletingResume(true);
    try {
      await deleteResume();
      setResumeProfile(null);
      setResumeFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch {
      setResumeError("Could not delete the resume profile.");
    } finally {
      setDeletingResume(false);
    }
  };

  const handleJdAnalysis = async () => {
    const text = jobDescriptionText.trim();
    if (!text) return;
    setJdAnalyzing(true);
    try {
      const analysis = await analyzeJobDescription(text);
      setJdAnalysis(analysis);
    } catch {
      setJdAnalysis(null);
    } finally {
      setJdAnalyzing(false);
    }
  };

  const applyRecommendedFocusAreas = () => {
    if (!jdAnalysis) return;
    const recommended = jdAnalysis.recommended_focus_areas.slice(0, 8);
    if (!recommended.length) return;
    const existing = new Set(focusAreas);
    const merged = [
      ...focusAreas,
      ...recommended.filter((area) => !existing.has(area)),
    ].slice(0, 8);
    setFocusAreasRaw(merged.join(", "));
  };

  const handleStart = async () => {
    setStarting(true);
    setErrorMsg("");
    setStartProgress("Preparing interview session...");
    const payload = {
      track,
      level,
      interview_type: interviewType,
      turn_count: turnCount,
      target_role: targetRole.trim(),
      interviewer_style: interviewerStyle,
      feedback_mode: feedbackMode,
      job_description_text: jobDescriptionText.trim(),
      resume_summary_text: resumeSummaryText.trim(),
      focus_areas: focusAreas,
      company,
      resume_profile: attachResume && resumeProfile !== null,
      jd_text: jobDescriptionText.trim(),
      llm_config: llmConfig,
    };
    try {
      const done = await createInterviewSessionStream(payload, {
        onStart: (event) => {
          setStartProgress(event.message || "Creating interview session...");
        },
        onProgress: (event) => {
          setStartProgress(
            event.message || "Generating first interview question...",
          );
        },
        onDone: () => {
          setStartProgress("Interview session ready.");
        },
        onError: (event) => {
          setErrorMsg(event.message || "Could not start mock interview.");
        },
      });
      navigate(`/interview/${done.session.session_id}`);
    } catch (streamErr) {
      if (isInterviewStreamError(streamErr) && !streamErr.fallbackEligible) {
        setErrorMsg(streamErr.message || "Could not start mock interview.");
        return;
      }
      try {
        const fallback = await createInterviewSession(payload);
        navigate(`/interview/${fallback.session.session_id}`);
      } catch (fallbackErr) {
        setErrorMsg(interviewStartErrorMessage(fallbackErr));
      }
    } finally {
      setStartProgress("");
      setStarting(false);
    }
  };

  const handleStartVoiceInterview = async () => {
    const cloudAvailable =
      voice.config?.available_tiers?.includes("cloud") ?? false;
    if (!voice.voiceEnabled || !cloudAvailable) {
      setVoiceInterviewError(
        "Voice interviews need the cloud voice tier. Enable it in settings.",
      );
      return;
    }
    setVoiceInterviewError("");
    if (voice.currentTier !== "cloud") {
      voice.setCurrentTier("cloud");
    }
    const config = {
      track,
      level,
      interview_type: interviewType,
      turn_count: turnCount,
      target_role: targetRole.trim(),
      interviewer_style: interviewerStyle,
      feedback_mode: feedbackMode,
      job_description_text: jobDescriptionText.trim(),
      resume_summary_text: resumeSummaryText.trim(),
      focus_areas: focusAreas,
      company,
      resume_profile: attachResume && resumeProfile !== null,
      jd_text: jobDescriptionText.trim(),
      llm_config: llmConfig,
    };
    setVoiceInterviewActive(true);
    await voice.startInterview(config);
  };

  const handleExitVoiceInterview = () => {
    voice.stopSession();
    setVoiceInterviewActive(false);
    setVoiceInterviewError("");
  };

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
    >
      <div className="bg-udemy-dark text-white">
        <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-8">
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
            <MessageSquare className="w-7 h-7 text-udemy-purple-light" />
            Mock Interview
          </h1>
          <p className="text-gray-400 mt-2">
            Configure a personalized interview session and practice
            turn-by-turn.
          </p>
        </div>
      </div>

      <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-8">
        {errorMsg && (
          <div className="mb-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
            {errorMsg}
          </div>
        )}

        <motion.div
          className="grid grid-cols-1 lg:grid-cols-3 gap-6"
          variants={containerVariants}
          initial="hidden"
          animate="show"
        >
          <motion.div variants={cardVariants} className="lg:col-span-2">
            <div className="udemy-card p-6 space-y-5">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Track
                  </label>
                  <select
                    value={track}
                    onChange={(e) => setTrack(e.target.value as LearningTrack)}
                    className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  >
                    <option value="backend">Backend</option>
                    <option value="frontend">Frontend</option>
                    <option value="system_design">System Design</option>
                    <option value="ai_stack">AI Stack</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Level
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
                    Interview Type
                  </label>
                  <select
                    value={interviewType}
                    onChange={(e) =>
                      setInterviewType(e.target.value as InterviewType)
                    }
                    className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  >
                    <option value="mixed">Mixed</option>
                    <option value="technical">Technical</option>
                    <option value="coding">Coding</option>
                    <option value="behavioral">Behavioral</option>
                    <option value="system_design">System Design</option>
                    <option value="ai_fundamentals">AI Fundamentals</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Turn Count
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    step={1}
                    value={turnCount}
                    onChange={(e) => {
                      const next = Number(e.target.value);
                      if (!Number.isFinite(next)) return;
                      setTurnCount(
                        Math.max(1, Math.min(100, Math.trunc(next))),
                      );
                    }}
                    className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Interviewer Style
                  </label>
                  <select
                    value={interviewerStyle}
                    onChange={(e) =>
                      setInterviewerStyle(e.target.value as InterviewerStyle)
                    }
                    className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  >
                    <option value="supportive">Supportive</option>
                    <option value="neutral">Neutral</option>
                    <option value="challenging">Challenging</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Feedback Depth
                  </label>
                  <select
                    value={feedbackMode}
                    onChange={(e) =>
                      setFeedbackMode(e.target.value as FeedbackMode)
                    }
                    className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  >
                    <option value="concise">Concise</option>
                    <option value="deep">Deep</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Target Role (optional)
                </label>
                <input
                  type="text"
                  value={targetRole}
                  onChange={(e) => setTargetRole(e.target.value)}
                  placeholder="e.g. Senior Backend Engineer"
                  className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Focus Areas (comma-separated, optional)
                </label>
                <input
                  type="text"
                  value={focusAreasRaw}
                  onChange={(e) => setFocusAreasRaw(e.target.value)}
                  placeholder="api design, caching, tradeoffs"
                  className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Job Description Context (optional)
                </label>
                <textarea
                  value={jobDescriptionText}
                  onChange={(e) => setJobDescriptionText(e.target.value)}
                  rows={5}
                  className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  placeholder="Paste key responsibilities or required skills..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Resume Summary Context (optional)
                </label>
                <textarea
                  value={resumeSummaryText}
                  onChange={(e) => setResumeSummaryText(e.target.value)}
                  rows={4}
                  className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                  placeholder="Paste a concise summary of your experience..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Company (optional)
                </label>
                <div className="relative">
                  <Building2 className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-udemy-text-muted" />
                  <select
                    value={company}
                    onChange={(e) => setCompany(e.target.value)}
                    className="w-full border border-udemy-border rounded pl-9 pr-3 py-2.5 text-sm"
                  >
                    <option value="">Generic interview</option>
                    {companies.map((pack) => (
                      <option key={pack.company} value={pack.company}>
                        {pack.company}
                        {pack.track ? ` · ${pack.track}` : ""}
                      </option>
                    ))}
                  </select>
                </div>
                <p className="text-xs text-udemy-text-muted mt-1">
                  Company packs tune the interviewer style, question
                  tendencies, and difficulty.
                </p>
              </div>

              <div className="rounded-lg border border-udemy-border p-4 space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-udemy-purple" />
                    <span className="text-sm font-medium">
                      Resume Profile
                    </span>
                  </div>
                  {resumeProfile && (
                    <span className="inline-flex items-center gap-1 text-xs text-green-700 bg-green-100 rounded-full px-2 py-0.5">
                      <CheckCircle2 className="w-3 h-3" />
                      Parsed
                    </span>
                  )}
                </div>

                {resumeProfile ? (
                  <div className="space-y-2">
                    <div className="text-xs text-udemy-text-muted">
                      {resumeProfile.skills.length > 0 && (
                        <p>
                          <span className="font-medium">Skills:</span>{" "}
                          {resumeProfile.skills.join(", ")}
                        </p>
                      )}
                      {resumeProfile.roles.length > 0 && (
                        <p>
                          <span className="font-medium">Roles:</span>{" "}
                          {resumeProfile.roles.join(", ")}
                        </p>
                      )}
                      {typeof resumeProfile.experience_years ===
                        "number" && (
                        <p>
                          <span className="font-medium">Experience:</span>{" "}
                          {resumeProfile.experience_years} years
                        </p>
                      )}
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="inline-flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          checked={attachResume}
                          onChange={(e) => setAttachResume(e.target.checked)}
                          className="rounded border-udemy-border"
                        />
                        Auto-attach profile to new sessions
                      </label>
                      <button
                        onClick={handleResumeDelete}
                        disabled={deletingResume}
                        className="inline-flex items-center gap-1 text-xs text-red-600 hover:underline disabled:opacity-50"
                      >
                        <Trash2 className="w-3 h-3" />
                        Delete profile
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf,.docx,.txt"
                      onChange={(e) =>
                        setResumeFile(e.target.files?.[0] ?? null)
                      }
                      className="block w-full text-sm text-udemy-text-muted
                        file:mr-3 file:rounded file:border file:border-udemy-border
                        file:px-3 file:py-1.5 file:text-sm file:bg-white
                        hover:file:bg-gray-50"
                    />
                    <div className="flex items-center gap-3">
                      <button
                        onClick={handleResumeUpload}
                        disabled={!resumeFile || uploadingResume}
                        className="inline-flex items-center gap-1.5 text-sm btn-secondary disabled:opacity-50"
                      >
                        {uploadingResume ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Upload className="w-3.5 h-3.5" />
                        )}
                        {uploadingResume ? "Parsing..." : "Upload & parse"}
                      </button>
                      <span className="text-xs text-udemy-text-muted">
                        PDF, DOCX, or TXT · the raw file is never stored
                      </span>
                    </div>
                    {resumeError && (
                      <p className="text-xs text-red-600 flex items-center gap-1">
                        <AlertCircle className="w-3 h-3" />
                        {resumeError}
                      </p>
                    )}
                  </div>
                )}
              </div>

              <div className="rounded-lg border border-udemy-border p-4 space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm font-medium">
                    Job Description Gap Analysis
                  </span>
                  <button
                    onClick={handleJdAnalysis}
                    disabled={!jobDescriptionText.trim() || jdAnalyzing}
                    className="inline-flex items-center gap-1.5 text-sm btn-secondary disabled:opacity-50"
                  >
                    {jdAnalyzing ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Sparkles className="w-3.5 h-3.5" />
                    )}
                    {jdAnalyzing ? "Analyzing..." : "Analyze gaps"}
                  </button>
                </div>

                {jdAnalysis && (
                  <div className="space-y-3 text-sm">
                    {jdAnalysis.gaps.length > 0 && (
                      <div>
                        <p className="font-medium text-amber-700 mb-1">
                          Gaps ({jdAnalysis.gaps.length})
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {jdAnalysis.gaps.map((gap) => (
                            <span
                              key={gap}
                              className="text-xs bg-amber-100 text-amber-800 rounded-full px-2 py-0.5"
                            >
                              {gap}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {jdAnalysis.matched.length > 0 && (
                      <div>
                        <p className="font-medium text-green-700 mb-1">
                          Matched ({jdAnalysis.matched.length})
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {jdAnalysis.matched.map((skill) => (
                            <span
                              key={skill}
                              className="text-xs bg-green-100 text-green-800 rounded-full px-2 py-0.5"
                            >
                              {skill}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {jdAnalysis.weak_spots.length > 0 && (
                      <div>
                        <p className="font-medium text-udemy-text-muted mb-1">
                          Weak spots
                        </p>
                        <ul className="list-disc list-inside text-xs text-udemy-text-muted space-y-0.5">
                          {jdAnalysis.weak_spots.map((spot) => (
                            <li key={spot}>{spot}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {jdAnalysis.recommended_focus_areas.length > 0 && (
                      <button
                        onClick={applyRecommendedFocusAreas}
                        className="inline-flex items-center gap-1 text-xs text-udemy-purple hover:underline"
                      >
                        Use {jdAnalysis.recommended_focus_areas.length}{" "}
                        recommended focus areas
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                )}
                {!jdAnalysis && (
                  <p className="text-xs text-udemy-text-muted">
                    Compare the job description above against your resume
                    profile to surface skill gaps and focus areas.
                  </p>
                )}
              </div>

              <button
                onClick={handleStart}
                disabled={starting}
                className="btn-primary w-full sm:w-auto flex items-center gap-2 disabled:opacity-50"
              >
                {starting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4" />
                )}
                {starting ? "Starting..." : "Start Mock Interview"}
              </button>
              {starting && (
                <p className="text-xs text-udemy-text-muted">{startProgress}</p>
              )}
            </div>
          </motion.div>

          <motion.div variants={cardVariants}>
            <div className="udemy-card p-6">
              <h2 className="text-lg font-bold mb-3">Recent Sessions</h2>
              {loadingRecent ? (
                <div className="space-y-2">
                  {Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="skeleton h-16 rounded" />
                  ))}
                </div>
              ) : recent.length === 0 ? (
                <p className="text-sm text-udemy-text-muted">
                  No sessions yet.
                </p>
              ) : (
                <div className="space-y-2">
                  {recent.map((s) => (
                    <Link
                      key={s.session_id}
                      to={`/interview/${s.session_id}`}
                      className="block rounded border border-udemy-border px-3 py-2 hover:bg-gray-50"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-medium line-clamp-1">
                          {TRACK_LABELS[s.track]} · {s.level}
                        </span>
                        <span
                          className={`text-xs px-2 py-0.5 rounded-full ${
                            s.status === "completed"
                              ? "bg-green-100 text-green-700"
                              : "bg-blue-100 text-blue-700"
                          }`}
                        >
                          {s.status}
                        </span>
                      </div>
                      <div className="text-xs text-udemy-text-muted mt-1 flex items-center gap-2">
                        <Clock className="w-3.5 h-3.5" />
                        {s.turns_completed}/{s.turn_count} turns
                      </div>
                    </Link>
                  ))}
                </div>
              )}

              <div className="mt-5 text-xs text-udemy-text-muted">
                Powered by {settings.provider}
              </div>

              <Link
                to="/quiz"
                className="mt-4 inline-flex items-center gap-1 text-sm text-udemy-purple"
              >
                Practice quick quiz
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                to="/interview/trends"
                className="mt-3 inline-flex items-center gap-1 text-sm text-udemy-purple"
              >
                View interview trends
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </motion.div>
  );
}
