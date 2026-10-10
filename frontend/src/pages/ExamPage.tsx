import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  FileText,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  XCircle,
} from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import {
  createExam,
  fetchExam,
  fetchExams,
  remarkExam,
  startExam,
  submitExam,
} from "@/services/api";
import { useApi } from "@/hooks/useApi";
import type {
  ExamCreateRequest,
  ExamDetailResponse,
  ExamListResponse,
  ExamQuestion,
  ExamStartResponse,
  ExamSubmitResponse,
} from "@/types";

type Phase =
  | { name: "list" }
  | { name: "create" }
  | {
      name: "taking";
      exam: ExamStartResponse;
      startedAt: number;
      durationMs: number;
    }
  | { name: "results"; examId: string };

export default function ExamPage() {
  const [phase, setPhase] = useState<Phase>({ name: "list" });
  const [title, setTitle] = useState("");
  const [questionCount, setQuestionCount] = useState(10);
  const [durationMinutes, setDurationMinutes] = useState(30);
  const [questionTypes, setQuestionTypes] = useState<string[]>(["mcq"]);
  const [weakAreas, setWeakAreas] = useState(true);
  const [answers, setAnswers] = useState<
    Record<string, { text: string; timeMs: number }>
  >({});
  const [timeLeftMs, setTimeLeftMs] = useState(0);
  const submitRef = useRef<(() => void) | null>(null);

  const list = useApi<ExamListResponse>(async () => fetchExams(), {
    errorMessage: "Could not load exam history.",
    fixedErrorMessage: true,
  });

  const results = useApi<ExamSubmitResponse | null>(
    async () => null,
    { errorMessage: "", fixedErrorMessage: true },
  );

  const detail = useApi<ExamDetailResponse | null>(async () => null, {
    errorMessage: "",
    fixedErrorMessage: true,
  });

  const remark = useApi<{
    original_score: number;
    new_score: number;
    diff: number;
  } | null>(async () => null, {
    errorMessage: "",
    fixedErrorMessage: true,
  });

  // Countdown timer — the server-started_at is authoritative.
  useEffect(() => {
    if (phase.name !== "taking") return;
    const interval = window.setInterval(() => {
      const remaining =
        phase.durationMs - (Date.now() - phase.startedAt);
      setTimeLeftMs(Math.max(0, remaining));
      if (remaining <= 0) {
        window.clearInterval(interval);
        submitRef.current?.();
      }
    }, 250);
    return () => window.clearInterval(interval);
  }, [phase]);

  const formatTime = (ms: number) => {
    const totalSeconds = Math.ceil(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${String(seconds).padStart(2, "0")}`;
  };

  const createPayload: ExamCreateRequest = useMemo(
    () => ({
      title: title.trim() || "Practice Exam",
      source: { topic_ids: [], deck_ids: [], weak_areas: weakAreas },
      question_count: questionCount,
      duration_minutes: durationMinutes,
      question_types: questionTypes,
    }),
    [title, questionCount, durationMinutes, questionTypes, weakAreas],
  );

  const handleCreate = async () => {
    try {
      const created = await createExam(createPayload);
      const started = await startExam(created.exam_id);
      setAnswers({});
      setTimeLeftMs(started.duration_minutes * 60_000);
      setPhase({
        name: "taking",
        exam: started,
        startedAt: Date.now(),
        durationMs: started.duration_minutes * 60_000,
      });
    } catch (error) {
      console.error(error);
    }
  };

  const handleSubmit = async () => {
    if (phase.name !== "taking") return;
    const payload = {
      answers: phase.exam.questions.map((question) => ({
        question_ref: question.question_ref,
        answer_text: answers[question.question_ref]?.text ?? "",
        time_ms: answers[question.question_ref]?.timeMs ?? 0,
      })),
    };
    try {
      const submitted = await submitExam(phase.exam.exam_id, payload);
      results.setData(() => submitted);
      setPhase({ name: "results", examId: phase.exam.exam_id });
    } catch (error) {
      console.error(error);
    }
  };
  submitRef.current = handleSubmit;

  const handleRemark = async (examId: string) => {
    try {
      const result = await remarkExam(examId);
      remark.setData(() => result);
      const fresh = await fetchExam(examId);
      detail.setData(() => fresh);
    } catch (error) {
      console.error(error);
    }
  };

  const handleViewResults = async (examId: string) => {
    try {
      const fresh = await fetchExam(examId);
      detail.setData(() => fresh);
      setPhase({ name: "results", examId });
    } catch (error) {
      console.error(error);
    }
  };

  const setAnswer = (
    questionRef: string,
    text: string,
    timeMs = 0,
  ) => {
    setAnswers((prev) => ({
      ...prev,
      [questionRef]: { text, timeMs: prev[questionRef]?.timeMs ?? timeMs },
    }));
  };

  const toggleType = (type: string) => {
    setQuestionTypes((prev) =>
      prev.includes(type)
        ? prev.filter((t) => t !== type)
        : [...prev, type],
    );
  };

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
      className="max-w-[1200px] mx-auto px-4 sm:px-6 py-8"
    >
      {phase.name === "list" && (
        <ExamList
          list={list}
          loading={list.loading}
          error={list.error}
          onRetry={() => void list.retry()}
          onCreate={() => setPhase({ name: "create" })}
          onViewResults={handleViewResults}
        />
      )}

      {phase.name === "create" && (
        <div className="udemy-card p-6">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <FileText className="w-6 h-6 text-udemy-purple" />
              New Exam
            </h1>
            <button
              onClick={() => setPhase({ name: "list" })}
              className="btn-secondary"
            >
              Back
            </button>
          </div>
          <p className="text-sm text-udemy-text-muted mt-1">
            Questions are drawn from weak areas, due cards, and your
            decks. The server stamps the start time — the timer is
            authoritative.
          </p>

          <div className="mt-5 space-y-4">
            <label className="block">
              <span className="text-sm font-medium">Title</span>
              <input
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                placeholder="Practice Exam"
                className="mt-1 w-full rounded-lg border border-udemy-border px-3 py-2"
              />
            </label>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <label className="block">
                <span className="text-sm font-medium">
                  Questions (max 50)
                </span>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={questionCount}
                  onChange={(event) =>
                    setQuestionCount(
                      Number(event.target.value) || 1,
                    )
                  }
                  className="mt-1 w-full rounded-lg border border-udemy-border px-3 py-2"
                />
              </label>
              <label className="block">
                <span className="text-sm font-medium">
                  Duration (minutes)
                </span>
                <input
                  type="number"
                  min={1}
                  max={240}
                  value={durationMinutes}
                  onChange={(event) =>
                    setDurationMinutes(
                      Number(event.target.value) || 1,
                    )
                  }
                  className="mt-1 w-full rounded-lg border border-udemy-border px-3 py-2"
                />
              </label>
            </div>

            <div>
              <span className="text-sm font-medium">Question types</span>
              <div className="mt-1 flex gap-2">
                {["mcq", "written"].map((type) => (
                  <button
                    key={type}
                    onClick={() => toggleType(type)}
                    className={`px-3 py-1.5 rounded-lg border text-sm ${
                      questionTypes.includes(type)
                        ? "border-udemy-purple bg-udemy-purple/10 text-udemy-purple"
                        : "border-udemy-border text-udemy-text-muted"
                    }`}
                  >
                    {type === "mcq" ? "Multiple choice" : "Written"}
                  </button>
                ))}
              </div>
            </div>

            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={weakAreas}
                onChange={(event) => setWeakAreas(event.target.checked)}
              />
              <span className="text-sm">
                Include weak-area questions
              </span>
            </label>

            <button
              onClick={() => void handleCreate()}
              className="btn-primary inline-flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              Start exam
            </button>
          </div>
        </div>
      )}

      {phase.name === "taking" && (
        <ExamTaking
          exam={phase.exam}
          timeLeftMs={timeLeftMs}
          formatTime={formatTime}
          answers={answers}
          setAnswer={setAnswer}
          onSubmit={() => void handleSubmit()}
        />
      )}

      {phase.name === "results" && (
        <ExamResults
          examId={phase.examId}
          detail={detail.data}
          detailLoading={detail.loading}
          submitResult={results.data}
          remarkResult={remark.data}
          remarkLoading={remark.loading}
          onRemark={() => void handleRemark(phase.examId)}
          onBack={() => {
            setPhase({ name: "list" });
            void list.retry();
          }}
        />
      )}
    </motion.div>
  );
}

function ExamList({
  list,
  loading,
  error,
  onRetry,
  onCreate,
  onViewResults,
}: {
  list: { data?: ExamListResponse | null };
  loading: boolean;
  error?: string;
  onRetry: () => void;
  onCreate: () => void;
  onViewResults: (examId: string) => void;
}) {
  return (
    <div className="udemy-card p-6">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <FileText className="w-6 h-6 text-udemy-purple" />
            Exams
          </h1>
          <p className="text-sm text-udemy-text-muted mt-1">
            Timed practice exams with auto-grading.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={onRetry}
            disabled={loading}
            className="btn-secondary inline-flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
            Refresh
          </button>
          <button
            onClick={onCreate}
            className="btn-primary inline-flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New exam
          </button>
        </div>
      </div>

      {error && (
        <div
          className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2"
          role="alert"
        >
          <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="mt-5 space-y-3">
        {(list.data?.exams ?? []).length === 0 && !loading && (
          <div className="rounded-lg border border-udemy-border p-6 text-center text-sm text-udemy-text-muted">
            No exams yet. Create one to get started.
          </div>
        )}
        {(list.data?.exams ?? []).map((exam) => (
          <div
            key={exam.exam_id}
            className="rounded-lg border border-udemy-border p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3"
          >
            <div>
              <div className="font-medium">{exam.title}</div>
              <div className="text-sm text-udemy-text-muted mt-0.5">
                {exam.question_count} questions ·{" "}
                {exam.duration_minutes} min · {exam.status}
                {exam.score !== null &&
                  ` · ${Math.round(exam.score * 100)}%`}
                {exam.passed !== null &&
                  (exam.passed ? " · passed" : " · not passed")}
                {exam.time_exceeded && " · time exceeded"}
              </div>
            </div>
            {exam.status === "completed" && (
              <button
                onClick={() => onViewResults(exam.exam_id)}
                className="btn-secondary"
              >
                View report
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ExamTaking({
  exam,
  timeLeftMs,
  formatTime,
  answers,
  setAnswer,
  onSubmit,
}: {
  exam: ExamStartResponse;
  timeLeftMs: number;
  formatTime: (ms: number) => string;
  answers: Record<string, { text: string; timeMs: number }>;
  setAnswer: (ref: string, text: string, timeMs?: number) => void;
  onSubmit: () => void;
}) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const question: ExamQuestion = exam.questions[currentIndex];
  const answeredCount = exam.questions.filter((q) =>
    (answers[q.question_ref]?.text ?? "").trim() !== "",
  ).length;

  if (!question) return null;

  return (
    <div className="udemy-card p-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold">Exam in progress</h1>
          <p className="text-sm text-udemy-text-muted mt-0.5">
            Question {currentIndex + 1} of {exam.questions.length} ·{" "}
            {answeredCount} answered
          </p>
        </div>
        <div
          className={`flex items-center gap-2 rounded-lg px-3 py-1.5 font-mono text-lg ${
            timeLeftMs < 60_000
              ? "bg-red-50 text-red-700"
              : "bg-udemy-purple/10 text-udemy-purple"
          }`}
        >
          <Clock3 className="w-4 h-4" />
          {formatTime(timeLeftMs)}
        </div>
      </div>

      <div className="mt-6">
        <h2 className="text-lg font-medium">
          {question.question_text}
        </h2>
        {question.type === "mcq" ? (
          <div className="mt-4 space-y-2">
            {question.options.map((option, index) => {
              const letter = String.fromCharCode(65 + index);
              const selected =
                answers[question.question_ref]?.text === letter;
              return (
                <button
                  key={index}
                  onClick={() => setAnswer(question.question_ref, letter)}
                  className={`w-full text-left rounded-lg border p-3 flex items-center gap-3 ${
                    selected
                      ? "border-udemy-purple bg-udemy-purple/10"
                      : "border-udemy-border"
                  }`}
                >
                  <span className="w-6 h-6 rounded-full border border-udemy-border flex items-center justify-center text-sm font-medium">
                    {letter}
                  </span>
                  <span>{option}</span>
                </button>
              );
            })}
          </div>
        ) : (
          <textarea
            value={answers[question.question_ref]?.text ?? ""}
            onChange={(event) =>
              setAnswer(question.question_ref, event.target.value)
            }
            rows={6}
            placeholder="Write your answer…"
            className="mt-4 w-full rounded-lg border border-udemy-border px-3 py-2"
          />
        )}
      </div>

      <div className="mt-6 flex items-center justify-between">
        <button
          onClick={() => setCurrentIndex((i) => Math.max(0, i - 1))}
          disabled={currentIndex === 0}
          className="btn-secondary disabled:opacity-50"
        >
          Previous
        </button>
        {currentIndex < exam.questions.length - 1 ? (
          <button
            onClick={() => setCurrentIndex((i) => i + 1)}
            className="btn-primary"
          >
            Next
          </button>
        ) : (
          <button onClick={onSubmit} className="btn-primary">
            Submit exam
          </button>
        )}
      </div>
    </div>
  );
}

function ExamResults({
  examId,
  detail,
  detailLoading,
  submitResult,
  remarkResult,
  remarkLoading,
  onRemark,
  onBack,
}: {
  examId: string;
  detail?: ExamDetailResponse | null;
  detailLoading: boolean;
  submitResult?: ExamSubmitResponse | null;
  remarkResult?: {
    original_score: number;
    new_score: number;
    diff: number;
  } | null;
  remarkLoading: boolean;
  onRemark: () => void;
  onBack: () => void;
}) {
  const score = detail?.score ?? submitResult?.score ?? 0;
  const passed = detail?.passed ?? submitResult?.passed ?? false;
  const results = detail?.results ?? submitResult?.per_question ?? [];

  return (
    <div className="udemy-card p-6">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Exam report</h1>
          <p className="text-sm text-udemy-text-muted mt-1">
            {detail?.title ?? examId}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={onRemark}
            disabled={remarkLoading}
            className="btn-secondary inline-flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {remarkLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RotateCcw className="w-4 h-4" />
            )}
            Re-grade (remark)
          </button>
          <button onClick={onBack} className="btn-secondary">
            Back to exams
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5">
        <Metric
          label="Score"
          value={`${Math.round(score * 100)}%`}
        />
        <Metric
          label="Result"
          value={passed ? "Passed" : "Not passed"}
        />
        {detail?.time_exceeded && (
          <Metric label="Timing" value="Exceeded" />
        )}
        {remarkResult && (
          <Metric
            label="Remark delta"
            value={`${remarkResult.diff >= 0 ? "+" : ""}${Math.round(
              remarkResult.diff * 100,
            )}%`}
          />
        )}
      </div>

      {remarkResult && (
        <div className="mt-4 rounded-lg border border-udemy-border p-3 text-sm text-udemy-text-muted">
          Re-grade: {Math.round(remarkResult.original_score * 100)}% →{" "}
          {Math.round(remarkResult.new_score * 100)}%
          {remarkResult.diff === 0 ? " (no change)" : ""}
        </div>
      )}

      <div className="mt-6 space-y-3">
        {results.map((result) => {
          const correct = result.is_correct;
          return (
            <div
              key={result.question_ref}
              className="rounded-lg border border-udemy-border p-4"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2">
                  {correct === null ? (
                    <Clock3 className="w-4 h-4 mt-0.5 text-udemy-text-muted" />
                  ) : correct ? (
                    <CheckCircle2 className="w-4 h-4 mt-0.5 text-green-600" />
                  ) : (
                    <XCircle className="w-4 h-4 mt-0.5 text-red-600" />
                  )}
                  <div>
                    <div className="text-sm font-medium">
                      {result.question_ref}
                      <span className="ml-2 text-xs text-udemy-text-muted">
                        {result.type} · {result.graded_by}
                      </span>
                    </div>
                    <div className="text-sm text-udemy-text-muted mt-1">
                      {result.feedback}
                    </div>
                  </div>
                </div>
                <div className="text-sm font-medium">
                  {Math.round(result.score * 100)}%
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-udemy-border p-3">
      <div className="text-xs text-udemy-text-muted">{label}</div>
      <div className="mt-1 text-xl font-bold">{value}</div>
    </div>
  );
}
