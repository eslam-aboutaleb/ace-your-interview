import { useCallback, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Compass,
  Loader2,
  Sparkles,
  Target,
  XCircle,
} from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import { useAuthStore } from "@/store/authStore";
import {
  answerPlacement,
  finishPlacement,
  startPlacement,
} from "@/services/api";
import type {
  InterviewLevel,
  LearningTrack,
  PlacementFinishResponse,
  PlacementQuestion,
  PlacementStartResponse,
} from "@/types";

const TRACK_LABELS: Record<LearningTrack, string> = {
  backend: "Backend",
  frontend: "Frontend",
  system_design: "System Design",
  ai_stack: "AI Stack",
};

const DAILY_GOALS = [1, 2, 3, 5, 10];

type Stage = "welcome" | "intake" | "quiz" | "summary";

interface AnswerFeedback {
  is_correct: boolean;
  correct_answer: string;
  explanation: string;
  level: InterviewLevel;
}

const OPTION_LABELS = ["A", "B", "C", "D"];

function levelLabel(level: InterviewLevel): string {
  return level === "senior" ? "Senior" : level === "mid" ? "Mid" : "Junior";
}

export default function OnboardingPage() {
  const navigate = useNavigate();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const authLoading = useAuthStore((s) => s.isLoading);

  const [stage, setStage] = useState<Stage>("welcome");
  const [targetRole, setTargetRole] = useState("");
  const [track, setTrack] = useState<LearningTrack>("backend");
  const [dailyGoal, setDailyGoal] = useState(3);
  const [session, setSession] = useState<PlacementStartResponse | null>(null);
  const [question, setQuestion] = useState<PlacementQuestion | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<AnswerFeedback | null>(null);
  const [summary, setSummary] = useState<PlacementFinishResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const beginQuiz = async () => {
    setBusy(true);
    setErrorMsg("");
    try {
      const started = await startPlacement({
        track,
        target_role: targetRole.trim() || undefined,
        daily_goal: dailyGoal,
      });
      setSession(started);
      setQuestion(started.question);
      setFeedback(null);
      setSelected(null);
      setStage("quiz");
    } catch (err) {
      console.error(err);
      setErrorMsg("Could not start the placement quiz. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  const submitAnswer = async (option: string) => {
    if (!session || !question || feedback || busy) return;
    const optionLabel = OPTION_LABELS[question.options.indexOf(option)] ?? option;
    setBusy(true);
    setErrorMsg("");
    try {
      const result = await answerPlacement({
        session_id: session.session_id,
        question_id: question.question_id,
        selected_answer: optionLabel,
      });
      setSelected(option);
      setFeedback({
        is_correct: result.is_correct,
        correct_answer: result.correct_answer,
        explanation: result.explanation,
        level: result.level,
      });
      if (result.complete) {
        const finished = await finishPlacement({
          session_id: result.session_id,
        });
        setSummary(finished);
        setStage("summary");
      } else {
        setQuestion(result.question);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg("Could not grade that answer. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  const goToNextQuestion = () => {
    setFeedback(null);
    setSelected(null);
  };

  const openStudyPlan = useCallback(() => {
    navigate("/study-plan");
  }, [navigate]);

  if (!authLoading && !isAuthenticated) {
    return (
      <div className="min-h-screen bg-udemy-bg flex items-center justify-center px-4">
        <div className="udemy-card p-8 max-w-md w-full text-center">
          <h1 className="text-xl font-bold mb-2">Sign in to continue</h1>
          <p className="text-sm text-udemy-text-muted mb-6">
            Onboarding saves your placement result to your learner profile,
            so it needs an account.
          </p>
          <Link to="/login" className="btn-primary inline-flex">
            Go to sign in
          </Link>
        </div>
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
      className="min-h-screen bg-udemy-bg"
    >
      <div className="max-w-[860px] mx-auto px-4 sm:px-6 py-10">
        {stage === "welcome" && (
          <section className="udemy-card p-6 sm:p-8">
            <span className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-udemy-purple">
              <Compass className="w-4 h-4" />
              Placement onboarding
            </span>
            <h1 className="text-2xl sm:text-3xl font-bold mt-3 mb-3">
              Let&apos;s find your starting level
            </h1>
            <p className="text-sm sm:text-base text-udemy-text-muted mb-6">
              A short adaptive check places you on the junior, mid, or senior
              rung for your track. Correct answers move you up a level, wrong
              answers move you down, and the result seeds your study plan,
              recommendations, and readiness score — no guessing from a
              default profile.
            </p>
            <ul className="space-y-2 text-sm text-udemy-text-muted mb-8">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-udemy-purple mt-0.5" />
                8–12 questions at most, one at a time with instant feedback.
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-udemy-purple mt-0.5" />
                Every answer is graded deterministically against the
                curriculum answer key.
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-udemy-purple mt-0.5" />
                You can retake it any time from your study plan.
              </li>
            </ul>
            <button
              onClick={() => setStage("intake")}
              className="btn-primary inline-flex items-center gap-2"
            >
              Get started
              <ArrowRight className="w-4 h-4" />
            </button>
          </section>
        )}

        {stage === "intake" && (
          <section className="udemy-card p-6 sm:p-8">
            <h1 className="text-2xl font-bold mb-2">Your goal</h1>
            <p className="text-sm text-udemy-text-muted mb-6">
              Three answers and the placement quiz starts.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Target role
                </label>
                <input
                  value={targetRole}
                  onChange={(e) => setTargetRole(e.target.value)}
                  placeholder="e.g. Senior Backend Engineer"
                  className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Primary track
                </label>
                <select
                  value={track}
                  onChange={(e) => setTrack(e.target.value as LearningTrack)}
                  className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
                >
                  {(
                    Object.keys(TRACK_LABELS) as Array<LearningTrack>
                  ).map((key) => (
                    <option key={key} value={key}>
                      {TRACK_LABELS[key]}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Daily goal (items per day)
                </label>
                <select
                  value={dailyGoal}
                  onChange={(e) => setDailyGoal(Number(e.target.value))}
                  className="w-full border border-udemy-border rounded-lg px-3 py-2.5 text-sm"
                >
                  {DAILY_GOALS.map((goal) => (
                    <option key={goal} value={goal}>
                      {goal} {goal === 1 ? "item" : "items"}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {errorMsg && (
              <p className="mt-4 text-xs text-amber-800 bg-amber-50 border border-amber-300 rounded px-2 py-1">
                {errorMsg}
              </p>
            )}

            <div className="mt-6 flex flex-wrap gap-2">
              <button
                onClick={beginQuiz}
                disabled={busy}
                className="btn-primary inline-flex items-center gap-2 disabled:opacity-50"
              >
                {busy ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4" />
                )}
                Start placement quiz
              </button>
              <button
                onClick={() => setStage("welcome")}
                className="btn-secondary inline-flex"
              >
                Back
              </button>
            </div>
          </section>
        )}

        {stage === "quiz" && question && session && (
          <section className="udemy-card p-6 sm:p-8">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <span className="inline-flex items-center gap-2 text-sm font-semibold text-udemy-purple">
                <Target className="w-4 h-4" />
                Level: {levelLabel(question.level)}
              </span>
              <span className="text-xs text-udemy-text-muted">
                Question {Math.min(session.step_index + 1, session.max_steps)} of{" "}
                {session.max_steps}
              </span>
            </div>

            <div
              className="flex flex-wrap gap-1.5 mb-6"
              data-testid="placement-progress"
              aria-label={`Placement progress: ${session.step_index} of ${session.max_steps}`}
            >
              {Array.from({ length: session.max_steps }, (_, index) => (
                <span
                  key={index}
                  className={`h-1.5 rounded-full ${
                    index < session.step_index
                      ? "w-6 bg-udemy-purple"
                      : "w-3 bg-udemy-border"
                  }`}
                />
              ))}
            </div>

            <p className="text-xs text-udemy-text-muted mb-2">
              {question.topic_title}
              {question.concept ? ` · ${question.concept}` : ""}
            </p>
            <h1 className="text-lg sm:text-xl font-bold mb-6">
              {question.question}
            </h1>

            <div className="space-y-3">
              {question.options.map((option, index) => {
                const chosen = selected === option;
                const reveal = feedback !== null;
                const isCorrect =
                  reveal &&
                  OPTION_LABELS[index] === feedback.correct_answer;
                return (
                  <button
                    key={option}
                    onClick={() => submitAnswer(option)}
                    disabled={reveal || busy}
                    className={`w-full text-left border rounded-lg px-4 py-3 text-sm transition-colors disabled:cursor-default ${
                      reveal && isCorrect
                        ? "border-green-500 bg-green-50"
                        : reveal && chosen
                          ? "border-red-500 bg-red-50"
                          : "border-udemy-border hover:bg-gray-50"
                    }`}
                  >
                    <span className="font-semibold mr-2">
                      {OPTION_LABELS[index]}.
                    </span>
                    {option}
                  </button>
                );
              })}
            </div>

            {feedback && (
              <div
                className={`mt-5 rounded-lg border p-4 ${
                  feedback.is_correct
                    ? "border-green-300 bg-green-50"
                    : "border-amber-300 bg-amber-50"
                }`}
              >
                <p className="text-sm font-semibold flex items-center gap-2">
                  {feedback.is_correct ? (
                    <>
                      <CheckCircle2 className="w-4 h-4 text-green-600" />
                      Correct
                    </>
                  ) : (
                    <>
                      <XCircle className="w-4 h-4 text-amber-600" />
                      Incorrect — the correct answer is{" "}
                      {feedback.correct_answer}.
                    </>
                  )}
                </p>
                {feedback.explanation && (
                  <p className="text-sm text-udemy-text-muted mt-2">
                    {feedback.explanation}
                  </p>
                )}
                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <button
                    onClick={() =>
                      summary ? setStage("summary") : goToNextQuestion()
                    }
                    className="btn-primary inline-flex items-center gap-2"
                  >
                    {summary ? "See my results" : "Next question"}
                    <ArrowRight className="w-4 h-4" />
                  </button>
                  <span className="text-xs text-udemy-text-muted">
                    {summary
                      ? "Placement complete — your profile is seeded."
                      : "Correct answers move you up a level, wrong answers move you down."}
                  </span>
                </div>
              </div>
            )}

            {errorMsg && (
              <p className="mt-4 text-xs text-amber-800 bg-amber-50 border border-amber-300 rounded px-2 py-1 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                {errorMsg}
              </p>
            )}
          </section>
        )}

        {stage === "summary" && summary && (
          <section className="udemy-card p-6 sm:p-8" data-testid="placement-summary">
            <span className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-udemy-purple">
              <Sparkles className="w-4 h-4" />
              Placement complete
            </span>
            <h1 className="text-2xl font-bold mt-3 mb-2">
              You start at {levelLabel(summary.derived_level)} level in{" "}
              {TRACK_LABELS[summary.track]}
            </h1>
            <p className="text-sm text-udemy-text-muted mb-6">
              Derived from {summary.steps_taken} adaptive questions. Your
              learner profile, study plan, and recommendations now start from
              this evidence instead of a default guess.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
              <div className="rounded-lg border border-udemy-border p-4">
                <p className="text-xs text-udemy-text-muted mb-1">
                  Starting level
                </p>
                <p className="text-lg font-bold">
                  {levelLabel(summary.derived_level)}
                </p>
              </div>
              <div className="rounded-lg border border-udemy-border p-4">
                <p className="text-xs text-udemy-text-muted mb-1">
                  Per-track levels
                </p>
                <p className="text-sm font-medium">
                  {Object.entries(summary.per_track_levels)
                    .map(
                      ([trackKey, level]) =>
                        `${TRACK_LABELS[trackKey as LearningTrack] || trackKey}: ${levelLabel(
                          level as InterviewLevel,
                        )}`,
                    )
                    .join(" · ") || "Not measured yet"}
                </p>
              </div>
            </div>

            <div className="mb-6">
              <h2 className="text-sm font-bold mb-2">Level walk</h2>
              <div className="flex flex-wrap gap-1.5">
                {summary.level_path.map((level, index) => (
                  <span
                    key={`${level}-${index}`}
                    className="inline-flex items-center rounded-full border border-udemy-border bg-white px-2.5 py-1 text-xs font-medium"
                  >
                    {levelLabel(level as InterviewLevel)}
                  </span>
                ))}
              </div>
            </div>

            {summary.weak_objectives.length > 0 && (
              <div className="mb-6">
                <h2 className="text-sm font-bold mb-2">Focus these first</h2>
                <ul className="space-y-2">
                  {summary.weak_objectives.map((item) => (
                    <li
                      key={`${item.topic_id}-${item.objective}`}
                      className="rounded-lg border border-udemy-border bg-white p-3"
                    >
                      <p className="text-sm font-semibold">
                        {item.topic_title || item.topic_id}
                      </p>
                      <p className="text-xs text-udemy-text-muted mt-1">
                        {item.objective} · missed at{" "}
                        {levelLabel(item.level)} level
                      </p>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex flex-wrap gap-2">
              <button
                onClick={openStudyPlan}
                className="btn-primary inline-flex items-center gap-2"
              >
                Open my study plan
                <ArrowRight className="w-4 h-4" />
              </button>
              <Link to="/" className="btn-secondary inline-flex">
                Back to dashboard
              </Link>
            </div>
          </section>
        )}
      </div>
    </motion.div>
  );
}
