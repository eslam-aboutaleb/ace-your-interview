import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { Loader2, Trophy, ArrowRight } from "lucide-react";
import { pageVariants, pageTransition } from "@/utils/animations";
import { fetchInterviewReport } from "@/services/api";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import { normalizeEscapedSingleLineText } from "@/utils/textNormalization";
import type { InterviewReportResponse } from "@/types";

export default function InterviewReportPage() {
  const { sessionId = "" } = useParams();
  const [data, setData] = useState<InterviewReportResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    if (!sessionId) return;
    fetchInterviewReport(sessionId)
      .then(setData)
      .catch(() => setErrorMsg("Could not load interview report."))
      .finally(() => setLoading(false));
  }, [sessionId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-udemy-purple" />
      </div>
    );
  }

  if (errorMsg || !data) {
    return (
      <div className="max-w-[1000px] mx-auto px-4 sm:px-6 py-12">
        <p className="text-udemy-text-muted">{errorMsg || "No report available."}</p>
      </div>
    );
  }

  const report = data.report;

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
    >
      <div className="bg-udemy-dark text-white">
        <div className="max-w-[1000px] mx-auto px-4 sm:px-6 py-8">
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
            <Trophy className="w-7 h-7 text-udemy-purple-light" />
            Interview Report
          </h1>
          <p className="text-gray-300 mt-2">
            Readiness: <span className="font-semibold">{report.readiness_label}</span> · Score {report.overall_score}
          </p>
        </div>
      </div>

      <div className="max-w-[1000px] mx-auto px-4 sm:px-6 py-8 space-y-6">
        <div className="udemy-card p-6">
          <h2 className="font-bold mb-2">Summary</h2>
          <MarkdownRenderer content={report.summary} className="text-sm text-udemy-text-muted" />
        </div>

        <div className="udemy-card p-6">
          <h2 className="font-bold mb-3">Rubric Averages</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            {Object.entries(report.rubric_averages).map(([k, v]) => (
              <div key={k} className="bg-udemy-bg rounded px-3 py-2">
                <p className="text-[11px] text-udemy-text-muted uppercase">{k.replace(/_/g, " ")}</p>
                <p className="font-bold">{v}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="udemy-card p-6">
            <h2 className="font-bold mb-2">Weak Competencies</h2>
            <ul className="list-disc pl-5 text-sm text-udemy-text-muted space-y-1">
              {report.weak_competencies.map((x, i) => (
                <li key={`${x}-${i}`}>{normalizeEscapedSingleLineText(x)}</li>
              ))}
            </ul>
          </div>

          <div className="udemy-card p-6">
            <h2 className="font-bold mb-2">Top Strengths</h2>
            <ul className="list-disc pl-5 text-sm text-udemy-text-muted space-y-1">
              {report.strengths.map((x, i) => (
                <li key={`${x}-${i}`}>{normalizeEscapedSingleLineText(x)}</li>
              ))}
            </ul>
          </div>
        </div>

        <div className="udemy-card p-6">
          <h2 className="font-bold mb-2">Recommended Topics</h2>
          <div className="flex flex-wrap gap-2 mb-4">
            {report.recommended_topic_ids.map((topicId) => (
              <Link
                key={topicId}
                to={`/topics/${topicId}`}
                className="text-sm bg-udemy-purple/10 text-udemy-purple px-3 py-1 rounded-full"
              >
                {normalizeEscapedSingleLineText(topicId)}
              </Link>
            ))}
          </div>

          <h3 className="font-semibold mb-2">Next Steps</h3>
          <ol className="list-decimal pl-5 text-sm text-udemy-text-muted space-y-1">
            {report.next_steps.map((s, i) => (
              <li key={`${s}-${i}`}>{normalizeEscapedSingleLineText(s)}</li>
            ))}
          </ol>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <Link to="/interview" className="btn-secondary w-full sm:w-auto text-center">
              New Mock Interview
            </Link>
            <Link to="/quiz" className="btn-primary w-full sm:w-auto inline-flex items-center justify-center gap-2">
              Start Quiz
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
