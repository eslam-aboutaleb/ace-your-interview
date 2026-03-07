import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Loader2, Sparkles } from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import {
  createCustomTopicStream,
  isCustomTopicStreamError,
} from "@/services/api";
import { useSettingsStore } from "@/store/settingsStore";

type StreamedSection = {
  index: number;
  total_sections: number;
  heading: string;
  content: string;
};

function clampTargetSections(value: number): number {
  if (!Number.isFinite(value)) return 120;
  if (value < 100) return 100;
  if (value > 150) return 150;
  return Math.round(value);
}

export default function CustomTopicBuild() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const settings = useSettingsStore();
  const abortRef = useRef<AbortController | null>(null);

  const topic = (searchParams.get("topic") || "").trim();
  const targetSections = useMemo(
    () => clampTargetSections(Number(searchParams.get("target_sections") || "120")),
    [searchParams],
  );

  const [started, setStarted] = useState(false);
  const [progressMessage, setProgressMessage] = useState("Starting analysis...");
  const [streamedSections, setStreamedSections] = useState<StreamedSection[]>([]);
  const [doneTopicId, setDoneTopicId] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    if (!topic) {
      navigate("/topics", { replace: true });
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setStarted(true);
    setErrorMsg("");
    setProgressMessage("Preparing custom topic analysis...");
    setStreamedSections([]);

    createCustomTopicStream(
      {
        topic,
        target_sections: targetSections,
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
      },
      {
        onStart: () => {
          setProgressMessage("Analyzing your topic and building roadmap...");
        },
        onProgress: (event) => {
          setProgressMessage(event.message);
        },
        onSection: (event) => {
          setStreamedSections((prev) => [...prev, event]);
        },
        onDone: (event) => {
          setDoneTopicId(event.topic.id);
          setProgressMessage("Roadmap completed. Redirecting to your topic...");
          window.setTimeout(() => {
            navigate(`/topics/${event.topic.id}`, { replace: true });
          }, 600);
        },
        onError: (event) => {
          setErrorMsg(event.message || "Custom topic generation failed.");
        },
      },
      controller.signal,
    ).catch((err) => {
      if (controller.signal.aborted) return;
      console.error(err);
      if (isCustomTopicStreamError(err)) {
        setErrorMsg(err.message || "Custom topic generation failed.");
        return;
      }
      setErrorMsg("Custom topic generation failed.");
    });

    return () => {
      controller.abort();
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
    };
  }, [navigate, settings.maxTokens, settings.model, settings.provider, settings.temperature, targetSections, topic]);

  const generatedCount = streamedSections.length;

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
      className="max-w-[1100px] mx-auto px-4 sm:px-6 py-8"
    >
      <div className="udemy-card p-6">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-udemy-purple" />
              Building Custom Topic
            </h1>
            <p className="text-sm text-udemy-text-muted mt-1">
              Topic: <span className="font-semibold text-udemy-text">{topic}</span>
            </p>
          </div>
          {doneTopicId ? (
            <span className="inline-flex items-center gap-1 text-xs font-semibold rounded-full bg-green-100 text-green-700 px-2 py-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Completed
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-xs font-semibold rounded-full bg-udemy-purple/10 text-udemy-purple px-2 py-1">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              In Progress
            </span>
          )}
        </div>

        <div className="mt-4 text-sm">
          <p className="text-udemy-text-muted">{progressMessage}</p>
          <p className="mt-1 font-medium">
            Generated sections: {generatedCount} / {targetSections}
          </p>
        </div>

        {errorMsg && (
          <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <div>{errorMsg}</div>
          </div>
        )}

        <div className="mt-5 border border-udemy-border rounded-lg p-3 bg-white">
          <h2 className="text-sm font-bold mb-2">Streamed Course Content</h2>
          <div className="max-h-[420px] overflow-y-auto friendly-scrollbar pr-1 space-y-2">
            {streamedSections.length === 0 && started && !errorMsg ? (
              <p className="text-sm text-udemy-text-muted">
                Waiting for generated sections...
              </p>
            ) : (
              streamedSections.map((sec) => (
                <div
                  key={`${sec.index}-${sec.heading}`}
                  className="rounded border border-udemy-border px-3 py-2"
                >
                  <p className="text-sm font-semibold">
                    {sec.index}. {sec.heading}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>

        {doneTopicId && (
          <button
            onClick={() => navigate(`/topics/${doneTopicId}`)}
            className="btn-primary mt-5"
          >
            Open Generated Topic
          </button>
        )}
      </div>
    </motion.div>
  );
}
