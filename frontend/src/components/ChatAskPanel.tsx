import { useCallback, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  BookOpen,
  ChevronDown,
  FileText,
  Loader2,
  Quote,
  Send,
  X,
} from "lucide-react";
import {
  askDocumentQuestion,
  fetchDocuments,
} from "@/services/api";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import type { Citation, DocumentDetail } from "@/types";

interface ChatAskPanelProps {
  documentIds?: string[];
  topicId?: string;
  className?: string;
}

interface AskTurn {
  question: string;
  answer: string;
  citations: Citation[];
  conversationId: string;
}

export default function ChatAskPanel({
  documentIds,
  topicId,
  className = "",
}: ChatAskPanelProps) {
  const [message, setMessage] = useState("");
  const [turns, setTurns] = useState<AskTurn[]>([]);
  const [conversationId, setConversationId] = useState("");
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");
  const [disabledNotice, setDisabledNotice] = useState("");
  const [documents, setDocuments] = useState<DocumentDetail[]>([]);
  const [expandedCitations, setExpandedCitations] = useState<
    Set<string>
  >(new Set());

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const res = await fetchDocuments();
        if (active) setDocuments(res.documents);
      } catch (e: any) {
        if (e?.response?.status === 503) {
          setDisabledNotice(
            "Document RAG is not enabled on this server " +
              "(STUDY_ENABLE_RAG_V1=false).",
          );
        }
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const documentTitle = useCallback(
    (documentId: string) => {
      const match = documents.find(
        (doc) => doc.document_id === documentId,
      );
      return match ? match.title : documentId.slice(0, 8);
    },
    [documents],
  );

  const handleAsk = async () => {
    const question = message.trim();
    if (!question || asking) return;
    setAsking(true);
    setError("");
    try {
      const res = await askDocumentQuestion({
        message: question,
        document_ids: documentIds,
        topic_id: topicId,
        conversation_id: conversationId || undefined,
      });
      setConversationId(res.conversation_id);
      setTurns((prev) => [
        ...prev,
        {
          question,
          answer: res.answer,
          citations: res.citations,
          conversationId: res.conversation_id,
        },
      ]);
      setMessage("");
    } catch (e: any) {
      const status = e?.response?.status;
      if (status === 503) {
        setDisabledNotice(
          "Document RAG is not enabled on this server " +
            "(STUDY_ENABLE_RAG_V1=false).",
        );
      } else {
        setError(
          e?.response?.data?.detail ||
            e?.message ||
            "Failed to get an answer.",
        );
      }
    } finally {
      setAsking(false);
    }
  };

  const toggleCitation = (key: string) => {
    setExpandedCitations((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  return (
    <div className={`udemy-card p-5 ${className}`}>
      <div className="flex items-center gap-2 mb-3">
        <BookOpen className="w-4 h-4 text-udemy-purple" />
        <h3 className="font-bold text-[15px]">
          Ask your documents
        </h3>
      </div>

      {disabledNotice && (
        <div className="mb-3 rounded-lg border border-udemy-border bg-udemy-bg p-3 text-sm text-udemy-text-muted">
          {disabledNotice}
        </div>
      )}

      <div className="flex gap-2">
        <textarea
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              void handleAsk();
            }
          }}
          placeholder="Ask a question grounded in your uploaded documents…"
          rows={2}
          className="flex-1 rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-udemy-purple/40 resize-none"
        />
        <button
          type="button"
          onClick={() => void handleAsk()}
          disabled={asking || !message.trim()}
          className="btn-primary self-end disabled:opacity-60"
          aria-label="Ask documents"
        >
          {asking ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </div>

      {error && (
        <div className="mt-3 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <AnimatePresence>
        {turns.map((turn, turnIdx) => (
          <motion.div
            key={`${turn.conversationId}-${turnIdx}`}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 space-y-3"
          >
            <div className="text-sm font-medium">
              Q: {turn.question}
            </div>
            <div className="rounded-lg border border-udemy-border bg-white p-3">
              <MarkdownRenderer
                content={turn.answer}
                className="text-[14px]"
              />
            </div>
            {turn.citations.length > 0 && (
              <div className="space-y-2">
                <div className="text-[11px] font-bold uppercase tracking-[0.14em] text-udemy-text-muted flex items-center gap-1">
                  <Quote className="w-3 h-3" />
                  Citations
                </div>
                {turn.citations.map((citation, citationIdx) => {
                  const key = `${turnIdx}-${citationIdx}-${citation.document_id}-${citation.chunk_index}`;
                  const expanded = expandedCitations.has(key);
                  return (
                    <div
                      key={key}
                      className="rounded-lg border border-udemy-purple/20 bg-udemy-purple/5 overflow-hidden"
                    >
                      <button
                        type="button"
                        onClick={() => toggleCitation(key)}
                        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-udemy-purple/10 transition-colors"
                      >
                        <FileText className="w-3.5 h-3.5 text-udemy-purple flex-shrink-0" />
                        <span className="text-xs font-medium text-udemy-purple truncate flex-1">
                          {documentTitle(citation.document_id)} ·
                          chunk {citation.chunk_index}
                        </span>
                        {expanded ? (
                          <X className="w-3.5 h-3.5 text-udemy-text-muted" />
                        ) : (
                          <ChevronDown className="w-3.5 h-3.5 text-udemy-text-muted" />
                        )}
                      </button>
                      <AnimatePresence>
                        {expanded && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="px-3 pb-2"
                          >
                            <blockquote className="rounded border-l-2 border-udemy-purple bg-white px-3 py-2 text-xs text-udemy-text-muted italic">
                              {citation.quote}
                            </blockquote>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  );
                })}
              </div>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}