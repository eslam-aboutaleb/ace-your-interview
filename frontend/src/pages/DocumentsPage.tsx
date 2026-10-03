import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle2,
  FileText,
  Loader2,
  RefreshCw,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import {
  deleteDocument,
  fetchDocuments,
  uploadDocument,
} from "@/services/api";
import ChatAskPanel from "@/components/ChatAskPanel";
import type { DocumentDetail } from "@/types";

const ALLOWED_EXTENSIONS = ["pdf", "txt", "md", "markdown", "docx"];
const MAX_FILE_SIZE = 25 * 1024 * 1024;

function formatDateTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function statusBadge(status: DocumentDetail["status"]) {
  switch (status) {
    case "ready":
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-udemy-success-bg px-2 py-0.5 text-xs font-medium text-udemy-success">
          <CheckCircle2 className="w-3 h-3" />
          Ready
        </span>
      );
    case "processing":
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-udemy-purple/10 px-2 py-0.5 text-xs font-medium text-udemy-purple">
          <Loader2 className="w-3 h-3 animate-spin" />
          Processing
        </span>
      );
    case "failed":
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-600">
          <AlertTriangle className="w-3 h-3" />
          Failed
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600">
          Uploaded
        </span>
      );
  }
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentDetail[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");
  const [disabledNotice, setDisabledNotice] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [deletingIds, setDeletingIds] = useState<Set<string>>(
    new Set(),
  );
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<number | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetchDocuments();
      setDocuments(res.documents || []);
      setErrorMsg("");
      setDisabledNotice("");
      return res.documents || [];
    } catch (err: any) {
      if (err?.response?.status === 503) {
        setDisabledNotice(
          "Document RAG is not enabled on this server " +
            "(STUDY_ENABLE_RAG_V1=false).",
        );
        setErrorMsg("");
      } else {
        setErrorMsg("Could not load documents. Please retry.");
      }
      return [];
    }
  }, []);

  useEffect(() => {
    (async () => {
      setLoading(true);
      await load();
      setLoading(false);
    })();
  }, [load]);

  // Poll while any document is still processing.
  const anyProcessing = documents.some(
    (doc) => doc.status === "processing" || doc.status === "uploaded",
  );
  useEffect(() => {
    if (!anyProcessing) {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    if (pollRef.current !== null) return;
    pollRef.current = window.setInterval(() => {
      void load();
    }, 3000);
    return () => {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [anyProcessing, load]);

  const handleFiles = async (files: FileList | null) => {
    if (!files || !files.length || uploading) return;
    const file = files[0];
    const extension = file.name.split(".").pop()?.toLowerCase() || "";
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setErrorMsg(
        "Unsupported file type. Allowed: PDF, TXT, MD, DOCX.",
      );
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setErrorMsg("File exceeds the 25 MB limit.");
      return;
    }
    setUploading(true);
    setErrorMsg("");
    setUploadProgress(0);
    try {
      await uploadDocument(file, setUploadProgress);
      await load();
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail;
      if (status === 503) {
        setDisabledNotice(
          "Document RAG is not enabled on this server " +
            "(STUDY_ENABLE_RAG_V1=false).",
        );
      } else if (status === 415) {
        setErrorMsg(
          typeof detail === "string"
            ? detail
            : "Unsupported file type. Allowed: PDF, TXT, MD, DOCX.",
        );
      } else if (status === 413) {
        setErrorMsg("File exceeds the 25 MB limit.");
      } else if (status === 429) {
        setErrorMsg(
          typeof detail === "string"
            ? detail
            : "Daily document limit reached.",
        );
      } else {
        setErrorMsg(
          typeof detail === "string"
            ? detail
            : "Upload failed. Please retry.",
        );
      }
    } finally {
      setUploading(false);
      setUploadProgress(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDelete = async (documentId: string) => {
    if (deletingIds.has(documentId)) return;
    setDeletingIds((prev) => new Set(prev).add(documentId));
    try {
      await deleteDocument(documentId);
      setDocuments((prev) =>
        prev.filter((doc) => doc.document_id !== documentId),
      );
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.detail || "Delete failed. Please retry.",
      );
    } finally {
      setDeletingIds((prev) => {
        const next = new Set(prev);
        next.delete(documentId);
        return next;
      });
    }
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
      <div className="udemy-card p-6">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold">Documents</h1>
            <p className="text-sm text-udemy-text-muted mt-1">
              Upload study documents (PDF, TXT, MD, DOCX) and ask
              questions grounded in them with verbatim citations.
            </p>
          </div>
          <button
            onClick={() => void load()}
            disabled={loading}
            className="btn-secondary flex items-center gap-2"
          >
            <RefreshCw
              className={`w-4 h-4 ${loading ? "animate-spin" : ""}`}
            />
            Refresh
          </button>
        </div>

        {disabledNotice && (
          <div className="mt-4 rounded-lg border border-udemy-border bg-udemy-bg p-3 text-sm text-udemy-text-muted">
            {disabledNotice}
          </div>
        )}

        {errorMsg && (
          <div className="mt-4 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        <div
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragActive(false);
            void handleFiles(event.dataTransfer.files);
          }}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Upload a document"
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          className={`mt-5 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors cursor-pointer
            ${
              dragActive
                ? "border-udemy-purple bg-udemy-purple/5"
                : "border-udemy-border hover:border-udemy-purple/50 hover:bg-udemy-bg"
            }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md,.markdown,.docx"
            className="hidden"
            onChange={(event) =>
              void handleFiles(event.target.files)
            }
          />
          {uploading ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="w-8 h-8 text-udemy-purple animate-spin" />
              <p className="text-sm font-medium">
                Uploading… {uploadProgress}%
              </p>
              <div className="h-1.5 w-48 rounded-full bg-udemy-border overflow-hidden">
                <div
                  className="h-full bg-udemy-purple transition-all"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          ) : (
            <>
              <UploadCloud className="w-8 h-8 text-udemy-purple mx-auto mb-2" />
              <p className="text-sm font-medium">
                Drop a document here or click to browse
              </p>
              <p className="text-xs text-udemy-text-muted mt-1">
                PDF, TXT, MD, DOCX · up to 25 MB
              </p>
            </>
          )}
        </div>

        {loading && documents.length === 0 ? (
          <div className="mt-6 space-y-3">
            {[0, 1, 2].map((i) => (
              <div key={i} className="skeleton h-16 w-full" />
            ))}
          </div>
        ) : documents.length === 0 ? (
          <div className="mt-6 rounded-lg border border-udemy-border bg-udemy-bg p-6 text-center">
            <FileText className="w-8 h-8 text-udemy-text-muted mx-auto mb-2" />
            <p className="text-sm text-udemy-text-muted">
              No documents yet. Upload study material to start asking
              citation-grounded questions.
            </p>
          </div>
        ) : (
          <div className="mt-6 space-y-3">
            {documents.map((doc) => (
              <div
                key={doc.document_id}
                className="rounded-lg border border-udemy-border bg-white p-4 flex items-start gap-3"
              >
                <FileText className="w-5 h-5 text-udemy-purple mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium text-[15px] truncate">
                      {doc.title}
                    </span>
                    {statusBadge(doc.status)}
                    <span className="text-xs text-udemy-text-muted">
                      {doc.chunk_count} chunks
                    </span>
                  </div>
                  <p className="text-xs text-udemy-text-muted mt-1 truncate">
                    {doc.filename} · {doc.mime_type} · updated{" "}
                    {formatDateTime(doc.updated_at)}
                  </p>
                  {doc.status === "failed" && doc.error && (
                    <p className="text-xs text-red-600 mt-1">
                      {doc.error}
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => void handleDelete(doc.document_id)}
                  disabled={deletingIds.has(doc.document_id)}
                  className="p-2 rounded-lg text-udemy-text-muted hover:text-red-600 hover:bg-red-50 transition-colors disabled:opacity-50"
                  aria-label={`Delete ${doc.title}`}
                >
                  {deletingIds.has(doc.document_id) ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Trash2 className="w-4 h-4" />
                  )}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="mt-6">
        <ChatAskPanel />
      </div>
    </motion.div>
  );
}