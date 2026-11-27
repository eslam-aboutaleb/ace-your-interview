import { memo, useMemo, useRef, useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { normalizeEscapedMultilineText } from "@/utils/textNormalization";

/**
 * Syntax-highlighted code panel (issue #10).
 *
 * - Prism + oneDark (existing pipeline), unknown languages fall back to
 *   plain text rendering without throwing.
 * - Horizontal overflow is the default; a Wrap toggle soft-wraps long
 *   lines. Line numbers follow the compact default and can be toggled.
 * - Copy gives feedback through an aria-live region (and reports
 *   clipboard failures instead of pretending to succeed).
 * - Optional filename and highlighted lines (typed code section
 *   contract from the answer schema work).
 * - Memoized so parent re-renders do not re-highlight unchanged code.
 */

interface CodePanelProps {
  code: string;
  language?: string;
  compact?: boolean;
  filename?: string;
  lineHighlights?: number[];
  /** Override the line-number default (!compact). */
  showLineNumbers?: boolean;
  /** Override the wrap default (off = horizontal scroll). */
  wrap?: boolean;
}

const HIGHLIGHT_STYLE: React.CSSProperties = {
  display: "block",
  backgroundColor: "rgba(250, 204, 21, 0.14)",
  borderLeft: "2px solid rgba(250, 204, 21, 0.6)",
};

function CodePanel({
  code,
  language = "",
  compact = false,
  filename = "",
  lineHighlights,
  showLineNumbers,
  wrap,
}: CodePanelProps) {
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);
  const [wrapOverride, setWrapOverride] = useState<boolean | null>(null);
  const [lineNumbersOverride, setLineNumbersOverride] = useState<boolean | null>(null);
  const feedbackTimer = useRef<number | null>(null);

  const lang = (language || "text").toLowerCase();
  const normalizedCode = useMemo(
    () => normalizeEscapedMultilineText(code).replace(/\n$/, ""),
    [code],
  );

  const wrapLongLines = wrapOverride ?? wrap ?? false;
  const lineNumbers = lineNumbersOverride ?? showLineNumbers ?? !compact;

  const highlights = useMemo(() => new Set(lineHighlights ?? []), [lineHighlights]);

  const flashFeedback = (setter: (value: boolean) => void) => {
    if (feedbackTimer.current) window.clearTimeout(feedbackTimer.current);
    setter(true);
    feedbackTimer.current = window.setTimeout(() => setter(false), 1600);
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(normalizedCode);
      setCopyFailed(false);
      flashFeedback(setCopied);
    } catch {
      setCopied(false);
      flashFeedback(setCopyFailed);
    }
  };

  return (
    <div className={`code-panel ${compact ? "code-panel-compact" : ""}`.trim()}>
      <div className="code-panel-header">
        <span className="code-panel-language">
          {filename ? `${filename} · ` : ""}
          {lang}
        </span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setLineNumbersOverride(!lineNumbers)}
            aria-pressed={lineNumbers}
            aria-label="Toggle line numbers"
            className="code-panel-copy"
          >
            #
          </button>
          <button
            type="button"
            onClick={() => setWrapOverride(!wrapLongLines)}
            aria-pressed={wrapLongLines}
            aria-label="Toggle line wrap"
            className="code-panel-copy"
          >
            Wrap
          </button>
          <button
            type="button"
            onClick={handleCopy}
            aria-label="Copy code"
            className="code-panel-copy"
          >
            Copy
          </button>
        </div>
      </div>
      <div
        aria-live="polite"
        role="status"
        aria-label="Copy status"
        className="sr-only"
      >
        {copied ? "Copied to clipboard" : copyFailed ? "Copy failed" : ""}
      </div>
      <div className="code-panel-body">
        <SyntaxHighlighter
          language={lang}
          style={oneDark}
          customStyle={{
            margin: 0,
            background: "transparent",
            padding: compact ? "0.75rem" : "1rem",
            fontSize: compact ? "0.75rem" : "0.8125rem",
            lineHeight: 1.6,
          }}
          codeTagProps={{
            style: wrapLongLines
              ? { whiteSpace: "pre-wrap", wordBreak: "break-word" }
              : { whiteSpace: "pre" },
          }}
          wrapLines
          wrapLongLines={wrapLongLines}
          showLineNumbers={lineNumbers}
          lineProps={(lineNumber) =>
            highlights.has(lineNumber as number) ? { style: HIGHLIGHT_STYLE } : {}
          }
        >
          {normalizedCode}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}

export default memo(CodePanel);
