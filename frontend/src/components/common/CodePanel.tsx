import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { normalizeEscapedMultilineText } from "@/utils/textNormalization";

interface CodePanelProps {
  code: string;
  language?: string;
  compact?: boolean;
}

export default function CodePanel({
  code,
  language = "",
  compact = false,
}: CodePanelProps) {
  const [copied, setCopied] = useState(false);
  const lang = (language || "text").toLowerCase();
  const normalizedCode = normalizeEscapedMultilineText(code).replace(/\n$/, "");

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(normalizedCode);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className={`code-panel ${compact ? "code-panel-compact" : ""}`.trim()}>
      <div className="code-panel-header">
        <span className="code-panel-language">{lang}</span>
        <button type="button" onClick={handleCopy} className="code-panel-copy">
          {copied ? "Copied" : "Copy"}
        </button>
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
          wrapLongLines
          wrapLines
          showLineNumbers={!compact}
        >
          {normalizedCode}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
