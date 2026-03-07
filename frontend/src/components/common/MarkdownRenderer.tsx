import type { ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import CodePanel from "@/components/common/CodePanel";
import MermaidDiagram from "@/components/common/MermaidDiagram";
import { normalizeEscapedMultilineText } from "@/utils/textNormalization";

interface MarkdownRendererProps {
  content: string;
  className?: string;
  compact?: boolean;
}

function isDividerRow(row: string): boolean {
  if (!row.startsWith("|") || !row.endsWith("|")) return false;
  const cells = row
    .slice(1, -1)
    .split("|")
    .map((cell) => cell.trim())
    .filter(Boolean);
  if (cells.length < 2) return false;
  return cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function normalizeInlinePipeTableLine(line: string): string {
  if (!line.includes("|")) return line;

  const firstPipe = line.indexOf("|");
  const lastPipe = line.lastIndexOf("|");
  if (firstPipe === -1 || lastPipe <= firstPipe) return line;

  const tableInlineChunk = line.slice(firstPipe, lastPipe + 1);
  if (!/\|\s+\|/.test(tableInlineChunk)) return line;

  const rows = tableInlineChunk
    .replace(/\|\s+\|/g, "|\n|")
    .split("\n")
    .map((row) => row.trim())
    .filter(Boolean);

  if (rows.length < 2) return line;
  if (!rows.every((row) => row.startsWith("|") && row.endsWith("|"))) return line;
  if (!isDividerRow(rows[1])) return line;

  const prefix = line.slice(0, firstPipe).trimEnd();
  const suffix = line.slice(lastPipe + 1).trimStart();
  const rebuilt: string[] = [];

  if (prefix) rebuilt.push(prefix, "");
  rebuilt.push(...rows);
  if (suffix) rebuilt.push("", suffix);

  return rebuilt.join("\n");
}

function normalizeStarStyledLine(line: string): string {
  const trimmed = line.trim();
  if (!trimmed) return line;

  // *Label*: value -> **Label:** value
  const labelMatch = trimmed.match(/^\*{1,2}\s*([^*\n][^*\n]{1,80}?)\s*\*{1,2}\s*:\s*(.+)$/);
  if (labelMatch) {
    return `**${labelMatch[1].trim()}:** ${labelMatch[2].trim()}`;
  }

  // *Title* -> ### Title
  const headingMatch = trimmed.match(/^\*{1,2}\s*([^*\n][^*\n]{2,80}?)\s*\*{1,2}$/);
  if (headingMatch) {
    return `### ${headingMatch[1].trim()}`;
  }

  return line;
}

function normalizeMarkdownForRendering(content: string): string {
  const decoded = normalizeEscapedMultilineText(content);
  const normalizedNewlines = decoded.replace(/\r\n?/g, "\n");
  const tableFixed = normalizedNewlines
    .split("\n")
    .map(normalizeInlinePipeTableLine)
    .join("\n");
  return tableFixed.split("\n").map(normalizeStarStyledLine).join("\n");
}

function childrenToText(children: ReactNode): string {
  if (children == null) return "";
  if (typeof children === "string" || typeof children === "number") {
    return String(children);
  }
  if (Array.isArray(children)) {
    return children.map(childrenToText).join("");
  }
  if (typeof children === "object" && "props" in children) {
    const candidate = children as { props?: { children?: ReactNode } };
    return childrenToText(candidate.props?.children);
  }
  return "";
}

function getMarkdownComponents(compact: boolean): Components {
  return {
    table: ({ node: _node, ...props }) => (
      <div className="markdown-table-wrap">
        <table {...props} />
      </div>
    ),
    code: ({ node: _node, className, children, ...props }) => {
      const codeText = childrenToText(children);
      const isBlock = (className || "").includes("language-") || codeText.includes("\n");

      if (!isBlock) {
        return (
          <code className={className} {...props}>
            {children}
          </code>
        );
      }

      const match = /language-([a-zA-Z0-9_-]+)/.exec(className || "");
      const language = (match?.[1] || "").toLowerCase();
      if (language === "mermaid") {
        return <MermaidDiagram code={codeText} compact={compact} />;
      }
      return <CodePanel code={codeText} language={language || "text"} compact={compact} />;
    },
  };
}

export default function MarkdownRenderer({
  content,
  className = "",
  compact = false,
}: MarkdownRendererProps) {
  const compactClass = compact ? " markdown-content-compact" : "";
  const normalizedContent = normalizeMarkdownForRendering(content || "");
  return (
    <div className={`markdown-content${compactClass} ${className}`.trim()}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={getMarkdownComponents(compact)}>
        {normalizedContent}
      </ReactMarkdown>
    </div>
  );
}
