import { Children, isValidElement, type ReactNode } from "react";
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

const COACH_PARAGRAPH_LABELS = new Set([
  "Answer:",
  "Why it's right:",
  "Common mistake:",
  "What the interviewer is really testing:",
  "Short answer:",
  "How to think about it:",
  "Why it works:",
  "Time and space:",
  "Tradeoff / scaling caveat:",
  "Invariant note:",
]);

const COACH_SECTION_HEADINGS = new Set([
  "Problem",
  "Solution Walkthrough",
  "Complexity",
  "Code",
]);

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

function extractCoachParagraphLabel(children: ReactNode): string | null {
  const nodes = Children.toArray(children);
  if (!nodes.length) return null;
  const firstNode = nodes.find((node) => {
    if (typeof node === "string" || typeof node === "number") {
      return String(node).trim().length > 0;
    }
    return node != null;
  });
  if (!firstNode) return null;
  if (typeof firstNode === "string" || typeof firstNode === "number") {
    const trimmed = String(firstNode).trim();
    return COACH_PARAGRAPH_LABELS.has(trimmed) ? trimmed : null;
  }
  if (!isValidElement(firstNode)) return null;
  const firstText = childrenToText(
    (firstNode as { props?: { children?: ReactNode } }).props?.children,
  ).trim();
  return COACH_PARAGRAPH_LABELS.has(firstText) ? firstText : null;
}

function coachSectionKey(heading: string): string | null {
  const normalized = heading.trim();
  if (!COACH_SECTION_HEADINGS.has(normalized)) return null;
  return normalized.toLowerCase().replace(/\s+/g, "-");
}

function getMarkdownComponents(compact: boolean): Components {
  return {
    h2: ({ node: _node, className, children, ...props }) => (
      <h2 className={`${className || ""} markdown-heading markdown-heading-2`.trim()} {...props}>
        {children}
      </h2>
    ),
    h3: ({ node: _node, className, children, ...props }) => {
      const sectionKey = coachSectionKey(childrenToText(children));
      return (
        <h3
          className={`${className || ""}${sectionKey ? " coach-section-heading" : " markdown-heading markdown-heading-3"}`.trim()}
          data-coach-section={sectionKey || undefined}
          {...props}
        >
          {children}
        </h3>
      );
    },
    h4: ({ node: _node, className, children, ...props }) => (
      <h4 className={`${className || ""} markdown-heading markdown-heading-4`.trim()} {...props}>
        {children}
      </h4>
    ),
    p: ({ node: _node, className, children, ...props }) => {
      const coachLabel = extractCoachParagraphLabel(children);
      return (
        <p
          className={`${className || ""}${coachLabel ? " coach-paragraph" : ""}`.trim()}
          data-coach-label={coachLabel || undefined}
          {...props}
        >
          {children}
        </p>
      );
    },
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
