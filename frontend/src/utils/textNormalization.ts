const ESCAPED_NEWLINE_RE = /\\n/g;

function countMatches(text: string, pattern: RegExp): number {
  const matches = text.match(pattern);
  return matches ? matches.length : 0;
}

function shouldDecodeEscapedNewlines(text: string): boolean {
  if (!text.includes("\\n")) return false;

  const escapedCount = countMatches(text, ESCAPED_NEWLINE_RE);
  const actualCount = countMatches(text, /\n/g);

  const hasEscapedParagraphs = text.includes("\\n\\n");
  const hasEscapedCodeFence = /```[a-zA-Z0-9_-]*\\n/.test(text) || /\\n```/.test(text);
  const hasEscapedMarkdownStructure = /\\n(?:#{1,6}\s|[-*]\s|\d+\.\s|\|)/.test(text);

  if (hasEscapedParagraphs || hasEscapedCodeFence || hasEscapedMarkdownStructure) {
    return true;
  }

  if (escapedCount >= 2 && actualCount <= Math.max(1, Math.floor(escapedCount / 2))) {
    return true;
  }

  return false;
}

export function normalizeEscapedMultilineText(input: string): string {
  const normalized = String(input || "").replace(/\r\n?/g, "\n");
  if (!shouldDecodeEscapedNewlines(normalized)) {
    return normalized;
  }
  return normalized.replace(ESCAPED_NEWLINE_RE, "\n");
}

export function normalizeEscapedSingleLineText(input: string): string {
  const multiline = normalizeEscapedMultilineText(input);
  return multiline
    .replace(/\n+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();
}
