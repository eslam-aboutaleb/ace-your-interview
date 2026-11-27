/**
 * Typed rich learning answer schema (issue #3), mirroring
 * backend/app/schemas/answers.py.
 *
 * The reader validates any structured `sections` payload before rendering.
 * The safety rules (no executable HTML, no skipped heading levels) mirror
 * the backend exactly; rendering must additionally escape everything, since
 * validation is a second line of defense, not the first.
 */
import { z } from "zod";

export const ANSWER_SCHEMA_VERSION = "1.0";

const UNSAFE_HTML_RE =
  /<\s*\/?\s*(script|iframe|object|embed|form|input|button|textarea|select|style|link|meta|base|svg|math)\b|\son[a-z]+\s*=|javascript\s*:|data\s*:\s*text\/html|vbscript\s*:/i;

const SECTION_ID_RE = /^[a-z0-9][a-z0-9-]{0,80}$/;
const LANGUAGE_RE = /^[a-z0-9][a-z0-9+#.-]{0,40}$/;
const HEADING_RE = /^(#{1,6})\s+\S/gm;

export function hasUnsafeHtml(text: string): boolean {
  return UNSAFE_HTML_RE.test(text);
}

export function hasHeadingLevelSkip(text: string): boolean {
  let previous = 0;
  for (const match of text.matchAll(HEADING_RE)) {
    const level = match[1].length;
    if (previous && level > previous + 1) return true;
    previous = level;
  }
  return false;
}

const safeMarkdown = (field: string, { min = 1, max = 50000 }: { min?: number; max?: number } = {}) =>
  z
    .string()
    .min(min)
    .max(max)
    .refine((value) => !hasUnsafeHtml(value), `${field}: unsafe_html`)
    .refine((value) => !hasHeadingLevelSkip(value), `${field}: heading_level_skip`);

const safeText = (field: string, { min = 0, max = 2000 }: { min?: number; max?: number } = {}) =>
  z
    .string()
    .min(min)
    .max(max)
    .refine((value) => !hasUnsafeHtml(value), `${field}: unsafe_html`);

const sectionBase = {
  id: z
    .string()
    .max(81)
    .refine((value) => value === "" || SECTION_ID_RE.test(value), "invalid_section_id")
    .default(""),
  title: safeText("title").default(""),
};

export const summarySection = z.object({
  ...sectionBase,
  type: z.literal("summary"),
  text: safeMarkdown("summary.text"),
});

export const explanationSection = z.object({
  ...sectionBase,
  type: z.literal("explanation"),
  text: safeMarkdown("explanation.text"),
});

export const codeSection = z.object({
  ...sectionBase,
  type: z.literal("code"),
  language: z
    .string()
    .min(1)
    .max(42)
    .transform((value) => value.trim().toLowerCase())
    .pipe(z.string().regex(LANGUAGE_RE, "invalid_language")),
  code: z.string().min(1).max(100000),
  filename: z.string().max(200).default(""),
  line_highlights: z.array(z.number().int().min(1)).max(500).default([]),
});

// NOTE: kept as a plain object (no .refine) because z.discriminatedUnion
// only accepts ZodObject options; the row-width rule lives in richAnswer's
// superRefine below.
export const tableSection = z.object({
  ...sectionBase,
  type: z.literal("table"),
  columns: z.array(safeText("table.cell", { min: 1 })).min(1).max(20),
  rows: z.array(z.array(safeText("table.cell"))).max(500).default([]),
});

export const diagramSection = z.object({
  ...sectionBase,
  type: z.literal("diagram"),
  format: z.literal("mermaid"),
  source: z
    .string()
    .min(1)
    .max(50000)
    .refine((value) => !hasUnsafeHtml(value), "diagram.source: unsafe_html"),
  alt: z.string().min(1).max(500),
});

export const tradeoffsSection = z.object({
  ...sectionBase,
  type: z.literal("tradeoffs"),
  items: z
    .array(
      z.object({
        option: safeText("tradeoffs.option", { min: 1, max: 300 }),
        pros: z.array(safeText("tradeoffs.item")).max(50).default([]),
        cons: z.array(safeText("tradeoffs.item")).max(50).default([]),
      }),
    )
    .min(1)
    .max(50),
});

export const commonMistakesSection = z.object({
  ...sectionBase,
  type: z.literal("common_mistakes"),
  items: z
    .array(
      z.object({
        mistake: safeMarkdown("common_mistakes.mistake", { max: 2000 }),
        correction: safeMarkdown("common_mistakes.correction", { max: 2000 }),
      }),
    )
    .min(1)
    .max(50),
});

export const referencesSection = z.object({
  ...sectionBase,
  type: z.literal("references"),
  items: z
    .array(
      z.object({
        label: safeText("references.label", { min: 1, max: 300 }),
        source_ref: safeText("references.source_ref", { min: 1, max: 500 }),
        quote: safeText("references.quote").default(""),
      }),
    )
    .min(1)
    .max(100),
});

export const followUpSection = z.object({
  ...sectionBase,
  type: z.literal("follow_up"),
  prompts: z
    .array(safeText("follow_up.prompts").refine((v) => v.trim().length > 0, "empty_follow_up_prompt"))
    .min(1)
    .max(20),
});

export const answerSection = z.discriminatedUnion("type", [
  summarySection,
  explanationSection,
  codeSection,
  tableSection,
  diagramSection,
  tradeoffsSection,
  commonMistakesSection,
  referencesSection,
  followUpSection,
]);

export type AnswerSection = z.infer<typeof answerSection>;

export const richAnswer = z
  .object({
    answer_schema_version: z.string().default(ANSWER_SCHEMA_VERSION),
    sections: z.array(answerSection).min(1).max(100),
  })
  .superRefine((value, ctx) => {
    const ids = value.sections.map((s) => s.id).filter(Boolean);
    if (new Set(ids).size !== ids.length) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: "duplicate_section_id" });
    }
    value.sections.forEach((section, index) => {
      if (section.type === "table") {
        section.rows.forEach((row, rowIndex) => {
          if (row.length !== section.columns.length) {
            ctx.addIssue({
              code: z.ZodIssueCode.custom,
              message: `table_row_width:${rowIndex}`,
              path: ["sections", index, "rows", rowIndex],
            });
          }
        });
      }
    });
  });

export type RichAnswer = z.infer<typeof richAnswer>;

/** Validate a structured payload; returns issues instead of throwing. */
export function validateRichAnswer(payload: unknown):
  | { ok: true; answer: RichAnswer }
  | { ok: false; issues: string[] } {
  const result = richAnswer.safeParse(payload);
  if (result.success) {
    return { ok: true, answer: result.data };
  }
  return {
    ok: false,
    issues: result.error.issues.map((issue) => `${issue.path.join(".")}: ${issue.message}`),
  };
}
