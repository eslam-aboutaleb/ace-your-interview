import CodePanel from "@/components/common/CodePanel";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import MermaidDiagram from "@/components/common/MermaidDiagram";
import { answerSection, type AnswerSection } from "@/schemas/answerSchema";

/**
 * Typed answer renderer (issue #9).
 *
 * Maps each discriminated section of a structured answer payload to a safe
 * component. Safety model: react-markdown (no raw-HTML plugin) and React
 * escaping do the real work; the zod schema is a second line of defense.
 * Sections are validated individually so one malformed or unknown section
 * degrades to an accessible fallback instead of breaking the whole answer.
 */

interface AnswerRendererProps {
  /** Untrusted structured payload (validated per section before render). */
  payload: unknown;
  compact?: boolean;
}

function sectionAnchorId(section: AnswerSection, index: number): string {
  return section.id || `answer-section-${index}`;
}

function isSafeLink(ref: string): boolean {
  return /^https?:\/\//i.test(ref);
}

function SectionFrame({
  anchorId,
  title,
  children,
}: {
  anchorId: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={anchorId} aria-labelledby={`${anchorId}-title`} className="mt-6 first:mt-0">
      {title && (
        <h2 id={`${anchorId}-title`} className="text-lg font-semibold text-udemy-text">
          {title}
        </h2>
      )}
      {children}
    </section>
  );
}

function UnknownSectionFallback({ anchorId, title }: { anchorId: string; title?: string }) {
  return (
    <section id={anchorId} className="mt-6 first:mt-0" aria-label={title || "Unsupported section"}>
      <p className="rounded-md border border-udemy-border bg-udemy-bg p-3 text-sm text-udemy-text-muted">
        This section can&apos;t be displayed yet{title ? `: ${title}` : ""}.
      </p>
    </section>
  );
}

function SectionBody({ section, compact }: { section: AnswerSection; compact: boolean }) {
  switch (section.type) {
    case "summary":
    case "explanation":
      return <MarkdownRenderer content={section.text} compact={compact} />;
    case "code":
      return <CodePanel code={section.code} language={section.language} compact={compact} />;
    case "table":
      return (
        <div className="overflow-x-auto">
          <table className="mt-2 w-full border-collapse text-sm">
            {section.title && <caption className="sr-only">{section.title}</caption>}
            <thead>
              <tr>
                {section.columns.map((column, i) => (
                  <th
                    key={i}
                    scope="col"
                    className="border border-udemy-border bg-udemy-bg px-3 py-2 text-left font-semibold text-udemy-text"
                  >
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {section.rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <td
                      key={cellIndex}
                      className="border border-udemy-border px-3 py-2 text-udemy-text"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "diagram":
      return (
        <figure role="img" aria-label={section.alt}>
          <MermaidDiagram code={section.source} compact={compact} />
          <figcaption className="sr-only">{section.alt}</figcaption>
        </figure>
      );
    case "tradeoffs":
      return (
        <div className="mt-2 space-y-3">
          {section.items.map((item, i) => (
            <div key={i} className="rounded-md border border-udemy-border p-3">
              <h3 className="text-sm font-semibold text-udemy-text">{item.option}</h3>
              <div className="mt-2 grid gap-3 sm:grid-cols-2">
                <div>
                  <p className="text-xs font-semibold uppercase text-udemy-success">Pros</p>
                  <ul className="mt-1 list-disc pl-5 text-sm text-udemy-text">
                    {item.pros.map((pro, j) => (
                      <li key={j}>{pro}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase text-udemy-danger">Cons</p>
                  <ul className="mt-1 list-disc pl-5 text-sm text-udemy-text">
                    {item.cons.map((con, j) => (
                      <li key={j}>{con}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ))}
        </div>
      );
    case "common_mistakes":
      return (
        <div className="mt-2 space-y-3">
          {section.items.map((item, i) => (
            <div key={i} className="rounded-md border border-udemy-border p-3">
              <p className="text-xs font-semibold uppercase text-udemy-danger">Mistake</p>
              <MarkdownRenderer content={item.mistake} compact />
              <p className="mt-2 text-xs font-semibold uppercase text-udemy-success">Fix</p>
              <MarkdownRenderer content={item.correction} compact />
            </div>
          ))}
        </div>
      );
    case "references":
      return (
        <ul className="mt-2 space-y-2">
          {section.items.map((item, i) => (
            <li key={i} className="text-sm text-udemy-text">
              <span className="font-medium">{item.label}</span>
              {" — "}
              {isSafeLink(item.source_ref) ? (
                <a
                  href={item.source_ref}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-udemy-purple underline"
                >
                  {item.source_ref}
                </a>
              ) : (
                <span className="text-udemy-text-muted">{item.source_ref}</span>
              )}
              {item.quote && (
                <blockquote className="mt-1 border-l-2 border-udemy-border pl-3 text-udemy-text-muted">
                  {item.quote}
                </blockquote>
              )}
            </li>
          ))}
        </ul>
      );
    case "follow_up":
      return (
        <ul className="mt-2 list-disc pl-5 text-sm text-udemy-text">
          {section.prompts.map((prompt, i) => (
            <li key={i}>{prompt}</li>
          ))}
        </ul>
      );
    default:
      return null;
  }
}

export default function AnswerRenderer({ payload, compact = false }: AnswerRendererProps) {
  const sections: unknown[] =
    typeof payload === "object" &&
    payload !== null &&
    Array.isArray((payload as { sections?: unknown[] }).sections)
      ? (payload as { sections: unknown[] }).sections
      : [];

  if (sections.length === 0) {
    return (
      <p role="alert" className="text-sm text-udemy-text-muted">
        This answer could not be displayed.
      </p>
    );
  }

  return (
    <div>
      {sections.map((raw, index) => {
        const parsed = answerSection.safeParse(raw);
        if (!parsed.success) {
          const title =
            typeof raw === "object" && raw !== null && "title" in raw
              ? String((raw as { title?: unknown }).title ?? "")
              : "";
          return (
            <UnknownSectionFallback
              key={index}
              anchorId={`answer-section-${index}`}
              title={title}
            />
          );
        }
        const section = parsed.data;
        const anchorId = sectionAnchorId(section, index);
        return (
          <SectionFrame key={anchorId} anchorId={anchorId} title={section.title}>
            <SectionBody section={section} compact={compact} />
          </SectionFrame>
        );
      })}
    </div>
  );
}
