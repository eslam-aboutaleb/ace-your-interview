import { ChevronLeft, ChevronRight } from "lucide-react";
import type { ReactNode } from "react";

interface ReaderToolbarProps {
  title: string;
  hasPrev: boolean;
  hasNext: boolean;
  onPrev: () => void;
  onNext: () => void;
  children?: ReactNode;
}

/** Toolbar over the reading pane: title, prev/next, action slot (#7). */
export default function ReaderToolbar({
  title,
  hasPrev,
  hasNext,
  onPrev,
  onNext,
  children,
}: ReaderToolbarProps) {
  return (
    <div className="flex items-center gap-2 border-b border-udemy-border px-4 py-2">
      <button
        type="button"
        onClick={onPrev}
        disabled={!hasPrev}
        aria-label="Previous question"
        className="p-2 rounded-md hover:bg-udemy-bg disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <ChevronLeft className="w-5 h-5" aria-hidden="true" />
      </button>
      <h2 className="flex-1 truncate text-sm font-semibold text-udemy-text">{title}</h2>
      {children}
      <button
        type="button"
        onClick={onNext}
        disabled={!hasNext}
        aria-label="Next question"
        className="p-2 rounded-md hover:bg-udemy-bg disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <ChevronRight className="w-5 h-5" aria-hidden="true" />
      </button>
    </div>
  );
}
