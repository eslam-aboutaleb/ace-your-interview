import { Menu, X } from "lucide-react";
import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

interface ReaderShellProps {
  /** Curriculum rail content (objectives tree). */
  rail: ReactNode;
  /** Center reading pane. */
  children: ReactNode;
  /** Optional right-hand table-of-contents pane (rendered on lg+ only). */
  toc?: ReactNode;
  /** Breadcrumb trail rendered above the toolbar. */
  breadcrumbs?: ReactNode;
  /** Toolbar rendered above the reading pane. */
  toolbar?: ReactNode;
  railOpen: boolean;
  onToggleRail: () => void;
  mobileRailOpen: boolean;
  onToggleMobileRail: () => void;
}

const FOCUSABLE =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

/**
 * Responsive three-region reader shell (issues #7 and #8): a collapsible
 * curriculum rail on the left (a focus-managed drawer on mobile), the
 * center reading pane, and an optional right-hand table of contents.
 * Accessibility: skip link, semantic regions, focus trap and focus
 * restoration for the mobile drawer, reduced-motion-aware animation.
 */
export default function ReaderShell({
  rail,
  children,
  toc,
  breadcrumbs,
  toolbar,
  railOpen,
  onToggleRail,
  mobileRailOpen,
  onToggleMobileRail,
}: ReaderShellProps) {
  const drawerRef = useRef<HTMLDivElement | null>(null);
  const mobileOpenButtonRef = useRef<HTMLButtonElement | null>(null);

  // Mobile drawer focus management: move focus inside on open, trap Tab,
  // and restore focus to the button that opened it on close.
  useEffect(() => {
    if (!mobileRailOpen) return;
    const drawer = drawerRef.current;
    const opener = mobileOpenButtonRef.current;
    if (!drawer) return;
    const focusables = Array.from(drawer.querySelectorAll<HTMLElement>(FOCUSABLE));
    (focusables[0] ?? drawer).focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const items = Array.from(drawer.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
        (el) => el.offsetParent !== null,
      );
      if (items.length === 0) return;
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (event.shiftKey && (active === first || !drawer.contains(active))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (active === last || !drawer.contains(active))) {
        event.preventDefault();
        first.focus();
      }
    };
    drawer.addEventListener("keydown", onKeyDown);
    return () => {
      drawer.removeEventListener("keydown", onKeyDown);
      opener?.focus();
    };
  }, [mobileRailOpen]);

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col bg-udemy-bg">
      <a
        href="#study-reader-main"
        className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:bg-udemy-purple focus:text-white focus:px-4 focus:py-2"
      >
        Skip to content
      </a>
      <div className="flex items-center gap-2 border-b border-udemy-border bg-white px-4 py-2">
        <button
          type="button"
          onClick={onToggleRail}
          aria-label={railOpen ? "Collapse curriculum rail" : "Expand curriculum rail"}
          aria-expanded={railOpen}
          className="hidden md:inline-flex p-2 rounded-md hover:bg-udemy-bg"
        >
          <Menu className="w-5 h-5" aria-hidden="true" />
        </button>
        <button
          type="button"
          ref={mobileOpenButtonRef}
          onClick={onToggleMobileRail}
          aria-label="Open curriculum"
          aria-expanded={mobileRailOpen}
          className="md:hidden p-2 rounded-md hover:bg-udemy-bg"
        >
          <Menu className="w-5 h-5" aria-hidden="true" />
        </button>
        <div className="min-w-0 flex-1">{breadcrumbs}</div>
      </div>
      <div className="flex min-h-0 flex-1">
        {railOpen && (
          <aside
            aria-label="Curriculum"
            className="hidden md:flex w-80 shrink-0 flex-col overflow-y-auto border-r border-udemy-border bg-white"
          >
            {rail}
          </aside>
        )}
        <main
          id="study-reader-main"
          tabIndex={-1}
          aria-label="Reader"
          className="flex min-w-0 flex-1 flex-col bg-white"
        >
          {toolbar}
          <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
        </main>
        {toc && (
          <aside
            aria-label="Table of contents"
            className="hidden lg:flex w-64 shrink-0 flex-col overflow-y-auto border-l border-udemy-border bg-white"
          >
            {toc}
          </aside>
        )}
      </div>
      {mobileRailOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div
            className="absolute inset-0 bg-black/40"
            aria-hidden="true"
            onClick={onToggleMobileRail}
          />
          <div
            ref={drawerRef}
            role="dialog"
            aria-modal="true"
            aria-label="Curriculum"
            tabIndex={-1}
            className="absolute inset-y-0 left-0 w-80 max-w-[85vw] overflow-y-auto bg-white shadow-card"
            onKeyDown={(event) => {
              if (event.key === "Escape") onToggleMobileRail();
            }}
          >
            <div className="flex items-center justify-between border-b border-udemy-border px-4 py-2">
              <span className="text-sm font-semibold text-udemy-text">Curriculum</span>
              <button
                type="button"
                onClick={onToggleMobileRail}
                aria-label="Close curriculum"
                className="p-2 rounded-md hover:bg-udemy-bg"
              >
                <X className="w-5 h-5" aria-hidden="true" />
              </button>
            </div>
            {rail}
          </div>
        </div>
      )}
    </div>
  );
}
