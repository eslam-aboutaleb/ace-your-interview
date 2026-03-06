import type { Variants, Transition } from "framer-motion";

// ── Page transitions ────────────────────────────────────────
export const pageVariants: Variants = {
  initial: { opacity: 0, x: 20 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -20 },
};

export const pageTransition: Transition = {
  type: "tween",
  ease: "easeInOut",
  duration: 0.3,
};

// ── Card stagger ────────────────────────────────────────────
export const containerVariants: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08 },
  },
};

export const cardVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: {
    opacity: 1,
    y: 0,
    transition: { type: "spring", stiffness: 300, damping: 24 },
  },
};

// ── Accordion expand ────────────────────────────────────────
export const expandVariants: Variants = {
  collapsed: {
    height: 0,
    opacity: 0,
    transition: { duration: 0.3, ease: "easeInOut" },
  },
  expanded: {
    height: "auto",
    opacity: 1,
    transition: { duration: 0.4, ease: "easeOut" },
  },
};

// ── Fade-in scale (for modals, celebrations) ────────────────
export const scaleInVariants: Variants = {
  hidden: { opacity: 0, scale: 0.85 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: { type: "spring", stiffness: 400, damping: 25 },
  },
  exit: { opacity: 0, scale: 0.85, transition: { duration: 0.2 } },
};

// ── Progress bar spring ─────────────────────────────────────
export const progressSpring: Transition = {
  type: "spring",
  stiffness: 100,
  damping: 15,
};

// ── Sidebar slide ───────────────────────────────────────────
export const sidebarVariants: Variants = {
  hidden: { x: -280, opacity: 0 },
  visible: {
    x: 0,
    opacity: 1,
    transition: { type: "spring", stiffness: 300, damping: 30 },
  },
};

// ── Score ring ──────────────────────────────────────────────
export const scoreRingVariants: Variants = {
  hidden: { pathLength: 0, opacity: 0 },
  visible: {
    pathLength: 1,
    opacity: 1,
    transition: { duration: 1.5, ease: "easeOut" },
  },
};

// ── Skeleton pulse ──────────────────────────────────────────
export const pulseVariants: Variants = {
  pulse: {
    opacity: [0.4, 1, 0.4],
    transition: { duration: 1.5, repeat: Infinity, ease: "easeInOut" },
  },
};
