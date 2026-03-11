const AUTO_ID_PREFIX = "app";
let observer: MutationObserver | null = null;
const usedIds = new Set<string>();
const TEXT_HINT_TAGS = new Set(["a", "button", "label", "option", "summary", "h1", "h2", "h3", "h4", "h5", "h6"]);

const sanitize = (value: string, maxLength = 28): string =>
  value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, maxLength);

const currentRouteHint = (): string => {
  if (typeof window === "undefined") {
    return "app";
  }

  const path = window.location.pathname === "/" ? "home" : window.location.pathname.replace(/\//g, "-");
  return sanitize(path, 20) || "home";
};

const textHint = (element: Element): string => {
  if (!TEXT_HINT_TAGS.has(element.tagName.toLowerCase())) {
    return "";
  }

  const raw = (element.textContent || "").replace(/\s+/g, " ").trim();
  if (!raw) {
    return "";
  }

  return sanitize(raw.split(" ").slice(0, 4).join("-"), 32);
};

const idHint = (element: Element): string => {
  const fromName = element.getAttribute("name");
  const fromAria = element.getAttribute("aria-label");
  const fromTestId = element.getAttribute("data-testid");
  const fromRole = element.getAttribute("role");
  const fromPlaceholder = element.getAttribute("placeholder");
  const fromTitle = element.getAttribute("title");
  const fromType = element.getAttribute("type");

  return sanitize(
    fromName ||
      fromAria ||
      fromPlaceholder ||
      fromTitle ||
      fromTestId ||
      fromRole ||
      fromType ||
      textHint(element) ||
      "",
  );
};

const parentHint = (element: Element): string => {
  const parent = element.parentElement;
  if (!parent) {
    return "";
  }

  if (parent.id) {
    const parts = parent.id.split("-").filter(Boolean).reverse();
    const candidate = parts.find((part) => !/^\d+$/.test(part) && part !== AUTO_ID_PREFIX);
    if (candidate) {
      return sanitize(candidate, 18);
    }
  }

  return sanitize(parent.tagName.toLowerCase(), 12);
};

const baseId = (element: Element): string => {
  const route = currentRouteHint();
  const tag = element.tagName.toLowerCase();
  const hint = idHint(element);
  const parent = parentHint(element);

  const segments = [AUTO_ID_PREFIX, route, parent, tag, hint].filter(Boolean);
  const dedupedSegments = segments.filter((segment, index) => segment !== segments[index - 1]);
  return dedupedSegments.join("-").slice(0, 90).replace(/-+$/g, "");
};

const reserve = (id: string): void => {
  if (id) {
    usedIds.add(id);
  }
};

const nextUniqueId = (base: string): string => {
  let candidate = base;
  let suffix = 1;

  while (usedIds.has(candidate) || document.getElementById(candidate)) {
    candidate = `${base}-${suffix}`;
    suffix += 1;
  }

  reserve(candidate);
  return candidate;
};

const assignId = (element: Element): void => {
  if (element.id) {
    reserve(element.id);
    return;
  }

  const generatedId = nextUniqueId(baseId(element));
  element.id = generatedId;
  element.setAttribute("data-auto-id", "true");
};

const walkAndAssign = (element: Element): void => {
  assignId(element);
  element.querySelectorAll("*").forEach(assignId);
};

const shouldProcessNode = (node: Node): node is Element =>
  node instanceof HTMLElement || node instanceof SVGElement;

export const startAutoElementIds = (): (() => void) => {
  if (typeof document === "undefined" || !document.body) {
    return () => {};
  }

  if (observer) {
    return () => observer?.disconnect();
  }

  document.querySelectorAll("[id]").forEach((element) => reserve((element as HTMLElement).id));

  walkAndAssign(document.body);

  observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (shouldProcessNode(node)) {
          walkAndAssign(node);
        }
      });
    });
  });

  observer.observe(document.body, { childList: true, subtree: true });

  return () => {
    observer?.disconnect();
    observer = null;
  };
};
