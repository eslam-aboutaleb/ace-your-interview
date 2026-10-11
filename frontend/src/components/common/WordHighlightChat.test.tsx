import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import WordHighlightChat from "./WordHighlightChat";
import { chatFollowUp, chatHighlightAction } from "@/services/api";
import type { ChatHighlightActionResponse } from "@/types";

vi.mock("@/services/api", () => ({
  chatFollowUp: vi.fn(),
  chatHighlightAction: vi.fn(),
}));

vi.mock("@/components/common/MarkdownRenderer", () => ({
  default: ({ content }: { content: string }) => (
    <div data-testid="markdown">{content}</div>
  ),
}));

const HIGHLIGHT_TEXT = "Idempotent APIs keep retries safe.";

function renderChat() {
  return render(
    <MemoryRouter>
      <WordHighlightChat
        contextQuestion="What makes an API idempotent?"
        contextAnswer="Retrying the request leaves the same state."
        sessionKey="study:api-design"
        topicId="api-design"
        topicTitle="API Design"
        topicTrack="backend"
        sectionTitle="HTTP Semantics"
        sourceSection="HTTP Semantics"
        sourceQuote="Retrying the request leaves the same state."
        mode="study"
        qaKey="api-design:q1"
      >
        <p data-word-chat-target="true">{HIGHLIGHT_TEXT}</p>
      </WordHighlightChat>
    </MemoryRouter>,
  );
}

function selectHighlightedWord() {
  const target = document.querySelector('[data-word-chat-target="true"]');
  if (!target) throw new Error("no highlight target");
  const range = document.createRange();
  range.selectNodeContents(target);
  // jsdom reports an empty rect, which the component treats as no selection.
  range.getBoundingClientRect = () =>
    ({
      width: 60,
      height: 12,
      top: 0,
      left: 0,
      bottom: 12,
      right: 60,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }) as DOMRect;
  const selection = window.getSelection();
  selection?.removeAllRanges();
  selection?.addRange(range);
  fireEvent.mouseUp(target);
}

/** Click one action on the selection popover, waiting for it to appear. */
async function clickSelectionAction(name: RegExp) {
  const popover = await screen.findByTestId("highlight-selection-popover");
  fireEvent.click(within(popover).getByRole("button", { name }));
}

function actionResponse(
  overrides: Partial<ChatHighlightActionResponse> = {},
): ChatHighlightActionResponse {
  return {
    action: "explain",
    reply: "### Meaning\nAn idempotent request leaves the same state.",
    provider_used: "openai",
    model_used: "gpt-4o-mini",
    card_id: "",
    deck_id: "",
    deck_name: "",
    due_at: "",
    source_section: "HTTP Semantics",
    source_quote: "Retrying the request leaves the same state.",
    ...overrides,
  };
}

describe("WordHighlightChat action bar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(chatHighlightAction).mockResolvedValue(actionResponse());
  });

  it("offers the four deep actions on a selection", async () => {
    renderChat();
    selectHighlightedWord();

    const popover = await screen.findByTestId("highlight-selection-popover");
    expect(
      within(popover).getByRole("button", { name: /Explain/ }),
    ).toBeInTheDocument();
    expect(
      within(popover).getByRole("button", { name: /Add flashcard/ }),
    ).toBeInTheDocument();
    expect(
      within(popover).getByRole("button", { name: /Schedule review/ }),
    ).toBeInTheDocument();
    expect(
      within(popover).getByRole("button", { name: /Interview angle/ }),
    ).toBeInTheDocument();
    expect(
      within(popover).getByRole("button", { name: /Ask about this/ }),
    ).toBeInTheDocument();
  });

  it("runs the explain action with the highlighted term and its context", async () => {
    renderChat();
    selectHighlightedWord();
    await clickSelectionAction(/Explain/);

    await waitFor(() =>
      expect(chatHighlightAction).toHaveBeenCalledWith(
        expect.objectContaining({
          action: "explain",
          word: HIGHLIGHT_TEXT,
          context_question: "What makes an API idempotent?",
          context_answer: "Retrying the request leaves the same state.",
          source_section: "HTTP Semantics",
          source_quote: "Retrying the request leaves the same state.",
          topic_id: "api-design",
          topic_title: "API Design",
          topic_track: "backend",
          section_title: "HTTP Semantics",
          mode: "study",
        }),
      ),
    );

    const panel = await screen.findByTestId("highlight-action-explain");
    expect(panel).toHaveAttribute("data-status", "done");
    expect(screen.getByTestId("markdown")).toHaveTextContent(
      "An idempotent request leaves the same state.",
    );
  });

  it("confirms a created flashcard with its card id and a Flashcards link", async () => {
    vi.mocked(chatHighlightAction).mockResolvedValue(
      actionResponse({
        action: "flashcard",
        reply: "",
        card_id: "card-42",
        deck_id: "deck-1",
        deck_name: "Highlights - API Design",
        due_at: "2026-10-11T14:30:00.000Z",
      }),
    );
    renderChat();
    selectHighlightedWord();
    await clickSelectionAction(/Add flashcard/);

    const panel = await screen.findByTestId("highlight-action-flashcard");
    expect(panel).toHaveAttribute("data-status", "done");
    expect(
      screen.getByText("Flashcard added to Highlights - API Design."),
    ).toBeInTheDocument();
    expect(screen.getByText(/Card card-42/)).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Open Flashcards" }),
    ).toHaveAttribute("href", "/decks");
  });

  it("confirms a scheduled review with its card id, due date, and Review Queue link", async () => {
    vi.mocked(chatHighlightAction).mockResolvedValue(
      actionResponse({
        action: "review",
        reply: "",
        card_id: "highlight:api-design:idempotency",
        due_at: "2026-10-11T14:40:00.000Z",
      }),
    );
    renderChat();
    selectHighlightedWord();
    await clickSelectionAction(/Schedule review/);

    const panel = await screen.findByTestId("highlight-action-review");
    expect(panel).toHaveAttribute("data-status", "done");
    expect(screen.getByText("Review item scheduled.")).toBeInTheDocument();
    expect(
      screen.getByText(/Card highlight:api-design:idempotency/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Open Review Queue" }),
    ).toHaveAttribute("href", "/review");
  });

  it("runs the interview angle action", async () => {
    renderChat();
    selectHighlightedWord();
    await clickSelectionAction(/Interview angle/);

    await waitFor(() =>
      expect(chatHighlightAction).toHaveBeenCalledWith(
        expect.objectContaining({ action: "interview_angle" }),
      ),
    );
    expect(
      await screen.findByTestId("highlight-action-interview_angle"),
    ).toHaveAttribute("data-status", "done");
  });

  it("keeps the follow-up chat untouched when an action runs", async () => {
    renderChat();
    selectHighlightedWord();
    await clickSelectionAction(/Add flashcard/);
    await screen.findByTestId("highlight-action-flashcard");

    expect(chatFollowUp).not.toHaveBeenCalled();
  });

  it("shows an error state when the action fails", async () => {
    vi.mocked(chatHighlightAction).mockRejectedValue(new Error("boom"));
    renderChat();
    selectHighlightedWord();
    await clickSelectionAction(/Add flashcard/);

    const panel = await screen.findByTestId("highlight-action-flashcard");
    expect(panel).toHaveAttribute("data-status", "error");
    expect(
      screen.getByText("That action failed. Please try again."),
    ).toBeInTheDocument();
  });

  it("runs an action from the panel bar and disables it while running", async () => {
    let resolveAction: (value: ChatHighlightActionResponse) => void = () => {};
    vi.mocked(chatHighlightAction).mockReturnValue(
      new Promise<ChatHighlightActionResponse>((resolve) => {
        resolveAction = resolve;
      }),
    );
    renderChat();
    selectHighlightedWord();
    await clickSelectionAction(/Explain/);

    const panelBar = await screen.findByTestId("highlight-panel-actions");
    const panel = await screen.findByTestId("highlight-action-explain");
    expect(panel).toHaveAttribute("data-status", "loading");
    expect(
      within(panelBar).getByRole("button", { name: /Interview angle/ }),
    ).toBeDisabled();
    expect(
      within(panelBar).getByRole("button", { name: /Add flashcard/ }),
    ).toBeDisabled();

    resolveAction(actionResponse());
    await waitFor(() =>
      expect(screen.getByTestId("highlight-action-explain")).toHaveAttribute(
        "data-status",
        "done",
      ),
    );
  });
});
