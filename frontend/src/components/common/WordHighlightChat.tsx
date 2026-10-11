import {
  useState,
  useRef,
  useCallback,
  useEffect,
  useMemo,
  type ReactNode,
} from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageCircle,
  Send,
  X,
  Loader2,
  Sparkles,
  BookOpenCheck,
  Layers,
  CalendarClock,
  Mic,
} from "lucide-react";
import { chatFollowUp, chatHighlightAction } from "@/services/api";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import { useSettingsStore } from "@/store/settingsStore";
import type {
  ChatMessage,
  ChatHighlightActionResponse,
  HighlightAction,
  ResponseDetail,
} from "@/types";

interface Props {
  contextQuestion: string;
  contextAnswer: string;
  sessionKey: string;
  topicId?: string;
  topicTitle?: string;
  topicTrack?: string;
  sectionTitle?: string;
  /** Handbook section the Q&A was generated from, for grounded citations. */
  sourceSection?: string;
  /** Verbatim quote from the handbook section the Q&A was generated from. */
  sourceQuote?: string;
  mode?: "study" | "quiz";
  selectionTargetSelector?: string;
  showFloatingTrigger?: boolean;
  floatingTriggerWord?: string;
  floatingTriggerPrompt?: string;
  responseDetail?: ResponseDetail;
  preferredLanguage?: string;
  requiresProgramming?: boolean;
  /** Unique key to reset transient UI state when Q&A changes */
  qaKey: string;
  children: ReactNode;
}

const DEFAULT_SELECTION_SELECTOR = '[data-word-chat-target="true"]';
const messageCacheBySession = new Map<string, ChatMessage[]>();

interface HighlightActionSpec {
  action: HighlightAction;
  label: string;
  icon: typeof BookOpenCheck;
}

/** Action bar shown on a selection, and again inside the open panel. */
const HIGHLIGHT_ACTIONS: HighlightActionSpec[] = [
  { action: "explain", label: "Explain", icon: BookOpenCheck },
  { action: "flashcard", label: "Add flashcard", icon: Layers },
  { action: "review", label: "Schedule review", icon: CalendarClock },
  { action: "interview_angle", label: "Interview angle", icon: Mic },
];

interface HighlightActionResult {
  id: number;
  action: HighlightAction;
  word: string;
  status: "loading" | "done" | "error";
  reply: string;
  card: ChatHighlightActionResponse | null;
  error: string;
}

function getNodeElement(node: Node | null): Element | null {
  if (!node) return null;
  if (node.nodeType === Node.ELEMENT_NODE) return node as Element;
  return node.parentElement;
}

function formatDueDate(dueAt: string): string {
  if (!dueAt) return "";
  const parsed = new Date(dueAt);
  if (Number.isNaN(parsed.getTime())) return dueAt;
  return parsed.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export default function WordHighlightChat({
  contextQuestion,
  contextAnswer,
  sessionKey,
  topicId = "",
  topicTitle = "",
  topicTrack = "",
  sectionTitle = "",
  sourceSection = "",
  sourceQuote = "",
  mode,
  selectionTargetSelector = DEFAULT_SELECTION_SELECTOR,
  showFloatingTrigger = false,
  floatingTriggerWord = "",
  floatingTriggerPrompt = "",
  responseDetail = "concise",
  preferredLanguage = "",
  requiresProgramming = false,
  qaKey,
  children,
}: Props) {
  const [selectedText, setSelectedText] = useState("");
  const [tooltipPos, setTooltipPos] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatWord, setChatWord] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>(
    () => messageCacheBySession.get(sessionKey) || [],
  );
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [actionResults, setActionResults] = useState<HighlightActionResult[]>([]);
  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  // A click on an action button can collapse the selection, so the word
  // the popover was opened for is kept in a ref instead of read back.
  const selectedWordRef = useRef("");
  const nextActionResultId = useRef(1);
  const settings = useSettingsStore();
  const actionPending = actionResults.some((r) => r.status === "loading");

  const targetSelector = useMemo(
    () => selectionTargetSelector || DEFAULT_SELECTION_SELECTOR,
    [selectionTargetSelector],
  );

  // Reset transient state when question changes; history persists by sessionKey.
  useEffect(() => {
    setChatOpen(false);
    setChatWord("");
    setSelectedText("");
    setTooltipPos(null);
    setInput("");
    setActionResults([]);
    selectedWordRef.current = "";
  }, [qaKey]);

  // Restore per-topic chat thread when session key changes.
  useEffect(() => {
    setMessages(messageCacheBySession.get(sessionKey) || []);
    setChatOpen(false);
    setChatWord("");
    setSelectedText("");
    setTooltipPos(null);
    setInput("");
    setActionResults([]);
    selectedWordRef.current = "";
  }, [sessionKey]);

  // Persist thread by topic session key.
  useEffect(() => {
    messageCacheBySession.set(sessionKey, messages);
  }, [sessionKey, messages]);

  // Scroll to bottom on new messages.
  useEffect(() => {
    const container = messagesScrollRef.current;
    if (!container) return;
    container.scrollTo({
      top: container.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const isSelectionInTarget = useCallback(
    (selection: Selection, container: HTMLDivElement) => {
      const anchorNode = selection.anchorNode;
      const focusNode = selection.focusNode;
      if (!anchorNode || !focusNode) return false;
      if (!container.contains(anchorNode) || !container.contains(focusNode)) {
        return false;
      }

      const anchorElement = getNodeElement(anchorNode);
      const focusElement = getNodeElement(focusNode);
      if (!anchorElement || !focusElement) return false;

      return (
        !!anchorElement.closest(targetSelector) &&
        !!focusElement.closest(targetSelector)
      );
    },
    [targetSelector],
  );

  const updateSelection = useCallback(() => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || !selection.toString().trim()) {
      if (!chatOpen) {
        setTooltipPos(null);
        setSelectedText("");
      }
      return;
    }

    const text = selection.toString().trim();
    if (text.length < 2 || text.length > 200) {
      setTooltipPos(null);
      setSelectedText("");
      return;
    }

    const container = containerRef.current;
    if (!container) return;

    if (!isSelectionInTarget(selection, container)) {
      setTooltipPos(null);
      setSelectedText("");
      return;
    }

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    if (!rect.width && !rect.height) {
      setTooltipPos(null);
      setSelectedText("");
      return;
    }

    setSelectedText(text);
    selectedWordRef.current = text;
    const x = rect.left - containerRect.left + rect.width / 2;
    // The popover is wider than a single button now, so its half width
    // decides how close to the container edge it may sit.
    const margin = Math.min(160, Math.max(28, containerRect.width / 2));
    const clampedX = Math.min(
      Math.max(x, margin),
      Math.max(containerRect.width - margin, margin),
    );
    const y = rect.top - containerRect.top - 8;
    setTooltipPos({
      x: clampedX,
      y: Math.max(y, 16),
    });
  }, [chatOpen, isSelectionInTarget]);

  const handleSelectionEnd = useCallback(() => {
    // Let the browser finish native selection updates on touch devices.
    setTimeout(updateSelection, 0);
  }, [updateSelection]);

  const openChat = (word: string, seedPrompt: string) => {
    const normalizedWord = (word || "").trim();
    if (!normalizedWord) return;
    setMessages(messageCacheBySession.get(sessionKey) || []);
    setChatWord(normalizedWord);
    setChatOpen(true);
    setTooltipPos(null);
    setInput(seedPrompt.trim());
  };

  const openSelectionChat = () => {
    if (!selectedText) return;
    openChat(
      selectedText,
      `What does "${selectedText}" mean in this context?`,
    );
  };

  const openFloatingChat = () => {
    const fallbackWord =
      floatingTriggerWord.trim() ||
      selectedText.trim() ||
      sectionTitle.trim() ||
      topicTitle.trim() ||
      "this topic";
    const prompt =
      floatingTriggerPrompt.trim() ||
      `Explain "${fallbackWord}" in this topic context with practical examples.`;
    openChat(fallbackWord, prompt);
  };

  const closeChat = () => {
    setChatOpen(false);
    setInput("");
  };

  const runHighlightAction = useCallback(
    async (action: HighlightAction, word?: string) => {
      const target = (word || selectedWordRef.current || chatWord).trim();
      if (!target || actionPending) return;
      const resultId = nextActionResultId.current++;
      setChatWord(target);
      setChatOpen(true);
      setTooltipPos(null);
      setActionResults((prev) => [
        {
          id: resultId,
          action,
          word: target,
          status: "loading",
          reply: "",
          card: null,
          error: "",
        },
        ...prev,
      ]);
      try {
        const card = await chatHighlightAction({
          action,
          word: target,
          context_question: contextQuestion,
          context_answer: contextAnswer,
          source_section: sourceSection,
          source_quote: sourceQuote,
          topic_id: topicId,
          topic_title: topicTitle,
          topic_track: topicTrack,
          section_title: sectionTitle,
          mode,
          response_detail: responseDetail,
          preferred_language: preferredLanguage,
          requires_programming: requiresProgramming,
          history: messages,
          conversation_id: sessionKey,
          use_memory: true,
          llm_config: {
            provider: settings.provider,
            model: settings.model,
            temperature: settings.temperature,
            max_tokens: settings.maxTokens,
          },
        });
        setActionResults((prev) =>
          prev.map((result) =>
            result.id === resultId
              ? { ...result, status: "done", reply: card.reply, card }
              : result,
          ),
        );
      } catch (err) {
        console.error(err);
        setActionResults((prev) =>
          prev.map((result) =>
            result.id === resultId
              ? {
                  ...result,
                  status: "error",
                  error: "That action failed. Please try again.",
                }
              : result,
          ),
        );
      }
    },
    [
      actionPending,
      chatWord,
      contextAnswer,
      contextQuestion,
      messages,
      mode,
      preferredLanguage,
      requiresProgramming,
      responseDetail,
      sessionKey,
      settings,
      sourceQuote,
      sourceSection,
      sectionTitle,
      topicId,
      topicTitle,
      topicTrack,
    ],
  );

  const sendMessage = async () => {
    const msg = input.trim();
    if (!msg || sending) return;

    const userMsg: ChatMessage = { role: "user", content: msg };
    const sessionMessages = messageCacheBySession.get(sessionKey) || messages;
    const newMessages = [...sessionMessages, userMsg];
    setMessages(newMessages);
    setInput("");
    setSending(true);

    try {
      const res = await chatFollowUp({
        word: chatWord,
        context_question: contextQuestion,
        context_answer: contextAnswer,
        topic_id: topicId,
        topic_title: topicTitle,
        topic_track: topicTrack,
        section_title: sectionTitle,
        mode,
        response_detail: responseDetail,
        preferred_language: preferredLanguage,
        requires_programming: requiresProgramming,
        user_message: msg,
        history: newMessages,
        conversation_id: sessionKey,
        use_memory: true,
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
      });
      setMessages((prev) => [...prev, { role: "assistant", content: res.reply }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, I couldn't process that. Please try again.",
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative"
      onMouseUp={handleSelectionEnd}
      onTouchEnd={handleSelectionEnd}
      onPointerUp={handleSelectionEnd}
    >
      {children}

      <AnimatePresence>
        {tooltipPos && !chatOpen && (
          <motion.div
            initial={{ opacity: 0, y: 5, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 5, scale: 0.9 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 flex max-w-[min(92vw,22rem)] flex-wrap items-center justify-center gap-1 rounded-xl bg-white p-1.5 shadow-2xl border border-udemy-border"
            data-testid="highlight-selection-popover"
            style={{
              left: tooltipPos.x,
              top: tooltipPos.y,
              transform: "translate(-50%, -100%)",
            }}
          >
            {HIGHLIGHT_ACTIONS.map(({ action, label, icon: Icon }) => (
              <button
                key={action}
                onClick={() => runHighlightAction(action, selectedText)}
                disabled={actionPending}
                className="flex items-center gap-1 rounded-full px-2.5 py-1.5 text-xs font-medium text-udemy-text hover:bg-udemy-bg disabled:opacity-40 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                title={label}
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
              </button>
            ))}
            <button
              onClick={openSelectionChat}
              className="flex items-center gap-1.5 bg-udemy-purple text-white text-xs font-medium px-3 py-1.5 rounded-full shadow-lg hover:bg-udemy-purple/90 transition-colors whitespace-nowrap"
            >
              <MessageCircle className="w-3.5 h-3.5" />
              Ask about this
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showFloatingTrigger && !chatOpen && (
          <motion.button
            initial={{ opacity: 0, y: 10, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.9 }}
            transition={{ duration: 0.2 }}
            onClick={openFloatingChat}
            className="fixed right-4 bottom-[calc(1rem+var(--sab))] z-40 w-12 h-12 rounded-full bg-udemy-purple text-white shadow-xl hover:bg-udemy-purple/90 flex items-center justify-center"
            aria-label="Open AI assistant"
            title="Ask AI about this topic"
          >
            <Sparkles className="w-5 h-5" />
          </motion.button>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {chatOpen && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 400, damping: 25 }}
            className="fixed inset-x-2 z-50 bottom-[calc(0.5rem+var(--sab))] h-[68dvh] max-h-[calc(100dvh-1rem-var(--sab))] bg-white rounded-xl shadow-2xl border border-udemy-border flex flex-col overflow-hidden sm:absolute sm:inset-x-auto sm:bottom-auto sm:right-0 sm:top-0 sm:w-[380px] sm:h-[480px] sm:max-h-[75vh]"
          >
            <div className="relative sticky top-0 z-10 flex items-center px-4 py-3 pr-11 bg-udemy-purple text-white">
              <div className="flex items-center gap-2 min-w-0 flex-1">
                <Sparkles className="w-4 h-4 flex-shrink-0" />
                <span className="text-sm font-medium truncate">
                  &ldquo;{chatWord}&rdquo;
                </span>
              </div>
              <button
                onClick={closeChat}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-1 hover:bg-white/20 rounded transition-colors"
                aria-label="Close chat"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div
              ref={messagesScrollRef}
              className="flex-1 min-h-0 overflow-y-auto px-4 py-3 space-y-3"
            >
              {actionResults.map((result) => (
                <div
                  key={result.id}
                  className="rounded-lg border border-udemy-border bg-white px-3 py-2 text-[13px] leading-relaxed"
                  data-testid={`highlight-action-${result.action}`}
                  data-status={result.status}
                >
                  <p className="text-[11px] font-semibold uppercase tracking-wide text-udemy-text-muted">
                    {HIGHLIGHT_ACTIONS.find((a) => a.action === result.action)?.label ??
                      result.action}
                    {result.word ? ` · “${result.word}”` : ""}
                  </p>
                  {result.status === "loading" && (
                    <div className="flex items-center gap-2 py-2 text-udemy-text-muted">
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Working on it…</span>
                    </div>
                  )}
                  {result.status === "error" && (
                    <p className="text-red-700 pt-1">{result.error}</p>
                  )}
                  {result.status === "done" && result.reply && (
                    <MarkdownRenderer
                      content={result.reply}
                      compact
                      className="text-[13px]"
                    />
                  )}
                  {result.status === "done" && result.card?.card_id && (
                    <div className="pt-1 space-y-1">
                      <p>
                        {result.action === "flashcard"
                          ? `Flashcard added to ${result.card.deck_name || "your deck"}.`
                          : "Review item scheduled."}
                      </p>
                      <p className="text-xs text-udemy-text-muted break-all">
                        Card {result.card.card_id}
                        {result.card.due_at
                          ? ` · due ${formatDueDate(result.card.due_at)}`
                          : ""}
                      </p>
                      <Link
                        to={result.action === "flashcard" ? "/decks" : "/review"}
                        className="inline-block text-udemy-purple font-medium underline"
                      >
                        {result.action === "flashcard"
                          ? "Open Flashcards"
                          : "Open Review Queue"}
                      </Link>
                    </div>
                  )}
                </div>
              ))}
              {messages.length === 0 && actionResults.length === 0 && (
                <p className="text-xs text-udemy-text-muted text-center py-4">
                  Ask anything about &ldquo;{chatWord}&rdquo; in this context.
                </p>
              )}
              {messages.map((m, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-lg px-3 py-2 text-[13px] leading-relaxed break-words ${
                      m.role === "user"
                        ? "bg-udemy-purple text-white rounded-br-sm"
                        : "bg-udemy-bg text-udemy-text rounded-bl-sm"
                    }`}
                  >
                    {m.role === "assistant" ? (
                      <MarkdownRenderer content={m.content} compact className="text-[13px]" />
                    ) : (
                      <span className="whitespace-pre-wrap break-words">{m.content}</span>
                    )}
                  </div>
                </motion.div>
              ))}
              {sending && (
                <div className="flex justify-start">
                  <div className="bg-udemy-bg rounded-lg px-3 py-2">
                    <Loader2 className="w-4 h-4 animate-spin text-udemy-purple" />
                  </div>
                </div>
              )}
            </div>

            <div className="border-t border-udemy-border px-3 py-2 space-y-2">
              <div className="flex flex-wrap items-center gap-1" data-testid="highlight-panel-actions">
                {HIGHLIGHT_ACTIONS.map(({ action, label, icon: Icon }) => (
                  <button
                    key={action}
                    onClick={() => runHighlightAction(action, chatWord)}
                    disabled={!chatWord.trim() || actionPending}
                    className="flex items-center gap-1 rounded-full border border-udemy-border px-2 py-1 text-[11px] font-medium text-udemy-text hover:bg-udemy-bg disabled:opacity-40 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    title={label}
                  >
                    <Icon className="w-3 h-3" />
                    {label}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a follow-up..."
                  className="flex-1 text-sm border border-udemy-border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-udemy-purple/30 focus:border-udemy-purple"
                  disabled={sending}
                  autoFocus
                />
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || sending}
                  className="p-2 bg-udemy-purple text-white rounded-lg hover:bg-udemy-purple/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
