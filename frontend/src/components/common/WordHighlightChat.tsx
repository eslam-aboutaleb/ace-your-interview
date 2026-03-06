import {
  useState,
  useRef,
  useCallback,
  useEffect,
  useMemo,
  type ReactNode,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageCircle, Send, X, Loader2, Sparkles } from "lucide-react";
import { chatFollowUp } from "@/services/api";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import { useSettingsStore } from "@/store/settingsStore";
import type { ChatMessage } from "@/types";

interface Props {
  contextQuestion: string;
  contextAnswer: string;
  sessionKey: string;
  topicId?: string;
  topicTitle?: string;
  topicTrack?: string;
  sectionTitle?: string;
  mode?: "study" | "quiz";
  selectionTargetSelector?: string;
  /** Unique key to reset transient UI state when Q&A changes */
  qaKey: string;
  children: ReactNode;
}

const DEFAULT_SELECTION_SELECTOR = '[data-word-chat-target="true"]';
const messageCacheBySession = new Map<string, ChatMessage[]>();

function getNodeElement(node: Node | null): Element | null {
  if (!node) return null;
  if (node.nodeType === Node.ELEMENT_NODE) return node as Element;
  return node.parentElement;
}

export default function WordHighlightChat({
  contextQuestion,
  contextAnswer,
  sessionKey,
  topicId = "",
  topicTitle = "",
  topicTrack = "",
  sectionTitle = "",
  mode,
  selectionTargetSelector = DEFAULT_SELECTION_SELECTOR,
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
  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const settings = useSettingsStore();

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
  }, [qaKey]);

  // Restore per-topic chat thread when session key changes.
  useEffect(() => {
    setMessages(messageCacheBySession.get(sessionKey) || []);
    setChatOpen(false);
    setChatWord("");
    setSelectedText("");
    setTooltipPos(null);
    setInput("");
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
    const x = rect.left - containerRect.left + rect.width / 2;
    const clampedX = Math.min(Math.max(x, 28), Math.max(containerRect.width - 28, 28));
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

  const openChat = () => {
    if (!selectedText) return;
    setMessages(messageCacheBySession.get(sessionKey) || []);
    setChatWord(selectedText);
    setChatOpen(true);
    setTooltipPos(null);
    setInput(`What does "${selectedText}" mean in this context?`);
  };

  const closeChat = () => {
    setChatOpen(false);
    setInput("");
  };

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
        user_message: msg,
        history: newMessages,
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
          <motion.button
            initial={{ opacity: 0, y: 5, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 5, scale: 0.9 }}
            transition={{ duration: 0.15 }}
            onClick={openChat}
            className="absolute z-50 flex items-center gap-1.5 bg-udemy-purple text-white text-xs font-medium px-3 py-1.5 rounded-full shadow-lg hover:bg-udemy-purple/90 transition-colors whitespace-nowrap max-sm:px-3 max-sm:py-2"
            style={{
              left: tooltipPos.x,
              top: tooltipPos.y,
              transform: "translate(-50%, -100%)",
            }}
          >
            <MessageCircle className="w-3.5 h-3.5" />
            Ask about this
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
            className="absolute right-0 top-0 z-50 w-[380px] h-[480px] max-h-[75vh] bg-white rounded-xl shadow-2xl border border-udemy-border flex flex-col overflow-hidden max-sm:fixed max-sm:inset-x-2 max-sm:bottom-2 max-sm:top-auto max-sm:h-[62vh] max-sm:max-h-[72vh] max-sm:w-auto"
            style={{ maxWidth: "calc(100vw - 2rem)" }}
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
              {messages.length === 0 && (
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
                    className={`max-w-[85%] rounded-lg px-3 py-2 text-[13px] leading-relaxed ${
                      m.role === "user"
                        ? "bg-udemy-purple text-white rounded-br-sm"
                        : "bg-udemy-bg text-udemy-text rounded-bl-sm"
                    }`}
                  >
                    {m.role === "assistant" ? (
                      <MarkdownRenderer content={m.content} compact className="text-[13px]" />
                    ) : (
                      m.content
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

            <div className="border-t border-udemy-border px-3 py-2">
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
