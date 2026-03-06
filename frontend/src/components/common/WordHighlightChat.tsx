import {
  useState,
  useRef,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageCircle, Send, X, Loader2, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { chatFollowUp } from "@/services/api";
import { useSettingsStore } from "@/store/settingsStore";
import type { ChatMessage } from "@/types";

interface Props {
  contextQuestion: string;
  contextAnswer: string;
  /** Unique key to reset chat when Q&A changes */
  qaKey: string;
  children: ReactNode;
}

export default function WordHighlightChat({
  contextQuestion,
  contextAnswer,
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
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const settings = useSettingsStore();

  // Reset chat when Q&A changes
  useEffect(() => {
    setMessages([]);
    setChatOpen(false);
    setChatWord("");
    setSelectedText("");
    setTooltipPos(null);
  }, [qaKey]);

  // Scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleMouseUp = useCallback(() => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || !selection.toString().trim()) {
      // Small delay to allow click on tooltip
      setTimeout(() => {
        if (!chatOpen) {
          setTooltipPos(null);
          setSelectedText("");
        }
      }, 200);
      return;
    }

    const text = selection.toString().trim();
    if (text.length < 2 || text.length > 200) return;

    // Check selection is within our container
    const container = containerRef.current;
    if (!container) return;
    const anchorNode = selection.anchorNode;
    if (!anchorNode || !container.contains(anchorNode)) return;

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    setSelectedText(text);
    setTooltipPos({
      x: rect.left - containerRect.left + rect.width / 2,
      y: rect.top - containerRect.top - 8,
    });
  }, [chatOpen]);

  const openChat = () => {
    setChatWord(selectedText);
    setChatOpen(true);
    setTooltipPos(null);
    // Preserve previous messages — session persists across highlighted words
    setInput(`What does "${selectedText}" mean in this context?`);
  };

  const closeChat = () => {
    setChatOpen(false);
    // Keep messages + chatWord so reopening continues the session
    setInput("");
  };

  const sendMessage = async () => {
    const msg = input.trim();
    if (!msg || sending) return;

    const userMsg: ChatMessage = { role: "user", content: msg };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setSending(true);

    try {
      const res = await chatFollowUp({
        word: chatWord,
        context_question: contextQuestion,
        context_answer: contextAnswer,
        user_message: msg,
        history: newMessages,
        llm_config: {
          provider: settings.provider,
          model: settings.model,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
      });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.reply },
      ]);
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
    <div ref={containerRef} className="relative" onMouseUp={handleMouseUp}>
      {/* Render the wrapped content */}
      {children}

      {/* Tooltip trigger */}
      <AnimatePresence>
        {tooltipPos && !chatOpen && (
          <motion.button
            initial={{ opacity: 0, y: 5, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 5, scale: 0.9 }}
            transition={{ duration: 0.15 }}
            onClick={openChat}
            className="absolute z-50 flex items-center gap-1.5 bg-udemy-purple text-white text-xs font-medium px-3 py-1.5 rounded-full shadow-lg hover:bg-udemy-purple/90 transition-colors whitespace-nowrap"
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

      {/* Chat panel */}
      <AnimatePresence>
        {chatOpen && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 400, damping: 25 }}
            className="absolute right-0 top-0 z-50 w-[380px] max-h-[480px] bg-white rounded-xl shadow-2xl border border-udemy-border flex flex-col overflow-hidden"
            style={{ maxWidth: "calc(100vw - 2rem)" }}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-udemy-purple text-white">
              <div className="flex items-center gap-2 min-w-0">
                <Sparkles className="w-4 h-4 flex-shrink-0" />
                <span className="text-sm font-medium truncate">
                  &ldquo;{chatWord}&rdquo;
                </span>
              </div>
              <button
                onClick={closeChat}
                className="p-1 hover:bg-white/20 rounded transition-colors flex-shrink-0"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-[120px] max-h-[320px]">
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
                      <div className="markdown-content prose prose-xs max-w-none [&>p]:m-0">
                        <ReactMarkdown>{m.content}</ReactMarkdown>
                      </div>
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
              <div ref={chatEndRef} />
            </div>

            {/* Input */}
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
