/**
 * VoicePanel — slide-up overlay showing voice conversation, controls, and settings.
 */

import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Mic,
  MicOff,
  X,
  Send,
  Volume2,
  VolumeX,
  Settings,
  Zap,
  Wifi,
  Globe,
} from "lucide-react";
import { useVoice } from "../../hooks/useVoice";
import type { VoiceSessionType, VoiceTier } from "../../types";

const TIER_LABELS: Record<
  VoiceTier,
  { label: string; icon: typeof Globe; desc: string }
> = {
  browser: {
    label: "Browser",
    icon: Globe,
    desc: "Free — uses browser speech APIs",
  },
  cloud: {
    label: "Cloud",
    icon: Wifi,
    desc: "~$0.01/session — Groq STT + Edge TTS",
  },
  realtime: {
    label: "Realtime",
    icon: Zap,
    desc: "~$0.50+/session — OpenAI Realtime",
  },
};

const SESSION_TYPE_LABELS: Record<VoiceSessionType, string> = {
  chat: "Chat",
  interview: "Interview",
  qa: "Q&A / Quiz",
};

interface VoicePanelProps {
  onClose: () => void;
  /** Pre-select session type (contextual — e.g., opened from interview page) */
  defaultSessionType?: VoiceSessionType;
  /** Pre-fill session ID for interview sessions */
  sessionId?: string;
  /** Custom system prompt */
  systemPrompt?: string;
}

export default function VoicePanel({
  onClose,
  defaultSessionType = "chat",
  sessionId,
  systemPrompt,
}: VoicePanelProps) {
  const voice = useVoice();
  const [textInput, setTextInput] = useState("");
  const [selectedType, setSelectedType] =
    useState<VoiceSessionType>(defaultSessionType);
  const [showSettings, setShowSettings] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new turns
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [voice.turns, voice.currentTranscript]);

  const handleStart = () => {
    voice.startSession(selectedType, {
      sessionId,
      systemPrompt,
      tier: voice.preferredTier,
    });
  };

  const handleStop = () => {
    voice.stopSession();
  };

  const handleSendText = () => {
    const text = textInput.trim();
    if (!text) return;
    voice.sendText(text);
    setTextInput("");
  };

  const handleMicToggle = () => {
    if (voice.isRecording) {
      voice.stopRecording();
    } else {
      voice.startRecording();
    }
  };

  const availableTiers = voice.config?.available_tiers ?? ["browser"];

  return (
    <motion.div
      initial={{ y: "100%" }}
      animate={{ y: 0 }}
      exit={{ y: "100%" }}
      transition={{ type: "spring", damping: 25, stiffness: 300 }}
      className="fixed inset-x-0 bottom-0 z-[60] max-h-[85vh] bg-udemy-dark border-t
        border-gray-700 rounded-t-2xl shadow-2xl flex flex-col"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700/50">
        <div className="flex items-center gap-2">
          <Mic className="w-5 h-5 text-udemy-purple" />
          <span className="font-semibold text-white text-sm">
            Voice{" "}
            {voice.sessionActive
              ? `— ${SESSION_TYPE_LABELS[voice.currentSessionType]}`
              : "Agent"}
          </span>
          {voice.sessionActive && (
            <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-green-500/20 text-green-400">
              {voice.currentTier.toUpperCase()}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-1.5 hover:bg-white/10 rounded transition-colors"
            title="Voice settings"
          >
            <Settings className="w-4 h-4 text-gray-400" />
          </button>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-white/10 rounded transition-colors"
          >
            <X className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <div className="px-4 py-3 border-b border-gray-700/50 space-y-3">
          {/* Tier selector */}
          <div>
            <label className="text-xs text-gray-400 font-medium mb-1 block">
              Voice Tier
            </label>
            <div className="flex gap-2">
              {availableTiers.map((tier) => {
                const info = TIER_LABELS[tier as VoiceTier];
                if (!info) return null;
                const Icon = info.icon;
                return (
                  <button
                    key={tier}
                    onClick={() => voice.setPreferredTier(tier as VoiceTier)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium
                      transition-colors ${
                        voice.preferredTier === tier
                          ? "bg-udemy-purple text-white"
                          : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                      }`}
                    title={info.desc}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    {info.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Auto-play toggle */}
          <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={voice.autoPlayResponses}
              onChange={(e) => voice.setAutoPlayResponses(e.target.checked)}
              className="rounded border-gray-600 bg-gray-700 text-udemy-purple focus:ring-udemy-purple"
            />
            Auto-play audio responses
          </label>
        </div>
      )}

      {/* Pre-session: type selection + start */}
      {!voice.sessionActive && (
        <div className="px-4 py-6 flex flex-col items-center gap-4">
          <p className="text-gray-400 text-sm">
            Select mode and start a voice session
          </p>

          <div className="flex gap-2">
            {(Object.keys(SESSION_TYPE_LABELS) as VoiceSessionType[]).map(
              (t) => (
                <button
                  key={t}
                  onClick={() => setSelectedType(t)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors
                  ${
                    selectedType === t
                      ? "bg-udemy-purple text-white"
                      : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                  }`}
                >
                  {SESSION_TYPE_LABELS[t]}
                </button>
              ),
            )}
          </div>

          <button
            onClick={handleStart}
            className="mt-2 px-6 py-3 bg-udemy-purple hover:bg-udemy-purple-light text-white
              rounded-full font-semibold text-sm transition-colors flex items-center gap-2"
          >
            <Mic className="w-4 h-4" />
            Start Voice Session
          </button>

          {voice.error && (
            <p className="text-red-400 text-xs mt-1">{voice.error}</p>
          )}
        </div>
      )}

      {/* Active session: conversation + controls */}
      {voice.sessionActive && (
        <>
          {/* Conversation area */}
          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-[200px] max-h-[50vh]"
          >
            {voice.turns.length === 0 && !voice.currentTranscript && (
              <p className="text-gray-500 text-sm text-center py-8">
                {voice.isRecording
                  ? "Listening..."
                  : "Tap the mic button or type a message to begin"}
              </p>
            )}

            {voice.turns.map((turn, i) => (
              <div
                key={i}
                className={`flex ${turn.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm ${
                    turn.role === "user"
                      ? "bg-udemy-purple/20 text-gray-100"
                      : "bg-gray-700/50 text-gray-200"
                  }`}
                >
                  <p>{turn.text}</p>
                  {turn.latency && (
                    <p className="text-[10px] text-gray-500 mt-1">
                      {turn.latency.total_ms}ms total
                      {turn.latency.stt_ms > 0 &&
                        ` • STT ${turn.latency.stt_ms}ms`}
                      {turn.latency.llm_ms > 0 &&
                        ` • LLM ${turn.latency.llm_ms}ms`}
                      {turn.latency.tts_ms > 0 &&
                        ` • TTS ${turn.latency.tts_ms}ms`}
                    </p>
                  )}
                  {turn.role === "assistant" && turn.audioBase64 && (
                    <button
                      onClick={() => voice.speak(turn.text)}
                      className="mt-1 text-[10px] text-udemy-purple hover:text-udemy-purple-light"
                    >
                      🔊 Replay
                    </button>
                  )}
                </div>
              </div>
            ))}

            {/* Interim transcript */}
            {voice.currentTranscript && (
              <div className="flex justify-end">
                <div className="max-w-[80%] rounded-2xl px-4 py-2.5 text-sm bg-udemy-purple/10 text-gray-400 italic">
                  {voice.currentTranscript}...
                </div>
              </div>
            )}
          </div>

          {/* Controls bar */}
          <div className="px-4 py-3 border-t border-gray-700/50">
            {/* Error */}
            {voice.error && (
              <p className="text-red-400 text-xs mb-2 flex items-center gap-1">
                <span>⚠</span> {voice.error}
                <button
                  onClick={voice.clearError}
                  className="ml-auto text-gray-500 hover:text-gray-300"
                >
                  ✕
                </button>
              </p>
            )}

            <div className="flex items-center gap-2">
              {/* Mic button */}
              <button
                onClick={handleMicToggle}
                className={`p-3 rounded-full transition-colors ${
                  voice.isRecording
                    ? "bg-red-500 hover:bg-red-600 animate-pulse"
                    : "bg-gray-700 hover:bg-gray-600"
                }`}
                title={voice.isRecording ? "Stop recording" : "Start speaking"}
              >
                {voice.isRecording ? (
                  <MicOff className="w-5 h-5 text-white" />
                ) : (
                  <Mic className="w-5 h-5 text-white" />
                )}
              </button>

              {/* Text input */}
              <div className="flex-1 flex items-center gap-2 bg-gray-700/50 rounded-full px-4 py-2">
                <input
                  type="text"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSendText()}
                  placeholder="Type a message..."
                  className="flex-1 bg-transparent text-sm text-white placeholder-gray-500
                    outline-none"
                />
                <button
                  onClick={handleSendText}
                  disabled={!textInput.trim()}
                  className="p-1 text-udemy-purple hover:text-udemy-purple-light disabled:text-gray-600
                    transition-colors"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>

              {/* Stop session */}
              <button
                onClick={handleStop}
                className="p-2 text-gray-400 hover:text-red-400 transition-colors"
                title="End voice session"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
        </>
      )}
    </motion.div>
  );
}
