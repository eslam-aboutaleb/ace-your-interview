/**
 * VoiceButton — floating mic toggle that appears globally when voice is enabled.
 *
 * Shows in the bottom-right corner. Tap to start/stop recording.
 * Long-press opens the full voice panel overlay.
 */

import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, MicOff, X } from "lucide-react";
import { useVoice } from "../../hooks/useVoice";
import VoicePanel from "./VoicePanel";

export default function VoiceButton() {
  const voice = useVoice();
  const [panelOpen, setPanelOpen] = useState(false);
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handlePress = useCallback(() => {
    longPressTimer.current = setTimeout(() => {
      setPanelOpen(true);
      longPressTimer.current = null;
    }, 500);
  }, []);

  const handleRelease = useCallback(() => {
    if (longPressTimer.current) {
      clearTimeout(longPressTimer.current);
      longPressTimer.current = null;
      // Short press — toggle recording
      if (!voice.sessionActive) return;
      if (voice.isRecording) {
        voice.stopRecording();
      } else {
        voice.startRecording();
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voice.isRecording, voice.sessionActive]);

  if (!voice.voiceEnabled) return null;

  return (
    <>
      {/* Floating button */}
      <motion.button
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0, opacity: 0 }}
        onPointerDown={handlePress}
        onPointerUp={handleRelease}
        onPointerLeave={() => {
          if (longPressTimer.current) {
            clearTimeout(longPressTimer.current);
            longPressTimer.current = null;
          }
        }}
        className={`fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full shadow-xl
          flex items-center justify-center transition-colors
          ${
            voice.isRecording
              ? "bg-red-500 hover:bg-red-600 animate-pulse"
              : voice.isSpeaking
                ? "bg-green-500 hover:bg-green-600"
                : voice.sessionActive
                  ? "bg-udemy-purple hover:bg-udemy-purple-light"
                  : "bg-gray-700 hover:bg-gray-600"
          }`}
        title={
          voice.isRecording
            ? "Stop recording"
            : voice.sessionActive
              ? "Start speaking"
              : "Open voice panel"
        }
        onClick={(e) => {
          if (!voice.sessionActive) {
            e.stopPropagation();
            setPanelOpen(true);
          }
        }}
      >
        {voice.isRecording ? (
          <MicOff className="w-6 h-6 text-white" />
        ) : (
          <Mic className="w-6 h-6 text-white" />
        )}

        {/* Recording indicator ring */}
        {voice.isRecording && (
          <motion.span
            className="absolute inset-0 rounded-full border-2 border-red-300"
            animate={{ scale: [1, 1.3, 1], opacity: [0.8, 0, 0.8] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
          />
        )}

        {/* Speaking indicator ring */}
        {voice.isSpeaking && (
          <motion.span
            className="absolute inset-0 rounded-full border-2 border-green-300"
            animate={{ scale: [1, 1.2, 1], opacity: [0.6, 0, 0.6] }}
            transition={{ repeat: Infinity, duration: 1 }}
          />
        )}
      </motion.button>

      {/* Voice panel overlay */}
      <AnimatePresence>
        {panelOpen && <VoicePanel onClose={() => setPanelOpen(false)} />}
      </AnimatePresence>
    </>
  );
}
