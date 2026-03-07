/**
 * useVoice hook — connects the Zustand store to the voice services.
 *
 * Provides a single `useVoice()` hook that components can call to control
 * voice interactions regardless of the active tier (browser / cloud / realtime).
 */

import { useCallback, useEffect, useRef } from "react";
import { useVoiceStore } from "../store/voiceStore";
import { useSettingsStore } from "../store/settingsStore";
import {
  BrowserVoiceService,
  MicRecorder,
  WebSocketVoiceService,
  blobToBase64,
  fetchVoiceConfig,
  isBrowserSpeechSupported,
  playAudioBase64,
  stopPlayback,
} from "../services/voiceService";
import type { VoiceCallbacks } from "../services/voiceService";
import type { VoiceSessionType, VoiceTier } from "../types";

export function useVoice() {
  const store = useVoiceStore();
  const { provider, model } = useSettingsStore();

  const wsServiceRef = useRef<WebSocketVoiceService | null>(null);
  const browserServiceRef = useRef<BrowserVoiceService | null>(null);
  const micRef = useRef<MicRecorder | null>(null);

  // ── Load config on mount ─────────────────────────────────
  useEffect(() => {
    if (!store.configLoaded) {
      fetchVoiceConfig()
        .then((cfg) => store.setConfig(cfg))
        .catch(() => {
          // Voice not available — disable silently
          store.setConfig({
            enabled: false,
            available_tiers: [],
            default_tier: "browser",
            user_tier: "browser",
            stt_provider: "",
            tts_provider: "",
          });
        });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Callbacks wired to store ─────────────────────────────
  const callbacks: VoiceCallbacks = {
    onReady: (sessionId, tier) => {
      store.setSessionActive(true, sessionId);
      store.setCurrentTier(tier);
      store.setError(null);
    },
    onTranscript: (text, isFinal) => {
      store.setCurrentTranscript(text);
      if (isFinal) {
        store.addUserTurn(text);
      }
    },
    onResponse: (text, audioBase64, latency) => {
      store.addAssistantTurn(text, audioBase64, latency);
      store.setLastLatency(latency);
    },
    onSpeaking: (speaking) => {
      store.setSpeaking(speaking);
    },
    onError: (msg) => {
      store.setError(msg);
    },
    onDisconnect: () => {
      store.setConnected(false);
      store.setSessionActive(false);
    },
  };

  // ── Initialize services lazily ───────────────────────────
  const getWsService = useCallback(() => {
    if (!wsServiceRef.current) {
      wsServiceRef.current = new WebSocketVoiceService(callbacks);
    }
    return wsServiceRef.current;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const getBrowserService = useCallback(() => {
    if (!browserServiceRef.current) {
      browserServiceRef.current = new BrowserVoiceService(callbacks);
    }
    return browserServiceRef.current;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Start a voice session ────────────────────────────────
  const startSession = useCallback(
    async (
      sessionType: VoiceSessionType,
      opts?: {
        sessionId?: string;
        systemPrompt?: string;
        tier?: VoiceTier;
      },
    ) => {
      const tier = opts?.tier ?? store.preferredTier;
      store.setCurrentSessionType(sessionType);
      store.setCurrentTier(tier);
      store.clearTurns();
      store.setError(null);

      if (tier === "browser") {
        // Browser tier uses Web Speech API — no WebSocket needed
        store.setConnected(true);
        store.setSessionActive(true, opts?.sessionId ?? "browser-session");
        return;
      }

      // Cloud / Realtime tier uses WebSocket
      const ws = getWsService();
      try {
        await ws.connect();
        store.setConnected(true);
        ws.startSession({
          tier,
          sessionType,
          sessionId: opts?.sessionId,
          systemPrompt: opts?.systemPrompt,
          llmConfig: { provider, model },
        });
      } catch (err: any) {
        store.setError(err.message || "Failed to connect");
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [provider, model, store.preferredTier],
  );

  // ── Start recording (mic) ───────────────────────────────
  const startRecording = useCallback(async () => {
    const tier = store.currentTier;

    if (tier === "browser") {
      const browser = getBrowserService();
      browser.startListening();
      store.setListening(true);
      store.setRecording(true);
      return;
    }

    // Cloud/realtime: record audio then send when stopped
    try {
      micRef.current = new MicRecorder();
      await micRef.current.start();
      store.setRecording(true);
      store.setListening(true);
    } catch (err: any) {
      store.setError("Microphone access denied");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.currentTier]);

  // ── Stop recording and process ──────────────────────────
  const stopRecording = useCallback(async () => {
    const tier = store.currentTier;

    if (tier === "browser") {
      const browser = getBrowserService();
      browser.stopListening();
      store.setListening(false);
      store.setRecording(false);
      return;
    }

    // Cloud/realtime: stop mic, encode, send via WS
    if (micRef.current) {
      const blob = await micRef.current.stop();
      store.setRecording(false);
      store.setListening(false);

      if (blob.size > 100) {
        const base64 = await blobToBase64(blob);
        const ws = getWsService();
        ws.sendAudio(base64, blob.type || "audio/webm");
      }
      micRef.current = null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.currentTier]);

  // ── Send text (typed fallback or browser tier follow-up) ─
  const sendText = useCallback(
    async (text: string) => {
      const tier = store.currentTier;

      if (tier === "browser") {
        // In browser tier, we can still use the backend LLM via a REST call
        // and then synthesize the response using browser TTS
        store.addUserTurn(text);
        // For now, use WebSocket if available, otherwise queue
        const ws = getWsService();
        if (ws.connected && ws.sessionActive) {
          ws.sendText(text, false); // don't synthesize server-side, we'll use browser TTS
        }
        return;
      }

      const ws = getWsService();
      if (ws.connected && ws.sessionActive) {
        store.addUserTurn(text);
        ws.sendText(text);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [store.currentTier],
  );

  // ── Speak text (TTS only) ──────────────────────────────
  const speak = useCallback(
    (text: string) => {
      const tier = store.currentTier;

      if (tier === "browser") {
        const browser = getBrowserService();
        store.setSpeaking(true);
        browser.speak(text, undefined, () => store.setSpeaking(false));
        return;
      }

      const ws = getWsService();
      if (ws.connected && ws.sessionActive) {
        ws.synthesize(text);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [store.currentTier],
  );

  // ── Stop session ─────────────────────────────────────────
  const stopSession = useCallback(() => {
    const browser = browserServiceRef.current;
    const ws = wsServiceRef.current;

    browser?.destroy();
    browserServiceRef.current = null;

    if (ws?.connected) {
      ws.stopSession();
      ws.disconnect();
    }
    wsServiceRef.current = null;

    micRef.current?.destroy();
    micRef.current = null;
    stopPlayback();

    store.setConnected(false);
    store.setSessionActive(false);
    store.setRecording(false);
    store.setListening(false);
    store.setSpeaking(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Cleanup on unmount ──────────────────────────────────
  useEffect(() => {
    return () => {
      browserServiceRef.current?.destroy();
      wsServiceRef.current?.disconnect();
      micRef.current?.destroy();
    };
  }, []);

  return {
    // State
    config: store.config,
    configLoaded: store.configLoaded,
    voiceEnabled: store.voiceEnabled && (store.config?.enabled ?? false),
    connected: store.connected,
    sessionActive: store.sessionActive,
    isListening: store.isListening,
    isRecording: store.isRecording,
    isSpeaking: store.isSpeaking,
    currentTier: store.currentTier,
    currentSessionType: store.currentSessionType,
    turns: store.turns,
    currentTranscript: store.currentTranscript,
    lastLatency: store.lastLatency,
    error: store.error,
    preferredTier: store.preferredTier,
    autoPlayResponses: store.autoPlayResponses,
    browserSpeechSupported: isBrowserSpeechSupported(),

    // Actions
    startSession,
    stopSession,
    startRecording,
    stopRecording,
    sendText,
    speak,
    setPreferredTier: store.setPreferredTier,
    setVoiceEnabled: store.setVoiceEnabled,
    setAutoPlayResponses: store.setAutoPlayResponses,
    clearError: () => store.setError(null),
    clearTurns: store.clearTurns,
  };
}
