/**
 * Voice service — manages WebSocket connection, browser speech APIs,
 * and audio playback for all voice tiers.
 */

import type {
  VoiceClientMessage,
  VoiceConfig,
  VoiceLatency,
  VoiceServerMessage,
  VoiceSessionType,
  VoiceTier,
} from "../types";

// ── API base URL ────────────────────────────────────────────
const API_URL = import.meta.env.VITE_API_URL || "";

function resolveWsUrl(): string {
  const base = API_URL || window.location.origin;
  const url = new URL(base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return `${url.origin}/api/voice/stream`;
}

function resolveHttpUrl(path: string): string {
  return `${API_URL}${path}`;
}

// ── Types ───────────────────────────────────────────────────

export interface VoiceCallbacks {
  onReady?: (sessionId: string, tier: VoiceTier) => void;
  onTranscript?: (text: string, isFinal: boolean) => void;
  onResponse?: (
    text: string,
    audioBase64: string,
    latency: VoiceLatency,
  ) => void;
  onSpeaking?: (speaking: boolean) => void;
  onError?: (message: string) => void;
  onDisconnect?: () => void;
}

// ── Fetch voice config ──────────────────────────────────────

export async function fetchVoiceConfig(): Promise<VoiceConfig> {
  const res = await fetch(resolveHttpUrl("/api/voice/config"), {
    credentials: "include",
  });
  if (!res.ok) throw new Error(`Voice config failed: ${res.status}`);
  return res.json();
}

// ── Audio playback helper ───────────────────────────────────

let currentAudio: HTMLAudioElement | null = null;

export function playAudioBase64(
  base64: string,
  onEnd?: () => void,
): HTMLAudioElement | null {
  if (!base64) {
    onEnd?.();
    return null;
  }
  // Stop any currently playing audio
  stopPlayback();

  const audio = new Audio(`data:audio/mp3;base64,${base64}`);
  currentAudio = audio;

  audio.onended = () => {
    currentAudio = null;
    onEnd?.();
  };
  audio.onerror = () => {
    currentAudio = null;
    onEnd?.();
  };
  audio.play().catch(() => {
    currentAudio = null;
    onEnd?.();
  });
  return audio;
}

export function stopPlayback() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }
}

export function isPlaying(): boolean {
  return currentAudio !== null && !currentAudio.paused;
}

// ── Browser Speech API (free tier) ──────────────────────────

const SpeechRecognition =
  (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

export function isBrowserSpeechSupported(): boolean {
  return !!SpeechRecognition && !!window.speechSynthesis;
}

export class BrowserVoiceService {
  private recognition: any = null;
  private callbacks: VoiceCallbacks;
  private isListening = false;

  constructor(callbacks: VoiceCallbacks) {
    this.callbacks = callbacks;
  }

  startListening(lang = "en-US") {
    if (!SpeechRecognition) {
      this.callbacks.onError?.("Browser speech recognition not supported");
      return;
    }
    if (this.isListening) return;

    this.recognition = new SpeechRecognition();
    this.recognition.continuous = false;
    this.recognition.interimResults = true;
    this.recognition.lang = lang;

    this.recognition.onresult = (event: any) => {
      const result = event.results[event.results.length - 1];
      const text = result[0].transcript;
      const isFinal = result.isFinal;
      this.callbacks.onTranscript?.(text, isFinal);
    };

    this.recognition.onerror = (event: any) => {
      if (event.error !== "aborted") {
        this.callbacks.onError?.(`Speech recognition error: ${event.error}`);
      }
      this.isListening = false;
    };

    this.recognition.onend = () => {
      this.isListening = false;
    };

    this.recognition.start();
    this.isListening = true;
  }

  stopListening(): void {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
      this.isListening = false;
    }
  }

  speak(text: string, voice?: string, onEnd?: () => void): void {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.0;
    utter.pitch = 1.0;

    if (voice) {
      const voices = window.speechSynthesis.getVoices();
      const match = voices.find((v) =>
        v.name.toLowerCase().includes(voice.toLowerCase()),
      );
      if (match) utter.voice = match;
    }

    this.callbacks.onSpeaking?.(true);
    utter.onend = () => {
      this.callbacks.onSpeaking?.(false);
      onEnd?.();
    };
    utter.onerror = () => {
      this.callbacks.onSpeaking?.(false);
      onEnd?.();
    };

    window.speechSynthesis.speak(utter);
  }

  stopSpeaking(): void {
    window.speechSynthesis.cancel();
    this.callbacks.onSpeaking?.(false);
  }

  destroy(): void {
    this.stopListening();
    this.stopSpeaking();
  }
}

// ── WebSocket Voice Service (cloud tier) ─────────────────

export class WebSocketVoiceService {
  private ws: WebSocket | null = null;
  private callbacks: VoiceCallbacks;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _sessionActive = false;

  constructor(callbacks: VoiceCallbacks) {
    this.callbacks = callbacks;
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get sessionActive(): boolean {
    return this._sessionActive;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      const url = resolveWsUrl();
      // Token will be sent via cookie (withCredentials equivalent)
      this.ws = new WebSocket(url);

      const timeout = setTimeout(() => {
        reject(new Error("WebSocket connection timeout"));
        this.ws?.close();
      }, 10000);

      this.ws.onopen = () => {
        clearTimeout(timeout);
        resolve();
      };

      this.ws.onclose = () => {
        clearTimeout(timeout);
        this._sessionActive = false;
        this.callbacks.onDisconnect?.();
      };

      this.ws.onerror = () => {
        clearTimeout(timeout);
        reject(new Error("WebSocket connection failed"));
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };
    });
  }

  private handleMessage(data: string) {
    try {
      const msg: VoiceServerMessage = JSON.parse(data);

      switch (msg.type) {
        case "ready":
          this._sessionActive = true;
          this.callbacks.onReady?.(msg.session_id, msg.tier);
          break;
        case "transcript":
          this.callbacks.onTranscript?.(msg.text, msg.final);
          break;
        case "response":
          this.callbacks.onResponse?.(msg.text, msg.audio, msg.latency);
          // Auto-play audio
          if (msg.audio) {
            this.callbacks.onSpeaking?.(true);
            playAudioBase64(msg.audio, () => {
              this.callbacks.onSpeaking?.(false);
            });
          }
          break;
        case "audio":
          if (msg.audio) {
            this.callbacks.onSpeaking?.(true);
            playAudioBase64(msg.audio, () => {
              this.callbacks.onSpeaking?.(false);
            });
          }
          break;
        case "stopped":
          this._sessionActive = false;
          break;
        case "error":
          this.callbacks.onError?.(msg.message);
          break;
      }
    } catch {
      this.callbacks.onError?.("Failed to parse server message");
    }
  }

  send(msg: VoiceClientMessage): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      this.callbacks.onError?.("WebSocket not connected");
      return;
    }
    this.ws.send(JSON.stringify(msg));
  }

  startSession(opts: {
    tier: VoiceTier;
    sessionType: VoiceSessionType;
    sessionId?: string;
    llmConfig?: { provider: string; model: string };
  }): void {
    this.send({
      type: "start",
      tier: opts.tier,
      session_type: opts.sessionType,
      session_id: opts.sessionId,
      llm_config: opts.llmConfig,
    });
  }

  sendAudio(audioBase64: string, mime = "audio/webm"): void {
    this.send({ type: "audio", data: audioBase64, mime });
  }

  sendText(text: string, speak = true): void {
    this.send({ type: "text", content: text, speak });
  }

  synthesize(text: string): void {
    this.send({ type: "synthesize", text });
  }

  stopSession(): void {
    this.send({ type: "stop" });
    this._sessionActive = false;
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this._sessionActive = false;
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    stopPlayback();
  }
}

// ── MediaRecorder helper for capturing mic audio ────────────

export class MicRecorder {
  private stream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private _recording = false;

  get recording(): boolean {
    return this._recording;
  }

  async start(mimeType = "audio/webm;codecs=opus"): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Check supported MIME type
    const mime = MediaRecorder.isTypeSupported(mimeType)
      ? mimeType
      : "audio/webm";

    this.mediaRecorder = new MediaRecorder(this.stream, { mimeType: mime });
    this.chunks = [];

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };

    this.mediaRecorder.start();
    this._recording = true;
  }

  stop(): Promise<Blob> {
    return new Promise((resolve) => {
      if (!this.mediaRecorder || this.mediaRecorder.state === "inactive") {
        resolve(new Blob([]));
        return;
      }

      this.mediaRecorder.onstop = () => {
        const blob = new Blob(this.chunks, {
          type: this.mediaRecorder?.mimeType || "audio/webm",
        });
        this._recording = false;
        resolve(blob);
      };

      this.mediaRecorder.stop();
      this.stream?.getTracks().forEach((t) => t.stop());
    });
  }

  destroy(): void {
    if (this.mediaRecorder?.state !== "inactive") {
      try {
        this.mediaRecorder?.stop();
      } catch {
        /* ignore */
      }
    }
    this.stream?.getTracks().forEach((t) => t.stop());
    this._recording = false;
  }
}

// ── Blob → Base64 helper ────────────────────────────────────

export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Strip data URL prefix
      const base64 = result.split(",")[1] || "";
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}
