/**
 * Voice service — manages WebSocket connection, browser speech APIs,
 * and audio playback for all voice tiers.
 */

import type {
  VoiceAudioMessage,
  VoiceClientMessage,
  VoiceConfig,
  VoiceInterviewCompletedEvent,
  VoiceInterviewReadyEvent,
  VoiceInterviewResponseEvent,
  VoiceInterviewStartMessage,
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
  onInterviewReady?: (event: VoiceInterviewReadyEvent) => void;
  onInterviewResponse?: (event: VoiceInterviewResponseEvent) => void;
  onInterviewCompleted?: (event: VoiceInterviewCompletedEvent) => void;
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
let currentAudioOnEnd: (() => void) | null = null;
// Active streamed-TTS player, if any.
let streamingPlayer: StreamingAudioPlayer | null = null;

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
  currentAudioOnEnd = onEnd ?? null;

  const finish = () => {
    if (currentAudio === audio) {
      currentAudio = null;
      currentAudioOnEnd = null;
    }
    onEnd?.();
  };
  audio.onended = finish;
  audio.onerror = finish;
  audio.play().catch(finish);
  return audio;
}

export function stopPlayback() {
  if (streamingPlayer) {
    const player = streamingPlayer;
    streamingPlayer = null;
    player.halt();
  }
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }
  // A stopped playback must settle the speaking state
  // the same way a naturally ended one does.
  const onEnd = currentAudioOnEnd;
  currentAudioOnEnd = null;
  onEnd?.();
}

export function isPlaying(): boolean {
  return currentAudio !== null && !currentAudio.paused;
}

/**
 * Plays queued TTS chunks sequentially so a streamed
 * utterance can be cut mid-stream by an interrupt.
 */
class StreamingAudioPlayer {
  private queue: string[] = [];
  private draining = false;
  private halted = false;
  onEnd?: () => void;

  enqueue(base64: string): void {
    if (this.halted || !base64) return;
    this.queue.push(base64);
    if (!this.draining) {
      void this.drain();
    }
  }

  /** Stop immediately: drop queued chunks and pause playback. */
  halt(): void {
    this.halted = true;
    this.queue = [];
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.currentTime = 0;
      currentAudio = null;
    }
    const onEnd = currentAudioOnEnd;
    currentAudioOnEnd = null;
    onEnd?.();
    this.onEnd?.();
  }

  private async drain(): Promise<void> {
    this.draining = true;
    try {
      while (!this.halted && this.queue.length > 0) {
        const chunk = this.queue.shift() as string;
        await this.playChunk(chunk);
      }
      if (!this.halted) {
        this.onEnd?.();
      }
    } finally {
      this.draining = false;
    }
  }

  private playChunk(base64: string): Promise<void> {
    return new Promise((resolve) => {
      if (this.halted) {
        resolve();
        return;
      }
      const audio = new Audio(`data:audio/mp3;base64,${base64}`);
      currentAudio = audio;
      currentAudioOnEnd = null;
      const finish = () => {
        if (currentAudio === audio) {
          currentAudio = null;
        }
        resolve();
      };
      audio.onended = finish;
      audio.onerror = finish;
      audio.play().catch(finish);
    });
  }
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

// ── Barge-in (VAD) ──────────────────────────────────

/**
 * Detects the user speaking while the assistant's TTS
 * is playing, so the utterance can cut the assistant
 * short (barge-in). Uses the Silero VAD via
 * ``@ricky0123/vad-web``; unsupported browsers (no
 * AudioContext / mic) degrade gracefully — push-to-talk
 * remains the fallback.
 */
export class BargeInDetector {
  private vad: import("@ricky0123/vad-web").MicVAD | null =
    null;
  private fired = false;
  private destroyed = false;

  static isSupported(): boolean {
    return (
      typeof AudioContext !== "undefined" &&
      typeof navigator !== "undefined" &&
      typeof navigator.mediaDevices?.getUserMedia === "function"
    );
  }

  /**
   * Create and start a detector. Resolves to ``null``
   * when VAD is unsupported or fails to initialise —
   * barge-in is optional and must never block the
   * interview flow. Fires ``onBargeIn`` at most once
   * per detector instance.
   */
  static async create(onBargeIn: () => void): Promise<BargeInDetector | null> {
    if (!BargeInDetector.isSupported()) return null;
    try {
      const { MicVAD } = await import("@ricky0123/vad-web");
      const detector = new BargeInDetector();
      detector.vad = await MicVAD.new({
        startOnLoad: false,
        processorType: "auto",
        getStream: () =>
          navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            },
          }),
        onSpeechStart: () => {
          if (detector.destroyed || detector.fired) return;
          detector.fired = true;
          onBargeIn();
        },
        onVADMisfire: () => {
          /* transient noise — ignore */
        },
      });
      await detector.vad.start();
      return detector;
    } catch {
      return null;
    }
  }

  async destroy(): Promise<void> {
    this.destroyed = true;
    if (this.vad) {
      const vad = this.vad;
      this.vad = null;
      try {
        await vad.destroy();
      } catch {
        /* already torn down */
      }
    }
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
        case "audio": {
          const audioMsg = msg as VoiceAudioMessage;
          if (audioMsg.final) {
            // Terminal frame of a streamed synthesis.
            // Queued chunks keep playing; the player
            // settles the speaking state when drained.
            break;
          }
          if (audioMsg.audio) {
            this.callbacks.onSpeaking?.(true);
            if (!streamingPlayer) {
              const player = new StreamingAudioPlayer();
              streamingPlayer = player;
              player.onEnd = () => {
                if (streamingPlayer === player) {
                  streamingPlayer = null;
                }
                this.callbacks.onSpeaking?.(false);
              };
            }
            streamingPlayer.enqueue(audioMsg.audio);
          }
          break;
        }
        case "stopped":
          this._sessionActive = false;
          break;
        case "interview_ready":
          this._sessionActive = true;
          this.callbacks.onInterviewReady?.(msg as VoiceInterviewReadyEvent);
          if ((msg as VoiceInterviewReadyEvent).audio) {
            this.callbacks.onSpeaking?.(true);
            playAudioBase64((msg as VoiceInterviewReadyEvent).audio, () => {
              this.callbacks.onSpeaking?.(false);
            });
          }
          break;
        case "interview_response":
          this.callbacks.onInterviewResponse?.(
            msg as VoiceInterviewResponseEvent,
          );
          if ((msg as VoiceInterviewResponseEvent).audio) {
            this.callbacks.onSpeaking?.(true);
            playAudioBase64(
              (msg as VoiceInterviewResponseEvent).audio,
              () => {
                this.callbacks.onSpeaking?.(false);
              },
            );
          }
          break;
        case "interview_completed":
          this._sessionActive = false;
          this.callbacks.onInterviewCompleted?.(
            msg as VoiceInterviewCompletedEvent,
          );
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

  startInterview(config: VoiceInterviewStartMessage["config"]): void {
    this.send({ type: "interview_start", config });
  }

  sendAudio(
    audioBase64: string,
    mime = "audio/webm",
    responseTimeMs = 0,
  ): void {
    this.send({
      type: "audio",
      data: audioBase64,
      mime,
      response_time_ms: Math.max(0, Math.round(responseTimeMs)),
    });
  }

  sendText(text: string, speak = true): void {
    this.send({ type: "text", content: text, speak });
  }

  /** Cancel the in-flight turn (barge-in). */
  sendInterrupt(): void {
    this.send({ type: "interrupt" });
  }

  synthesize(text: string, stream = false): void {
    this.send({ type: "synthesize", text, stream });
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
