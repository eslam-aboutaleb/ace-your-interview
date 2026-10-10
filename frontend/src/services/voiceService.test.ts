import { describe, expect, it, vi } from "vitest";
import {
  BargeInDetector,
  WebSocketVoiceService,
} from "./voiceService";
import type { VoiceCallbacks } from "./voiceService";

const noopCallbacks: VoiceCallbacks = {};

/**
 * Barge-in is optional: on browsers without the Web Audio
 * APIs (jsdom, older Safari) the detector must report
 * unsupported and degrade to a no-op so push-to-talk
 * remains the fallback.
 */
describe("BargeInDetector feature detection", () => {
  it("reports unsupported without AudioContext", () => {
    expect(typeof (window as any).AudioContext).toBe(
      "undefined",
    );
    expect(BargeInDetector.isSupported()).toBe(false);
  });

  it("resolves to null (graceful degradation) when unsupported", async () => {
    const detector = await BargeInDetector.create(() => {
      throw new Error("must not fire");
    });
    expect(detector).toBeNull();
  });
});

describe("WebSocketVoiceService wire protocol", () => {
  const connectFake = (
    service: WebSocketVoiceService,
  ): { sent: Record<string, unknown>[] } => {
    const sent: Record<string, unknown>[] = [];
    const fakeWs = {
      readyState: WebSocket.OPEN,
      send: (data: string) => {
        sent.push(JSON.parse(data));
      },
    };
    (service as any).ws = fakeWs;
    return { sent };
  };

  it("sends the measured response time with audio turns", () => {
    const service = new WebSocketVoiceService(noopCallbacks);
    const { sent } = connectFake(service);
    service.sendAudio("base64data", "audio/webm", 1500);
    expect(sent).toEqual([
      {
        type: "audio",
        data: "base64data",
        mime: "audio/webm",
        response_time_ms: 1500,
      },
    ]);
  });

  it("defaults the response time to zero", () => {
    const service = new WebSocketVoiceService(noopCallbacks);
    const { sent } = connectFake(service);
    service.sendAudio("base64data");
    expect(sent[0]["response_time_ms"]).toBe(0);
  });

  it("sends interrupt messages for barge-in", () => {
    const service = new WebSocketVoiceService(noopCallbacks);
    const { sent } = connectFake(service);
    service.sendInterrupt();
    expect(sent).toEqual([{ type: "interrupt" }]);
  });

  it("opts into chunked TTS streaming", () => {
    const service = new WebSocketVoiceService(noopCallbacks);
    const { sent } = connectFake(service);
    service.synthesize("read this aloud", true);
    expect(sent).toEqual([
      { type: "synthesize", text: "read this aloud", stream: true },
    ]);
  });

  it("keeps full synthesis as the default", () => {
    const service = new WebSocketVoiceService(noopCallbacks);
    const { sent } = connectFake(service);
    service.synthesize("read this aloud");
    expect(sent).toEqual([
      { type: "synthesize", text: "read this aloud", stream: false },
    ]);
  });
});
