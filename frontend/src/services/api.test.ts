import { beforeEach, describe, expect, it, vi } from "vitest";

const { responseUseMock } = vi.hoisted(() => ({
  responseUseMock: vi.fn(),
}));

vi.mock("axios", () => ({
  default: {
    create: () => ({
      interceptors: {
        response: {
          use: responseUseMock,
        },
      },
      request: vi.fn(),
      get: vi.fn(),
      post: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
    }),
  },
}));

import "@/services/api";
import { generateQuizV2Stream, generateQuestionsV2Stream } from "@/services/api";

interface MockLocation {
  pathname: string;
  href: string;
}

function mockLocation(pathname: string): MockLocation {
  const location: MockLocation = {
    pathname,
    href: `http://localhost${pathname}`,
  };
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: location,
  });
  return location;
}

function errorHandler(): (error: unknown) => Promise<unknown> {
  const registration = responseUseMock.mock.calls[0];
  return registration[1] as (error: unknown) => Promise<unknown>;
}

describe("api response interceptor", () => {
  let location: MockLocation;

  beforeEach(() => {
    location = mockLocation("/topics");
  });

  it("redirects to /login on a 401", async () => {
    const error = {
      response: { status: 401 },
      config: { url: "/api/topics" },
    };

    await expect(errorHandler()(error)).rejects.toBe(error);
    expect(location.href).toBe("/login");
  });

  it("keeps learning endpoints on a 401 without redirecting", async () => {
    const error = {
      response: { status: 401 },
      config: { url: "/api/learning/review-queue" },
    };

    await expect(errorHandler()(error)).rejects.toBe(error);
    expect(location.href).toBe("http://localhost/topics");
  });

  it("redirects approval-required 403s to the policy settings page", async () => {
    const error = {
      response: {
        status: 403,
        data: { detail: { code: "llm_service_approval_required" } },
      },
      config: { url: "/api/topics" },
    };

    await expect(errorHandler()(error)).rejects.toBe(error);
    expect(location.href).toBe(
      "/user-settings?policy=llm_service_approval_required",
    );
  });
});

function ndjsonResponse(lines: unknown[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const line of lines) {
        controller.enqueue(encoder.encode(`${JSON.stringify(line)}\n`));
      }
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "application/x-ndjson" },
  });
}

const QUIZ_DONE_EVENT = {
  type: "done",
  generated_count: 1,
  target_count: 1,
  topics_used: ["01-static"],
  provider_used: "openai",
  model_used: "gpt-4o-mini",
  retries_used: 0,
  malformed_items_dropped: 0,
  degraded: false,
  requested_count: 1,
  returned_count: 1,
};

const TOPIC_RESOLUTION_WARNING = {
  type: "warning",
  code: "topic_resolution_failed",
  message: "Some requested topics could not be resolved and were skipped.",
  topics: ["99-broken"],
  topics_used: ["01-static"],
};

describe("quiz stream warning events", () => {
  it("surfaces a warning through onWarning without aborting the stream", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        ndjsonResponse([TOPIC_RESOLUTION_WARNING, QUIZ_DONE_EVENT]),
      );
    vi.stubGlobal("fetch", fetchMock);
    try {
      const onWarning = vi.fn();
      const onDone = vi.fn();
      const done = await generateQuizV2Stream(
        { topic_ids: ["01-static"], count: 1, question_types: ["mcq"] },
        { onWarning, onDone },
      );

      expect(onWarning).toHaveBeenCalledTimes(1);
      expect(onWarning.mock.calls[0][0].code).toBe("topic_resolution_failed");
      expect(onWarning.mock.calls[0][0].topics).toEqual(["99-broken"]);
      expect(onWarning.mock.calls[0][0].topics_used).toEqual(["01-static"]);
      expect(onDone).toHaveBeenCalledTimes(1);
      expect(done.degraded).toBe(false);
      expect(done.requested_count).toBe(1);
      expect(done.returned_count).toBe(1);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("never turns a warning into a QuizStreamError", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(ndjsonResponse([TOPIC_RESOLUTION_WARNING, QUIZ_DONE_EVENT]));
    vi.stubGlobal("fetch", fetchMock);
    try {
      await expect(
        generateQuizV2Stream(
          { topic_ids: ["01-static"], count: 1, question_types: ["mcq"] },
          {},
        ),
      ).resolves.toMatchObject({ type: "done" });
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("questions stream warning events", () => {
  it("surfaces a warning through onWarning without aborting the stream", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      ndjsonResponse([
        TOPIC_RESOLUTION_WARNING,
        {
          type: "done",
          topic_id: "01-static",
          topic_title: "Static Topic",
          generated_count: 1,
          provider_used: "openai",
          model_used: "gpt-4o-mini",
          retries_used: 0,
          malformed_items_dropped: 0,
          degraded: false,
          requested_count: 1,
          returned_count: 1,
        },
      ]),
    );
    vi.stubGlobal("fetch", fetchMock);
    try {
      const onWarning = vi.fn();
      const done = await generateQuestionsV2Stream(
        { topic_id: "01-static", count: 1 },
        { onWarning },
      );

      expect(onWarning).toHaveBeenCalledTimes(1);
      expect(onWarning.mock.calls[0][0].code).toBe("topic_resolution_failed");
      expect(done.degraded).toBe(false);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
