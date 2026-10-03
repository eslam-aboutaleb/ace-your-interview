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
