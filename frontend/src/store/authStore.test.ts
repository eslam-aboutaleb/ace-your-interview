import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAuthStore } from "./authStore";

function mockFetchResponse(ok: boolean, payload: unknown = {}) {
  return vi.fn(() =>
    Promise.resolve({
      ok,
      json: () => Promise.resolve(payload),
    }),
  );
}

describe("authStore", () => {
  beforeEach(() => {
    useAuthStore.setState({
      user: null,
      isAuthenticated: false,
      isLoading: true,
    });
  });

  it("checkAuth stores the authenticated user", async () => {
    const user = {
      user: "learner@example.com",
      provider: "google",
      is_admin: false,
    };
    vi.stubGlobal("fetch", mockFetchResponse(true, user));
    try {
      await useAuthStore.getState().checkAuth();
      expect(useAuthStore.getState().isAuthenticated).toBe(true);
      expect(useAuthStore.getState().user).toEqual(user);
      expect(useAuthStore.getState().isLoading).toBe(false);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("checkAuth clears the session when the server rejects it", async () => {
    vi.stubGlobal("fetch", mockFetchResponse(false));
    try {
      await useAuthStore.getState().checkAuth();
      expect(useAuthStore.getState().isAuthenticated).toBe(false);
      expect(useAuthStore.getState().user).toBeNull();
      expect(useAuthStore.getState().isLoading).toBe(false);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("checkAuth treats a network failure as unauthenticated", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new Error("offline"))),
    );
    try {
      await useAuthStore.getState().checkAuth();
      expect(useAuthStore.getState().isAuthenticated).toBe(false);
      expect(useAuthStore.getState().isLoading).toBe(false);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("logout clears the session even when the server call fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new Error("offline"))),
    );
    try {
      useAuthStore.setState({
        user: {
          user: "learner@example.com",
          provider: "google",
          is_admin: false,
        },
        isAuthenticated: true,
        isLoading: false,
      });
      await useAuthStore.getState().logout();
      expect(useAuthStore.getState().isAuthenticated).toBe(false);
      expect(useAuthStore.getState().user).toBeNull();
      expect(useAuthStore.getState().isLoading).toBe(false);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
