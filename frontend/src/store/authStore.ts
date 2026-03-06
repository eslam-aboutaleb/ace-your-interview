import { create } from "zustand";

interface AuthUser {
  user: string;
  provider: string;
  is_admin: boolean;
}

interface AuthState {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  checkAuth: () => Promise<void>;
  logout: () => Promise<void>;
}

const apiBase = import.meta.env.VITE_API_URL || "/api";

export const useAuthStore = create<AuthState>()((set) => ({
  user: null,
  isAuthenticated: false,
  isLoading: true,

  checkAuth: async () => {
    try {
      const res = await fetch(`${apiBase}/auth/me`, {
        credentials: "include",
      });
      if (res.ok) {
        const data: AuthUser = await res.json();
        set({ user: data, isAuthenticated: true, isLoading: false });
      } else {
        set({ user: null, isAuthenticated: false, isLoading: false });
      }
    } catch {
      set({ user: null, isAuthenticated: false, isLoading: false });
    }
  },

  logout: async () => {
    try {
      await fetch(`${apiBase}/auth/logout`, {
        method: "POST",
        credentials: "include",
      });
    } catch {
      /* swallow */
    }
    set({ user: null, isAuthenticated: false, isLoading: false });
  },
}));
