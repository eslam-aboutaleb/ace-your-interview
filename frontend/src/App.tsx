import { Routes, Route, Navigate, Outlet } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import { useEffect, useRef } from "react";
import { Loader2 } from "lucide-react";
import Layout from "@/components/Layout/Layout";
import Dashboard from "@/pages/Dashboard";
import TopicsList from "@/pages/TopicsList";
import CustomTopicBuild from "@/pages/CustomTopicBuild";
import TopicStudy from "@/pages/TopicStudy";
import QuizMode from "@/pages/QuizMode";
import ReviewQueue from "@/pages/ReviewQueue";
import StudyPlan from "@/pages/StudyPlan";
import DocumentsPage from "@/pages/DocumentsPage";
import Settings from "@/pages/Settings";
import UserSettings from "@/pages/UserSettings";
import LoginPage from "@/pages/LoginPage";
import InterviewSetup from "@/pages/InterviewSetup";
import InterviewSessionPage from "@/pages/InterviewSessionPage";
import InterviewReportPage from "@/pages/InterviewReportPage";
import InterviewTrendsPage from "@/pages/InterviewTrends";
import StarStoriesPage from "@/pages/StarStoriesPage";
import { fetchUserSettings, updateUserPreferences } from "@/services/api";
import { useAuthStore } from "@/store/authStore";
import { useSettingsStore } from "@/store/settingsStore";
import type { UserPreferences } from "@/types";

function PublicRoute() {
  const { isLoading, checkAuth } = useAuthStore();

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-udemy-bg flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-udemy-purple animate-spin" />
      </div>
    );
  }

  return <Outlet />;
}

export function ProtectedRoute() {
  const { isAuthenticated, isLoading, checkAuth } = useAuthStore();
  const hydratedRef = useRef(false);
  const hydrateFromServer = useSettingsStore((s) => s.hydrateFromServer);
  const localLLMSource = useSettingsStore((s) => s.llmSource);
  const localProvider = useSettingsStore((s) => s.provider);
  const localModel = useSettingsStore((s) => s.model);
  const localTemperature = useSettingsStore((s) => s.temperature);
  const localMaxTokens = useSettingsStore((s) => s.maxTokens);
  const localRequireAnswerReveal = useSettingsStore(
    (s) => s.requireAnswerReveal,
  );

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  useEffect(() => {
    if (!isAuthenticated) {
      hydratedRef.current = false;
      return;
    }
    if (isLoading || hydratedRef.current) return;

    let active = true;
    (async () => {
      try {
        const res = await fetchUserSettings();
        const allowedProviders = new Set(res.providers.map((p) => p.provider));

        let prefs = res.preferences;

        if (!res.has_saved_preferences) {
          const provider = allowedProviders.has(
            localProvider as UserPreferences["provider"],
          )
            ? (localProvider as UserPreferences["provider"])
            : res.preferences.provider;
          const modelOptions =
            res.providers.find((p) => p.provider === provider)?.models || [];
          const model =
            localModel && modelOptions.includes(localModel) ? localModel : "";
          const seedPayload: UserPreferences = {
            provider,
            model,
            temperature: localTemperature,
            max_tokens: localMaxTokens,
            auth_mode: provider === "google" ? "api_key" : "api_key",
            llm_source: localLLMSource,
            require_answer_reveal: localRequireAnswerReveal,
          };
          prefs = await updateUserPreferences(seedPayload);
        }

        if (!active) return;
        hydrateFromServer({
          llmSource: prefs.llm_source,
          provider: prefs.provider,
          model: prefs.model,
          temperature: prefs.temperature,
          maxTokens: prefs.max_tokens,
          requireAnswerReveal: prefs.require_answer_reveal,
        });
      } catch (e) {
        console.error(e);
      } finally {
        if (active) hydratedRef.current = true;
      }
    })();

    return () => {
      active = false;
    };
  }, [
    hydrateFromServer,
    isAuthenticated,
    isLoading,
    localProvider,
    localLLMSource,
    localModel,
    localTemperature,
    localMaxTokens,
    localRequireAnswerReveal,
  ]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-udemy-bg flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-udemy-purple animate-spin" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <Outlet />;
}

function AdminRoute({ children }: { children: React.ReactNode }) {
  const user = useAuthStore((s) => s.user);
  if (!user?.is_admin) {
    return <Navigate to="/user-settings" replace />;
  }
  return <>{children}</>;
}

export default function App() {
  return (
    <AnimatePresence mode="wait">
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<PublicRoute />}>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="topics" element={<TopicsList />} />
            <Route path="topics/:topicId" element={<TopicStudy />} />
          </Route>
        </Route>
        <Route element={<ProtectedRoute />}>
          <Route path="/" element={<Layout />}>
            <Route path="topics/custom/build" element={<CustomTopicBuild />} />
            <Route path="quiz" element={<QuizMode />} />
            <Route path="quiz/:topicId" element={<QuizMode />} />
            <Route path="review" element={<ReviewQueue />} />
            <Route path="study-plan" element={<StudyPlan />} />
            <Route path="documents" element={<DocumentsPage />} />
            <Route path="interview" element={<InterviewSetup />} />
            <Route path="interview/trends" element={<InterviewTrendsPage />} />
            <Route
              path="interview/:sessionId"
              element={<InterviewSessionPage />}
            />
            <Route
              path="interview/:sessionId/report"
              element={<InterviewReportPage />}
            />
            <Route path="star-stories" element={<StarStoriesPage />} />
            <Route path="user-settings" element={<UserSettings />} />
            <Route
              path="settings"
              element={
                <AdminRoute>
                  <Settings />
                </AdminRoute>
              }
            />
          </Route>
        </Route>
      </Routes>
    </AnimatePresence>
  );
}
