import { Routes, Route, Navigate, Outlet } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import { useEffect } from "react";
import { Loader2 } from "lucide-react";
import Layout from "@/components/Layout/Layout";
import Dashboard from "@/pages/Dashboard";
import TopicsList from "@/pages/TopicsList";
import TopicStudy from "@/pages/TopicStudy";
import QuizMode from "@/pages/QuizMode";
import Settings from "@/pages/Settings";
import LoginPage from "@/pages/LoginPage";
import { useAuthStore } from "@/store/authStore";

function ProtectedRoute() {
  const { isAuthenticated, isLoading, checkAuth } = useAuthStore();

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

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <Outlet />;
}

export default function App() {
  return (
    <AnimatePresence mode="wait">
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<ProtectedRoute />}>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="topics" element={<TopicsList />} />
            <Route path="topics/:topicId" element={<TopicStudy />} />
            <Route path="quiz" element={<QuizMode />} />
            <Route path="quiz/:topicId" element={<QuizMode />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Route>
      </Routes>
    </AnimatePresence>
  );
}
