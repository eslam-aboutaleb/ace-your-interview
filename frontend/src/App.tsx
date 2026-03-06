import { Routes, Route } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import Layout from "@/components/Layout/Layout";
import Dashboard from "@/pages/Dashboard";
import TopicsList from "@/pages/TopicsList";
import TopicStudy from "@/pages/TopicStudy";
import QuizMode from "@/pages/QuizMode";
import Settings from "@/pages/Settings";

export default function App() {
  return (
    <AnimatePresence mode="wait">
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="topics" element={<TopicsList />} />
          <Route path="topics/:topicId" element={<TopicStudy />} />
          <Route path="quiz" element={<QuizMode />} />
          <Route path="quiz/:topicId" element={<QuizMode />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </AnimatePresence>
  );
}
