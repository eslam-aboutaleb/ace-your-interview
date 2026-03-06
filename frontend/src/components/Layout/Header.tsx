import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import {
  BookOpen,
  Settings,
  BarChart3,
  GraduationCap,
  ListChecks,
} from "lucide-react";
import { useProgressStore } from "@/store/progressStore";

export default function Header() {
  const location = useLocation();
  const totalProgress = useProgressStore((s) => s.totalProgress);
  const progress = totalProgress();

  return (
    <header className="sticky top-0 z-50 bg-udemy-dark text-white shadow-lg">
      <div className="max-w-[1340px] mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link
          to="/"
          className="flex items-center gap-2.5 hover:opacity-90 transition-opacity"
        >
          <GraduationCap className="w-8 h-8 text-udemy-purple-light" />
          <span className="text-lg font-bold tracking-tight">
            Polymarket{" "}
            <span className="text-udemy-purple-light">Study Hub</span>
          </span>
        </Link>

        {/* Nav links */}
        <nav className="hidden md:flex items-center gap-1">
          <NavLink
            to="/"
            label="Dashboard"
            icon={<BarChart3 className="w-4 h-4" />}
            active={location.pathname === "/"}
          />
          <NavLink
            to="/topics"
            label="Topics"
            icon={<BookOpen className="w-4 h-4" />}
            active={location.pathname.startsWith("/topics")}
          />
          <NavLink
            to="/quiz"
            label="Quiz"
            icon={<ListChecks className="w-4 h-4" />}
            active={location.pathname.startsWith("/quiz")}
          />
          <NavLink
            to="/settings"
            label="Settings"
            icon={<Settings className="w-4 h-4" />}
            active={location.pathname === "/settings"}
          />
        </nav>

        {/* Progress ring */}
        <div className="flex items-center gap-3">
          <div className="relative w-10 h-10">
            <svg className="w-10 h-10 -rotate-90" viewBox="0 0 36 36">
              <circle
                cx="18"
                cy="18"
                r="15.5"
                fill="none"
                stroke="#3e4143"
                strokeWidth="2.5"
              />
              <motion.circle
                cx="18"
                cy="18"
                r="15.5"
                fill="none"
                stroke="#a435f0"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeDasharray="97.39"
                initial={{ strokeDashoffset: 97.39 }}
                animate={{ strokeDashoffset: 97.39 - (97.39 * progress) / 100 }}
                transition={{ type: "spring", stiffness: 100, damping: 15 }}
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold">
              {progress}%
            </span>
          </div>

          {/* Mobile menu */}
          <div className="md:hidden flex items-center gap-2">
            <Link to="/" className="p-2 hover:bg-white/10 rounded">
              <BarChart3 className="w-5 h-5" />
            </Link>
            <Link to="/topics" className="p-2 hover:bg-white/10 rounded">
              <BookOpen className="w-5 h-5" />
            </Link>
            <Link to="/quiz" className="p-2 hover:bg-white/10 rounded">
              <ListChecks className="w-5 h-5" />
            </Link>
            <Link to="/settings" className="p-2 hover:bg-white/10 rounded">
              <Settings className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function NavLink({
  to,
  label,
  icon,
  active,
}: {
  to: string;
  label: string;
  icon: React.ReactNode;
  active: boolean;
}) {
  return (
    <Link
      to={to}
      className={`flex items-center gap-1.5 px-3 py-2 rounded text-sm font-medium transition-colors
        ${active ? "bg-white/10 text-white" : "text-gray-300 hover:text-white hover:bg-white/5"}`}
    >
      {icon}
      {label}
    </Link>
  );
}
