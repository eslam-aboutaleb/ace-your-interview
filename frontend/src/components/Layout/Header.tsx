import { Link, useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  BookOpen,
  Settings,
  BarChart3,
  ListChecks,
  MessageSquare,
  CalendarDays,
  Clock3,
  LogOut,
  User,
} from "lucide-react";
import { useProgressStore } from "@/store/progressStore";
import { useAuthStore } from "@/store/authStore";

export default function Header() {
  const location = useLocation();
  const navigate = useNavigate();
  const totalProgress = useProgressStore((s) => s.totalProgress);
  const progress = totalProgress();
  const { logout, user, isAuthenticated } = useAuthStore();

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <header className="sticky top-0 z-50 bg-udemy-dark text-white shadow-lg">
      <div className="max-w-[1340px] mx-auto px-3 sm:px-4 md:px-6">
        <div className="h-16 flex items-center justify-between">
          <Link
            to="/"
            className="flex items-center gap-2.5 hover:opacity-90 transition-opacity"
          >
            <span className="text-2xl">😴</span>
            <span className="hidden sm:inline text-lg font-bold tracking-tight">
              Too lazy for this{" "}
              <span className="text-udemy-purple-light">interview</span>
            </span>
          </Link>

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
            {isAuthenticated && (
              <>
                <NavLink
                  to="/quiz"
                  label="Quiz"
                  icon={<ListChecks className="w-4 h-4" />}
                  active={location.pathname.startsWith("/quiz")}
                />
                <NavLink
                  to="/review"
                  label="Review Queue"
                  icon={<Clock3 className="w-4 h-4" />}
                  active={location.pathname.startsWith("/review")}
                />
                <NavLink
                  to="/study-plan"
                  label="Study Plan"
                  icon={<CalendarDays className="w-4 h-4" />}
                  active={location.pathname.startsWith("/study-plan")}
                />
                <NavLink
                  to="/interview"
                  label="Interview"
                  icon={<MessageSquare className="w-4 h-4" />}
                  active={location.pathname.startsWith("/interview")}
                />
                <NavLink
                  to="/user-settings"
                  label="User Settings"
                  icon={<User className="w-4 h-4" />}
                  active={location.pathname === "/user-settings"}
                />
              </>
            )}
            {user?.is_admin && (
              <NavLink
                to="/settings"
                label="Settings"
                icon={<Settings className="w-4 h-4" />}
                active={location.pathname === "/settings"}
              />
            )}
          </nav>

          <div className="flex items-center gap-1.5 sm:gap-3">
            {isAuthenticated ? (
              <>
                {user && (
                  <span className="hidden md:inline text-xs text-gray-400 truncate max-w-[140px]">
                    {user.user}
                  </span>
                )}

                <div className="relative w-10 h-10 hidden sm:block">
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
                      stroke="#2563eb"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeDasharray="97.39"
                      initial={{ strokeDashoffset: 97.39 }}
                      animate={{
                        strokeDashoffset: 97.39 - (97.39 * progress) / 100,
                      }}
                      transition={{
                        type: "spring",
                        stiffness: 100,
                        damping: 15,
                      }}
                    />
                  </svg>
                  <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold">
                    {progress}%
                  </span>
                </div>

                <button
                  onClick={handleLogout}
                  title="Sign out"
                  className="p-1.5 sm:p-2 hover:bg-white/10 rounded transition-colors"
                >
                  <LogOut className="w-4 h-4 sm:w-5 sm:h-5 text-gray-400 hover:text-white" />
                </button>
              </>
            ) : (
              <Link
                to="/login"
                className="px-4 py-2 bg-udemy-purple hover:bg-udemy-purple-dark text-white text-sm font-medium rounded transition-colors"
              >
                Sign in
              </Link>
            )}
          </div>
        </div>

        <nav className="md:hidden pb-2 flex items-center gap-1 overflow-x-auto">
          <Link to="/" className="p-1.5 sm:p-2 hover:bg-white/10 rounded">
            <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5" />
          </Link>
          <Link to="/topics" className="p-1.5 sm:p-2 hover:bg-white/10 rounded">
            <BookOpen className="w-4 h-4 sm:w-5 sm:h-5" />
          </Link>
          {isAuthenticated && (
            <>
              <Link
                to="/quiz"
                className="p-1.5 sm:p-2 hover:bg-white/10 rounded"
              >
                <ListChecks className="w-4 h-4 sm:w-5 sm:h-5" />
              </Link>
              <Link
                to="/review"
                className="p-1.5 sm:p-2 hover:bg-white/10 rounded"
              >
                <Clock3 className="w-4 h-4 sm:w-5 sm:h-5" />
              </Link>
              <Link
                to="/study-plan"
                className="p-1.5 sm:p-2 hover:bg-white/10 rounded"
              >
                <CalendarDays className="w-4 h-4 sm:w-5 sm:h-5" />
              </Link>
              <Link
                to="/interview"
                className="p-1.5 sm:p-2 hover:bg-white/10 rounded"
              >
                <MessageSquare className="w-4 h-4 sm:w-5 sm:h-5" />
              </Link>
              <Link
                to="/user-settings"
                className="p-1.5 sm:p-2 hover:bg-white/10 rounded"
              >
                <User className="w-4 h-4 sm:w-5 sm:h-5" />
              </Link>
            </>
          )}
          {user?.is_admin && (
            <Link
              to="/settings"
              className="p-1.5 sm:p-2 hover:bg-white/10 rounded"
            >
              <Settings className="w-4 h-4 sm:w-5 sm:h-5" />
            </Link>
          )}
        </nav>
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
