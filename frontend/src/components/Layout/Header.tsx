import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  BookOpen,
  Settings,
  BarChart3,
  ListChecks,
  MessageSquare,
  CalendarDays,
  Clock3,
  FileText,
  Layers,
  LogOut,
  User,
  LogIn,
  Menu,
  X,
  Star,
} from "lucide-react";
import { useProgressStore } from "@/store/progressStore";
import { useAuthStore } from "@/store/authStore";

export default function Header() {
  const location = useLocation();
  const navigate = useNavigate();
  const totalProgress = useProgressStore((s) => s.totalProgress);
  const progress = totalProgress();
  const { logout, user, isAuthenticated } = useAuthStore();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  useEffect(() => {
    setIsMenuOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    if (!isMenuOpen) {
      return;
    }

    const handlePointerDown = (event: MouseEvent) => {
      if (
        menuRef.current &&
        !menuRef.current.contains(event.target as Node)
      ) {
        setIsMenuOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isMenuOpen]);

  const navItems = [
    {
      to: "/",
      label: "Dashboard",
      icon: <BarChart3 className="w-4 h-4" />,
      active: location.pathname === "/",
    },
    {
      to: "/topics",
      label: "Topics",
      icon: <BookOpen className="w-4 h-4" />,
      active: location.pathname.startsWith("/topics"),
    },
    ...(isAuthenticated
      ? [
          {
            to: "/quiz",
            label: "Quiz",
            icon: <ListChecks className="w-4 h-4" />,
            active: location.pathname.startsWith("/quiz"),
          },
          {
            to: "/review",
            label: "Review Queue",
            icon: <Clock3 className="w-4 h-4" />,
            active: location.pathname.startsWith("/review"),
          },
          {
            to: "/study-plan",
            label: "Study Plan",
            icon: <CalendarDays className="w-4 h-4" />,
            active: location.pathname.startsWith("/study-plan"),
          },
          {
            to: "/documents",
            label: "Documents",
            icon: <FileText className="w-4 h-4" />,
            active: location.pathname.startsWith("/documents"),
          },
          {
            to: "/decks",
            label: "Flashcards",
            icon: <Layers className="w-4 h-4" />,
            active: location.pathname.startsWith("/decks"),
          },
          {
            to: "/interview",
            label: "Interview",
            icon: <MessageSquare className="w-4 h-4" />,
            active: location.pathname.startsWith("/interview"),
          },
          {
            to: "/star-stories",
            label: "STAR Stories",
            icon: <Star className="w-4 h-4" />,
            active: location.pathname.startsWith("/star-stories"),
          },
          {
            to: "/user-settings",
            label: "User Settings",
            icon: <User className="w-4 h-4" />,
            active: location.pathname === "/user-settings",
          },
        ]
      : []),
    ...(user?.is_admin
      ? [
          {
            to: "/settings",
            label: "Settings",
            icon: <Settings className="w-4 h-4" />,
            active: location.pathname === "/settings",
          },
        ]
      : []),
  ];

  return (
    <header className="sticky top-0 z-50 bg-udemy-dark text-white shadow-lg">
      <div className="max-w-[1340px] mx-auto px-3 sm:px-4 md:px-6">
        <div className="h-16 flex items-center justify-between gap-3">
          <div ref={menuRef} className="relative flex items-center gap-2.5 min-w-0">
            <button
              type="button"
              aria-label={isMenuOpen ? "Close navigation menu" : "Open navigation menu"}
              aria-expanded={isMenuOpen}
              onClick={() => setIsMenuOpen((open) => !open)}
              className="flex h-10 w-10 items-center justify-center rounded-lg border border-white/10 bg-white/5 transition-colors hover:bg-white/10"
            >
              {isMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>

            <Link
              to="/"
              className="flex items-center gap-2.5 hover:opacity-90 transition-opacity min-w-0"
            >
              <span className="text-2xl shrink-0">😴</span>
              <span className="hidden sm:inline text-lg font-bold tracking-tight truncate">
                Too lazy for this{" "}
                <span className="text-udemy-purple-light">interview</span>
              </span>
            </Link>

            <AnimatePresence>
              {isMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -10, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -8, scale: 0.98 }}
                  transition={{ duration: 0.16, ease: "easeOut" }}
                  className="absolute left-0 top-full mt-2 w-[min(22rem,calc(100vw-1.5rem))] overflow-hidden rounded-2xl border border-white/10 bg-slate-900/95 shadow-2xl backdrop-blur"
                >
                  {user && (
                    <div className="border-b border-white/10 px-4 py-3">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-400">
                        Signed in
                      </div>
                      <div className="mt-1 truncate text-sm font-medium text-white">
                        {user.user}
                      </div>
                    </div>
                  )}

                  <nav className="p-2">
                    {navItems.map((item) => (
                      <MenuLink
                        key={item.to}
                        to={item.to}
                        label={item.label}
                        icon={item.icon}
                        active={item.active}
                      />
                    ))}
                  </nav>

                  <div className="border-t border-white/10 p-2">
                    {isAuthenticated ? (
                      <button
                        type="button"
                        onClick={handleLogout}
                        className="flex w-full items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium text-gray-300 transition-colors hover:bg-white/5 hover:text-white"
                      >
                        <LogOut className="h-4 w-4" />
                        Sign out
                      </button>
                    ) : (
                      <Link
                        to="/login"
                        className="flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium text-gray-300 transition-colors hover:bg-white/5 hover:text-white"
                      >
                        <LogIn className="h-4 w-4" />
                        Sign in
                      </Link>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {isAuthenticated && (
            <div className="flex items-center gap-3">
              <div className="relative hidden h-10 w-10 sm:block">
                <svg className="h-10 w-10 -rotate-90" viewBox="0 0 36 36">
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
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

function MenuLink({
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
      className={`flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium transition-colors
        ${
          active
            ? "bg-white/10 text-white"
            : "text-gray-300 hover:bg-white/5 hover:text-white"
        }`}
    >
      {icon}
      {label}
    </Link>
  );
}
