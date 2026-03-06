import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Settings as SettingsIcon,
  RefreshCw,
  Loader2,
  ShieldCheck,
  Github,
  Mail,
  Plus,
  Trash2,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import {
  addLLMServiceUser,
  addAllowedUser,
  fetchAllowedUsers,
  fetchHealth,
  fetchLLMServiceUsers,
  removeLLMServiceUser,
  removeAllowedUser,
} from "@/services/api";
import type { AllowedUsers, LLMServiceUsers } from "@/services/api";
import type { HealthStatus } from "@/types";
import { useProgressStore } from "@/store/progressStore";

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loadingHealth, setLoadingHealth] = useState(false);

  const [allowedUsers, setAllowedUsers] = useState<AllowedUsers | null>(null);
  const [newGithubUser, setNewGithubUser] = useState("");
  const [newGoogleEmail, setNewGoogleEmail] = useState("");
  const [loadingAllowed, setLoadingAllowed] = useState(false);
  const [llmServiceUsers, setLlmServiceUsers] = useState<LLMServiceUsers | null>(null);
  const [newLlmGithubUser, setNewLlmGithubUser] = useState("");
  const [newLlmGoogleEmail, setNewLlmGoogleEmail] = useState("");
  const [loadingLlmService, setLoadingLlmService] = useState(false);
  const [error, setError] = useState("");

  const resetProgress = useProgressStore((s) => s.reset);
  const llmGithubEntries = (llmServiceUsers?.users || []).flatMap((raw) => {
    const token = raw.trim();
    if (!token) return [];
    if (token.includes(":")) {
      const [provider, ...rest] = token.split(":");
      if (provider.toLowerCase() !== "github") return [];
      return [
        {
          display: token,
          identifier: rest.join(":"),
        },
      ];
    }
    if (token.includes("@")) return [];
    return [{ display: token, identifier: token }];
  });
  const llmGoogleEntries = (llmServiceUsers?.users || []).flatMap((raw) => {
    const token = raw.trim();
    if (!token) return [];
    if (token.includes(":")) {
      const [provider, ...rest] = token.split(":");
      if (provider.toLowerCase() !== "google") return [];
      return [
        {
          display: token,
          identifier: rest.join(":"),
        },
      ];
    }
    if (!token.includes("@")) return [];
    return [{ display: token, identifier: token }];
  });

  const loadAllowedUsers = async () => {
    setError("");
    try {
      const data = await fetchAllowedUsers();
      setAllowedUsers(data);
    } catch (e) {
      console.error(e);
      setError("Failed to load allowed users.");
    }
  };

  const loadLlmServiceUsers = async () => {
    setError("");
    try {
      const data = await fetchLLMServiceUsers();
      setLlmServiceUsers(data);
    } catch (e) {
      console.error(e);
      setError("Failed to load LLM service approved users.");
    }
  };

  useEffect(() => {
    loadAllowedUsers();
    loadLlmServiceUsers();
  }, []);

  const checkHealth = async () => {
    setLoadingHealth(true);
    try {
      const res = await fetchHealth();
      setHealth(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingHealth(false);
    }
  };

  const handleAddUser = async (provider: "github" | "google") => {
    const identifier =
      provider === "github" ? newGithubUser.trim() : newGoogleEmail.trim();
    if (!identifier) return;

    setLoadingAllowed(true);
    setError("");
    try {
      const data = await addAllowedUser(provider, identifier);
      setAllowedUsers(data);
      if (provider === "github") setNewGithubUser("");
      else setNewGoogleEmail("");
    } catch (e) {
      console.error(e);
      setError("Failed to add allowed user.");
    } finally {
      setLoadingAllowed(false);
    }
  };

  const handleRemoveUser = async (
    provider: "github" | "google",
    identifier: string,
  ) => {
    if (!confirm(`Remove ${identifier}?`)) return;

    setLoadingAllowed(true);
    setError("");
    try {
      const data = await removeAllowedUser(provider, identifier);
      setAllowedUsers(data);
    } catch (e) {
      console.error(e);
      setError("Failed to remove allowed user.");
    } finally {
      setLoadingAllowed(false);
    }
  };

  const handleAddLlmServiceUser = async (provider: "github" | "google") => {
    const identifier =
      provider === "github" ? newLlmGithubUser.trim() : newLlmGoogleEmail.trim();
    if (!identifier) return;

    setLoadingLlmService(true);
    setError("");
    try {
      const data = await addLLMServiceUser(provider, identifier);
      setLlmServiceUsers(data);
      if (provider === "github") setNewLlmGithubUser("");
      else setNewLlmGoogleEmail("");
    } catch (e) {
      console.error(e);
      setError("Failed to add LLM service approved user.");
    } finally {
      setLoadingLlmService(false);
    }
  };

  const handleRemoveLlmServiceUser = async (
    provider: "github" | "google",
    identifier: string,
  ) => {
    if (!confirm(`Remove ${identifier}?`)) return;

    setLoadingLlmService(true);
    setError("");
    try {
      const data = await removeLLMServiceUser(provider, identifier);
      setLlmServiceUsers(data);
    } catch (e) {
      console.error(e);
      setError("Failed to remove LLM service approved user.");
    } finally {
      setLoadingLlmService(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-[1200px] mx-auto px-6 py-8">
      <div className="udemy-card p-6 mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <SettingsIcon className="w-6 h-6 text-udemy-purple" />
          Admin Settings
        </h1>
        <p className="text-sm text-udemy-text-muted mt-2">
          Platform-level controls and access management.
        </p>
        {error && <p className="text-sm text-red-600 mt-3">{error}</p>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="udemy-card p-6 lg:col-span-2">
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-udemy-purple" />
            Allowed Users
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-bold flex items-center gap-2 mb-3">
                <Github className="w-4 h-4" />
                GitHub Usernames
              </h3>
              <div className="space-y-2 mb-3">
                {allowedUsers?.github_users.map((u) => (
                  <div key={u} className="flex items-center justify-between bg-udemy-bg rounded px-3 py-2 text-sm">
                    <span className="font-medium">{u}</span>
                    <button
                      onClick={() => handleRemoveUser("github", u)}
                      disabled={loadingAllowed}
                      className="text-red-400 hover:text-red-600 transition-colors p-1"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
                {allowedUsers?.github_users.length === 0 && (
                  <p className="text-xs text-udemy-text-muted italic">No GitHub users allowed</p>
                )}
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="github-username"
                  value={newGithubUser}
                  onChange={(e) => setNewGithubUser(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAddUser("github")}
                  className="flex-1 border border-udemy-border rounded px-3 py-2 text-sm"
                />
                <button
                  onClick={() => handleAddUser("github")}
                  disabled={loadingAllowed || !newGithubUser.trim()}
                  className="btn-secondary text-xs flex items-center gap-1 px-3 py-2"
                >
                  <Plus className="w-3.5 h-3.5" />
                  Add
                </button>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-bold flex items-center gap-2 mb-3">
                <Mail className="w-4 h-4" />
                Google Emails
              </h3>
              <div className="space-y-2 mb-3">
                {allowedUsers?.google_emails.map((e) => (
                  <div key={e} className="flex items-center justify-between bg-udemy-bg rounded px-3 py-2 text-sm">
                    <span className="font-medium">{e}</span>
                    <button
                      onClick={() => handleRemoveUser("google", e)}
                      disabled={loadingAllowed}
                      className="text-red-400 hover:text-red-600 transition-colors p-1"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
                {allowedUsers?.google_emails.length === 0 && (
                  <p className="text-xs text-udemy-text-muted italic">No Google emails allowed</p>
                )}
              </div>
              <div className="flex gap-2">
                <input
                  type="email"
                  placeholder="user@gmail.com"
                  value={newGoogleEmail}
                  onChange={(e) => setNewGoogleEmail(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAddUser("google")}
                  className="flex-1 border border-udemy-border rounded px-3 py-2 text-sm"
                />
                <button
                  onClick={() => handleAddUser("google")}
                  disabled={loadingAllowed || !newGoogleEmail.trim()}
                  className="btn-secondary text-xs flex items-center gap-1 px-3 py-2"
                >
                  <Plus className="w-3.5 h-3.5" />
                  Add
                </button>
              </div>
            </div>
          </div>

          <div className="border-t border-udemy-border mt-6 pt-6">
            <h2 className="text-lg font-bold mb-2 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-udemy-purple" />
              LLM Service Approved Users
            </h2>
            <p className="text-xs text-udemy-text-muted mb-4">
              Approved users can use backend-funded LLM fallback. Non-approved users must connect their own credentials.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-sm font-bold flex items-center gap-2 mb-3">
                  <Github className="w-4 h-4" />
                  GitHub Usernames
                </h3>
                <div className="space-y-2 mb-3 max-h-44 overflow-auto pr-1">
                  {llmGithubEntries.map((entry) => (
                      <div key={entry.display} className="flex items-center justify-between bg-udemy-bg rounded px-3 py-2 text-sm">
                        <span className="font-medium">{entry.display}</span>
                        <button
                          onClick={() =>
                            handleRemoveLlmServiceUser(
                              "github",
                              entry.identifier,
                            )
                          }
                          disabled={loadingLlmService}
                          className="text-red-400 hover:text-red-600 transition-colors p-1"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ))}
                  {llmGithubEntries.length === 0 && (
                    <p className="text-xs text-udemy-text-muted italic">No approved GitHub users</p>
                  )}
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="github-username"
                    value={newLlmGithubUser}
                    onChange={(e) => setNewLlmGithubUser(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleAddLlmServiceUser("github")}
                    className="flex-1 border border-udemy-border rounded px-3 py-2 text-sm"
                  />
                  <button
                    onClick={() => handleAddLlmServiceUser("github")}
                    disabled={loadingLlmService || !newLlmGithubUser.trim()}
                    className="btn-secondary text-xs flex items-center gap-1 px-3 py-2"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    Add
                  </button>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-bold flex items-center gap-2 mb-3">
                  <Mail className="w-4 h-4" />
                  Google Emails
                </h3>
                <div className="space-y-2 mb-3 max-h-44 overflow-auto pr-1">
                  {llmGoogleEntries.map((entry) => (
                      <div key={entry.display} className="flex items-center justify-between bg-udemy-bg rounded px-3 py-2 text-sm">
                        <span className="font-medium">{entry.display}</span>
                        <button
                          onClick={() =>
                            handleRemoveLlmServiceUser(
                              "google",
                              entry.identifier,
                            )
                          }
                          disabled={loadingLlmService}
                          className="text-red-400 hover:text-red-600 transition-colors p-1"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ))}
                  {llmGoogleEntries.length === 0 && (
                    <p className="text-xs text-udemy-text-muted italic">No approved Google users</p>
                  )}
                </div>
                <div className="flex gap-2">
                  <input
                    type="email"
                    placeholder="user@gmail.com"
                    value={newLlmGoogleEmail}
                    onChange={(e) => setNewLlmGoogleEmail(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleAddLlmServiceUser("google")}
                    className="flex-1 border border-udemy-border rounded px-3 py-2 text-sm"
                  />
                  <button
                    onClick={() => handleAddLlmServiceUser("google")}
                    disabled={loadingLlmService || !newLlmGoogleEmail.trim()}
                    className="btn-secondary text-xs flex items-center gap-1 px-3 py-2"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    Add
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="udemy-card p-6">
          <h2 className="text-lg font-bold mb-4">Service Health</h2>
          <button
            onClick={checkHealth}
            disabled={loadingHealth}
            className="btn-secondary w-full flex items-center justify-center gap-2"
          >
            {loadingHealth ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
            Check Health
          </button>

          {health && (
            <div className="mt-4 space-y-3">
              <StatusRow
                label="LLM Chain"
                ok={health.llm_chain}
                version={health.llm_chain_version}
              />
              <StatusRow
                label="CLI Agent"
                ok={health.cli_agent}
                version={health.cli_agent_version}
              />
            </div>
          )}

          <div className="border-t border-udemy-border pt-4 mt-6">
            <h3 className="text-sm font-bold text-udemy-danger uppercase mb-3">Danger Zone</h3>
            <button
              onClick={() => {
                if (confirm("Reset all progress? This cannot be undone.")) {
                  resetProgress();
                }
              }}
              className="w-full border-2 border-udemy-danger text-udemy-danger font-bold py-2 px-4 rounded hover:bg-red-50 transition-colors text-sm"
            >
              Reset All Progress
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function StatusRow({
  label,
  ok,
  version,
}: {
  label: string;
  ok: boolean;
  version: string;
}) {
  return (
    <div className="flex items-center justify-between p-3 bg-udemy-bg rounded-lg">
      <div className="flex items-center gap-2">
        {ok ? (
          <CheckCircle2 className="w-4 h-4 text-green-600" />
        ) : (
          <XCircle className="w-4 h-4 text-red-500" />
        )}
        <span className="text-sm font-medium">{label}</span>
      </div>
      <span className="text-xs text-udemy-text-muted">
        {ok ? version || "healthy" : "offline"}
      </span>
    </div>
  );
}
