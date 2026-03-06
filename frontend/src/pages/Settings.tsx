import { useEffect, useMemo, useState } from "react";
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
  Save,
  User,
} from "lucide-react";
import {
  addLLMServiceUser,
  addAllowedUser,
  deleteLLMAssignmentForUser,
  fetchAllowedUsers,
  fetchHealth,
  fetchLLMAssignmentUsers,
  fetchLLMServiceUsers,
  fetchMyLLMAssignment,
  removeLLMServiceUser,
  removeAllowedUser,
  saveLLMAssignmentForUser,
  saveMyLLMAssignment,
} from "@/services/api";
import type { AllowedUsers, LLMServiceUsers } from "@/services/api";
import type {
  HealthStatus,
  LLMAssignmentUserItem,
  LLMMyAssignmentResponse,
  UserSettingsProvider,
} from "@/types";
import { useProgressStore } from "@/store/progressStore";

type AssignmentDraft = {
  provider: UserSettingsProvider;
  model: string;
};

function firstModel(
  modelsMap: Record<string, string[]>,
  provider: string,
): string {
  const models = modelsMap[provider] || [];
  return models[0] || "";
}

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

  const [assignmentUsers, setAssignmentUsers] = useState<LLMAssignmentUserItem[]>([]);
  const [assignmentProviderModels, setAssignmentProviderModels] = useState<Record<string, string[]>>({});
  const [assignmentDrafts, setAssignmentDrafts] = useState<Record<string, AssignmentDraft>>({});
  const [savingAssignments, setSavingAssignments] = useState<Record<string, boolean>>({});
  const [myAssignment, setMyAssignment] = useState<LLMMyAssignmentResponse | null>(null);
  const [myDraft, setMyDraft] = useState<AssignmentDraft | null>(null);
  const [savingMyAssignment, setSavingMyAssignment] = useState(false);

  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const resetProgress = useProgressStore((s) => s.reset);

  const llmGithubEntries = (llmServiceUsers?.users || []).flatMap((raw) => {
    const token = raw.trim();
    if (!token) return [];
    if (token.includes(":")) {
      const [provider, ...rest] = token.split(":");
      if (provider.toLowerCase() !== "github") return [];
      return [{ display: token, identifier: rest.join(":") }];
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
      return [{ display: token, identifier: rest.join(":") }];
    }
    if (!token.includes("@")) return [];
    return [{ display: token, identifier: token }];
  });

  const providerOptions = useMemo(
    () => Object.keys(assignmentProviderModels) as UserSettingsProvider[],
    [assignmentProviderModels],
  );

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

  const loadAssignmentUsers = async () => {
    setError("");
    try {
      const data = await fetchLLMAssignmentUsers();
      setAssignmentUsers(data.users);
      setAssignmentProviderModels(data.provider_models || {});
      const drafts: Record<string, AssignmentDraft> = {};
      for (const user of data.users) {
        const provider = (user.assignment?.provider || providerOptions[0] || "openai") as UserSettingsProvider;
        const model = user.assignment?.model || firstModel(data.provider_models, provider);
        drafts[user.identity_key] = { provider, model };
      }
      setAssignmentDrafts(drafts);
    } catch (e) {
      console.error(e);
      setError("Failed to load Study App LLM assignments.");
    }
  };

  const loadMyAssignment = async () => {
    setError("");
    try {
      const data = await fetchMyLLMAssignment();
      setMyAssignment(data);
      const provider = (data.assignment?.provider || "openai") as UserSettingsProvider;
      const model = data.assignment?.model || firstModel(data.provider_models || {}, provider);
      setMyDraft({ provider, model });
      if (!Object.keys(assignmentProviderModels).length && data.provider_models) {
        setAssignmentProviderModels(data.provider_models);
      }
    } catch (e) {
      console.error(e);
      setError("Failed to load your Study App assignment.");
    }
  };

  useEffect(() => {
    loadAllowedUsers();
    loadLlmServiceUsers();
    loadAssignmentUsers();
    loadMyAssignment();
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
    const identifier = provider === "github" ? newGithubUser.trim() : newGoogleEmail.trim();
    if (!identifier) return;

    setLoadingAllowed(true);
    setError("");
    try {
      const data = await addAllowedUser(provider, identifier);
      setAllowedUsers(data);
      setMessage("Allowed user added.");
      if (provider === "github") setNewGithubUser("");
      else setNewGoogleEmail("");
      await loadAssignmentUsers();
    } catch (e) {
      console.error(e);
      setError("Failed to add allowed user.");
    } finally {
      setLoadingAllowed(false);
    }
  };

  const handleRemoveUser = async (provider: "github" | "google", identifier: string) => {
    if (!confirm(`Remove ${identifier}?`)) return;
    setLoadingAllowed(true);
    setError("");
    try {
      const data = await removeAllowedUser(provider, identifier);
      setAllowedUsers(data);
      setMessage("Allowed user removed.");
      await loadAssignmentUsers();
    } catch (e) {
      console.error(e);
      setError("Failed to remove allowed user.");
    } finally {
      setLoadingAllowed(false);
    }
  };

  const handleAddLlmServiceUser = async (provider: "github" | "google") => {
    const identifier = provider === "github" ? newLlmGithubUser.trim() : newLlmGoogleEmail.trim();
    if (!identifier) return;

    setLoadingLlmService(true);
    setError("");
    try {
      const data = await addLLMServiceUser(provider, identifier);
      setLlmServiceUsers(data);
      setMessage("LLM service user approved.");
      if (provider === "github") setNewLlmGithubUser("");
      else setNewLlmGoogleEmail("");
      await loadAssignmentUsers();
      await loadMyAssignment();
    } catch (e) {
      console.error(e);
      setError("Failed to add LLM service approved user.");
    } finally {
      setLoadingLlmService(false);
    }
  };

  const handleRemoveLlmServiceUser = async (provider: "github" | "google", identifier: string) => {
    if (!confirm(`Remove ${identifier}?`)) return;

    setLoadingLlmService(true);
    setError("");
    try {
      const data = await removeLLMServiceUser(provider, identifier);
      setLlmServiceUsers(data);
      setMessage("LLM service approval removed.");
      await loadAssignmentUsers();
      await loadMyAssignment();
    } catch (e) {
      console.error(e);
      setError("Failed to remove LLM service approved user.");
    } finally {
      setLoadingLlmService(false);
    }
  };

  const updateDraftProvider = (identityKey: string, provider: UserSettingsProvider) => {
    setAssignmentDrafts((prev) => ({
      ...prev,
      [identityKey]: {
        provider,
        model: firstModel(assignmentProviderModels, provider),
      },
    }));
  };

  const updateDraftModel = (identityKey: string, model: string) => {
    setAssignmentDrafts((prev) => {
      const existing = prev[identityKey] || {
        provider: "openai" as UserSettingsProvider,
        model: "",
      };
      return {
        ...prev,
        [identityKey]: {
          ...existing,
          model,
        },
      };
    });
  };

  const saveUserAssignment = async (item: LLMAssignmentUserItem) => {
    const draft = assignmentDrafts[item.identity_key];
    if (!draft?.provider || !draft?.model) return;
    setSavingAssignments((prev) => ({ ...prev, [item.identity_key]: true }));
    setError("");
    try {
      const updated = await saveLLMAssignmentForUser(
        item.login_provider,
        item.identifier,
        draft.provider,
        draft.model,
      );
      setAssignmentUsers((prev) =>
        prev.map((u) => (u.identity_key === updated.identity_key ? updated : u)),
      );
      setMessage(`Saved assignment for ${item.identity_key}.`);
    } catch (e) {
      console.error(e);
      setError(`Failed to save assignment for ${item.identity_key}.`);
    } finally {
      setSavingAssignments((prev) => ({ ...prev, [item.identity_key]: false }));
    }
  };

  const removeUserAssignment = async (item: LLMAssignmentUserItem) => {
    setSavingAssignments((prev) => ({ ...prev, [item.identity_key]: true }));
    setError("");
    try {
      const updated = await deleteLLMAssignmentForUser(item.login_provider, item.identifier);
      setAssignmentUsers((prev) =>
        prev.map((u) => (u.identity_key === updated.identity_key ? updated : u)),
      );
      setMessage(`Removed assignment for ${item.identity_key}.`);
    } catch (e) {
      console.error(e);
      setError(`Failed to remove assignment for ${item.identity_key}.`);
    } finally {
      setSavingAssignments((prev) => ({ ...prev, [item.identity_key]: false }));
    }
  };

  const saveAdminAssignment = async () => {
    if (!myDraft?.provider || !myDraft?.model) return;
    setSavingMyAssignment(true);
    setError("");
    try {
      const updated = await saveMyLLMAssignment(myDraft.provider, myDraft.model);
      setMyAssignment(updated);
      setMessage("Your Study App LLM assignment was saved.");
    } catch (e) {
      console.error(e);
      setError("Failed to save your Study App assignment.");
    } finally {
      setSavingMyAssignment(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-[1280px] mx-auto px-6 py-8">
      <div className="udemy-card p-6 mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <SettingsIcon className="w-6 h-6 text-udemy-purple" />
          Admin Settings
        </h1>
        <p className="text-sm text-udemy-text-muted mt-2">
          Platform-level controls, access management, and Study App LLM assignment policy.
        </p>
        {message && <p className="text-sm text-green-700 mt-3">{message}</p>}
        {error && <p className="text-sm text-red-600 mt-3">{error}</p>}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-6">
        <div className="udemy-card p-6 xl:col-span-2">
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
              Approved users can use backend-funded Study App LLM mode.
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
                        onClick={() => handleRemoveLlmServiceUser("github", entry.identifier)}
                        disabled={loadingLlmService}
                        className="text-red-400 hover:text-red-600 transition-colors p-1"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
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
                        onClick={() => handleRemoveLlmServiceUser("google", entry.identifier)}
                        disabled={loadingLlmService}
                        className="text-red-400 hover:text-red-600 transition-colors p-1"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
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
            {loadingHealth ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
            Check Health
          </button>

          {health && (
            <div className="mt-4 space-y-3">
              <StatusRow label="LLM Chain" ok={health.llm_chain} version={health.llm_chain_version} />
              <StatusRow label="CLI Agent" ok={health.cli_agent} version={health.cli_agent_version} />
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

      <div className="grid grid-cols-1 gap-6">
        <div className="udemy-card p-6">
          <h2 className="text-lg font-bold mb-2">Study App LLM Assignments</h2>
          <p className="text-xs text-udemy-text-muted mb-4">
            Assign fixed provider/model per allowed-login user. This applies when the user selects Study App LLM mode.
          </p>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left border-b border-udemy-border">
                  <th className="py-2 pr-4">Identity</th>
                  <th className="py-2 pr-4">Approved</th>
                  <th className="py-2 pr-4">Provider</th>
                  <th className="py-2 pr-4">Model</th>
                  <th className="py-2 pr-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {assignmentUsers.map((item) => {
                  const draft = assignmentDrafts[item.identity_key] || {
                    provider: "openai" as UserSettingsProvider,
                    model: firstModel(assignmentProviderModels, "openai"),
                  };
                  const modelOptions = assignmentProviderModels[draft.provider] || [];
                  const saving = !!savingAssignments[item.identity_key];
                  return (
                    <tr key={item.identity_key} className="border-b border-udemy-border/60">
                      <td className="py-3 pr-4">
                        <div className="font-medium">{item.identity_key}</div>
                      </td>
                      <td className="py-3 pr-4">
                        <span className={item.is_backend_approved ? "text-green-700" : "text-amber-700"}>
                          {item.is_backend_approved ? "Approved" : "Not approved"}
                        </span>
                      </td>
                      <td className="py-3 pr-4">
                        <select
                          value={draft.provider}
                          onChange={(e) => updateDraftProvider(item.identity_key, e.target.value as UserSettingsProvider)}
                          className="border border-udemy-border rounded px-2 py-1 min-w-[140px]"
                        >
                          {providerOptions.map((provider) => (
                            <option key={provider} value={provider}>
                              {provider === "google" ? "Gemini" : provider}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td className="py-3 pr-4">
                        <select
                          value={draft.model}
                          onChange={(e) => updateDraftModel(item.identity_key, e.target.value)}
                          className="border border-udemy-border rounded px-2 py-1 min-w-[220px]"
                        >
                          {modelOptions.map((model) => (
                            <option key={model} value={model}>
                              {model}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td className="py-3 pr-4">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => saveUserAssignment(item)}
                            disabled={saving || !draft.provider || !draft.model}
                            className="btn-secondary text-xs inline-flex items-center gap-1 px-3 py-1.5"
                          >
                            {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                            Save
                          </button>
                          <button
                            onClick={() => removeUserAssignment(item)}
                            disabled={saving || !item.assignment}
                            className="btn-secondary text-xs inline-flex items-center gap-1 px-3 py-1.5"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                            Remove
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
                {assignmentUsers.length === 0 && (
                  <tr>
                    <td colSpan={5} className="py-6 text-center text-udemy-text-muted">
                      No allowed-login users found.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="udemy-card p-6">
          <h2 className="text-lg font-bold mb-2 flex items-center gap-2">
            <User className="w-5 h-5 text-udemy-purple" />
            My Study App LLM
          </h2>
          <p className="text-xs text-udemy-text-muted mb-4">
            Set your own fixed Study App assignment for when you select Study App LLM mode.
          </p>
          {myAssignment && myDraft && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
              <div>
                <label className="block text-xs mb-1 text-udemy-text-muted">Identity</label>
                <div className="text-sm font-medium">{myAssignment.identity_key}</div>
              </div>
              <div>
                <label className="block text-xs mb-1 text-udemy-text-muted">Approved</label>
                <div className={myAssignment.is_backend_approved ? "text-green-700 text-sm" : "text-amber-700 text-sm"}>
                  {myAssignment.is_backend_approved ? "Approved" : "Not approved"}
                </div>
              </div>
              <div>
                <label className="block text-xs mb-1 text-udemy-text-muted">Provider</label>
                <select
                  value={myDraft.provider}
                  onChange={(e) =>
                    setMyDraft({
                      provider: e.target.value as UserSettingsProvider,
                      model: firstModel(assignmentProviderModels, e.target.value),
                    })
                  }
                  className="border border-udemy-border rounded px-2 py-2 w-full"
                >
                  {providerOptions.map((provider) => (
                    <option key={provider} value={provider}>
                      {provider === "google" ? "Gemini" : provider}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs mb-1 text-udemy-text-muted">Model</label>
                <select
                  value={myDraft.model}
                  onChange={(e) => setMyDraft((prev) => (prev ? { ...prev, model: e.target.value } : prev))}
                  className="border border-udemy-border rounded px-2 py-2 w-full"
                >
                  {(assignmentProviderModels[myDraft.provider] || []).map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}
          <button
            onClick={saveAdminAssignment}
            disabled={savingMyAssignment || !myDraft?.provider || !myDraft?.model}
            className="btn-primary mt-4 inline-flex items-center gap-2"
          >
            {savingMyAssignment ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Save My Assignment
          </button>
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
        {ok ? <CheckCircle2 className="w-4 h-4 text-green-600" /> : <XCircle className="w-4 h-4 text-red-500" />}
        <span className="text-sm font-medium">{label}</span>
      </div>
      <span className="text-xs text-udemy-text-muted">{ok ? version || "healthy" : "offline"}</span>
    </div>
  );
}
