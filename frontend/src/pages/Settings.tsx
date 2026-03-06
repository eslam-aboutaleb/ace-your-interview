import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Settings as SettingsIcon,
  CheckCircle2,
  XCircle,
  Thermometer,
  Cpu,
  RefreshCw,
  Loader2,
  Server,
  Download,
  Cloud,
  Wifi,
  WifiOff,
  Github,
  Mail,
  Trash2,
  Plus,
  ShieldCheck,
} from "lucide-react";
import {
  pageVariants,
  pageTransition,
  containerVariants,
  cardVariants,
} from "@/utils/animations";
import {
  fetchProviders,
  fetchHealth,
  testOllamaConnection,
  fetchOllamaModels,
  fetchAllowedUsers,
  addAllowedUser,
  removeAllowedUser,
} from "@/services/api";
import type { AllowedUsers } from "@/services/api";
import { useSettingsStore } from "@/store/settingsStore";
import { useProgressStore } from "@/store/progressStore";
import type {
  ProviderStatus,
  HealthStatus,
  OllamaModelsResponse,
  OllamaTestResponse,
} from "@/types";

const FALLBACK_PROVIDER_MODELS: Record<string, string[]> = {
  openai: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
  anthropic: [
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
  ],
  google: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
  groq: [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
  ],
  ollama: ["llama3.2", "llama3.1", "mistral", "codellama", "phi3"],
  github: ["gpt-4o-mini", "gpt-4o"],
};

function fallbackProviders(): ProviderStatus[] {
  return Object.entries(FALLBACK_PROVIDER_MODELS).map(([name, models]) => ({
    name,
    models,
    available: true,
    backend: "fallback",
  }));
}

export default function SettingsPage() {
  const [providers, setProviders] = useState<ProviderStatus[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loadingProviders, setLoadingProviders] = useState(true);
  const [loadingHealth, setLoadingHealth] = useState(false);
  const [providersError, setProvidersError] = useState("");

  // Ollama-specific state
  const [ollamaTest, setOllamaTest] = useState<OllamaTestResponse | null>(null);
  const [ollamaModels, setOllamaModels] = useState<OllamaModelsResponse | null>(
    null,
  );
  const [loadingOllama, setLoadingOllama] = useState(false);

  // Allowed users state
  const [allowedUsers, setAllowedUsers] = useState<AllowedUsers | null>(null);
  const [newGithubUser, setNewGithubUser] = useState("");
  const [newGoogleEmail, setNewGoogleEmail] = useState("");
  const [loadingAllowed, setLoadingAllowed] = useState(false);

  const settings = useSettingsStore();
  const resetProgress = useProgressStore((s) => s.reset);

  useEffect(() => {
    loadProviders();
    loadAllowedUsers();
  }, []);

  const loadProviders = async () => {
    setLoadingProviders(true);
    setProvidersError("");
    try {
      const res = await fetchProviders();
      if (res.providers.length === 0) {
        setProviders(fallbackProviders());
        setProvidersError("No providers returned by backend. Showing fallback list.");
      } else {
        setProviders(res.providers);
      }
    } catch (err) {
      console.error(err);
      setProviders(fallbackProviders());
      setProvidersError(
        "Could not load provider status from backend. Showing saved/default configuration.",
      );
    } finally {
      setLoadingProviders(false);
    }
  };

  const checkHealth = async () => {
    setLoadingHealth(true);
    try {
      const res = await fetchHealth();
      setHealth(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingHealth(false);
    }
  };

  const checkOllama = async () => {
    setLoadingOllama(true);
    try {
      const [testRes, modelsRes] = await Promise.all([
        testOllamaConnection(),
        fetchOllamaModels(),
      ]);
      setOllamaTest(testRes);
      setOllamaModels(modelsRes);
    } catch (err) {
      console.error(err);
      setOllamaTest({ connected: false, version: "" });
    } finally {
      setLoadingOllama(false);
    }
  };

  // Allowed users helpers
  const loadAllowedUsers = async () => {
    try {
      const data = await fetchAllowedUsers();
      setAllowedUsers(data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleAddUser = async (provider: "github" | "google") => {
    const identifier =
      provider === "github" ? newGithubUser.trim() : newGoogleEmail.trim();
    if (!identifier) return;
    setLoadingAllowed(true);
    try {
      const data = await addAllowedUser(provider, identifier);
      setAllowedUsers(data);
      if (provider === "github") setNewGithubUser("");
      else setNewGoogleEmail("");
    } catch (err) {
      console.error(err);
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
    try {
      const data = await removeAllowedUser(provider, identifier);
      setAllowedUsers(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingAllowed(false);
    }
  };

  // Auto-check Ollama when provider is ollama
  useEffect(() => {
    if (settings.provider === "ollama" && !ollamaModels) {
      checkOllama();
    }
  }, [settings.provider]);

  const selectedProviderData = providers.find(
    (p) => p.name === settings.provider,
  );

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
    >
      <div className="bg-udemy-dark text-white">
        <div className="max-w-[1340px] mx-auto px-6 py-8">
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
            <SettingsIcon className="w-7 h-7 text-udemy-purple-light" />
            LLM Configuration
          </h1>
          <p className="text-gray-400 mt-2">
            Choose which AI provider powers your study questions
          </p>
        </div>
      </div>

      <div className="max-w-[1340px] mx-auto px-6 py-8">
        <motion.div
          className="grid grid-cols-1 lg:grid-cols-3 gap-6"
          variants={containerVariants}
          initial="hidden"
          animate="show"
        >
          {/* Provider selection */}
          <motion.div variants={cardVariants} className="lg:col-span-2">
            <div className="udemy-card p-6">
              <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Cpu className="w-5 h-5 text-udemy-purple" />
                Provider & Model
              </h2>

              {providersError && (
                <div className="mb-4 text-xs text-amber-900 bg-amber-50 border border-amber-300 rounded px-3 py-2">
                  {providersError}
                </div>
              )}

              {/* Provider grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-6">
                {loadingProviders
                  ? Array.from({ length: 6 }).map((_, i) => (
                      <div key={i} className="skeleton h-20 rounded-lg" />
                    ))
                  : providers.map((prov) => (
                      <button
                        key={prov.name}
                        onClick={() => settings.setProvider(prov.name)}
                        className={`relative p-4 rounded-lg border-2 text-left transition-all
                        ${
                          settings.provider === prov.name
                            ? "border-udemy-purple bg-udemy-purple/5"
                            : "border-udemy-border hover:border-gray-400"
                        }
                        ${!prov.available ? "opacity-50" : ""}`}
                      >
                        <div className="font-bold text-sm capitalize mb-1">
                          {prov.name}
                        </div>
                        <div className="text-xs text-udemy-text-muted">
                          {prov.models.length} models
                        </div>
                        <div className="text-xs text-udemy-text-muted">
                          via {prov.backend}
                        </div>
                        {/* Status dot */}
                        <span
                          className={`absolute top-2 right-2 w-2.5 h-2.5 rounded-full ${
                            prov.available ? "bg-green-500" : "bg-red-400"
                          }`}
                        />
                        {settings.provider === prov.name && (
                          <motion.span
                            layoutId="provider-check"
                            className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-udemy-purple rounded-full flex items-center justify-center"
                          >
                            <CheckCircle2 className="w-3.5 h-3.5 text-white" />
                          </motion.span>
                        )}
                      </button>
                    ))}
              </div>

              {/* Model selector */}
              <div className="mb-4">
                <label className="block text-sm font-medium mb-1.5">
                  Model
                </label>
                <select
                  value={settings.model}
                  onChange={(e) => settings.setModel(e.target.value)}
                  className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                >
                  <option value="">Use provider default</option>
                  {selectedProviderData?.models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>

              {/* Temperature */}
              <div>
                <label className="flex items-center gap-2 text-sm font-medium mb-1.5">
                  <Thermometer className="w-4 h-4 text-udemy-purple" />
                  Temperature: {settings.temperature.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) =>
                    settings.setTemperature(parseFloat(e.target.value))
                  }
                  className="w-full accent-udemy-purple"
                />
                <div className="flex justify-between text-xs text-udemy-text-muted mt-1">
                  <span>Precise (0)</span>
                  <span>Creative (1.5)</span>
                </div>
              </div>

              {/* Ollama section — shown when Ollama is selected */}
              <AnimatePresence>
                {settings.provider === "ollama" && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="border-t border-udemy-border pt-4 mt-2">
                      <h3 className="text-sm font-bold flex items-center gap-2 mb-3">
                        <Server className="w-4 h-4 text-udemy-purple" />
                        Ollama Server
                      </h3>

                      {/* Connection status */}
                      <div className="flex items-center gap-3 mb-3">
                        <button
                          onClick={checkOllama}
                          disabled={loadingOllama}
                          className="btn-secondary text-xs flex items-center gap-1.5 py-1.5 px-3"
                        >
                          {loadingOllama ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <RefreshCw className="w-3.5 h-3.5" />
                          )}
                          Test Connection
                        </button>
                        {ollamaTest && (
                          <span
                            className={`text-xs font-medium flex items-center gap-1 ${
                              ollamaTest.connected
                                ? "text-green-600"
                                : "text-red-500"
                            }`}
                          >
                            {ollamaTest.connected ? (
                              <>
                                <Wifi className="w-3.5 h-3.5" />
                                Connected (v{ollamaTest.version})
                              </>
                            ) : (
                              <>
                                <WifiOff className="w-3.5 h-3.5" />
                                Not connected
                              </>
                            )}
                          </span>
                        )}
                      </div>

                      {ollamaModels && ollamaModels.connected && (
                        <>
                          {/* Downloaded models */}
                          {ollamaModels.downloaded_models.length > 0 && (
                            <div className="mb-3">
                              <h4 className="text-xs font-bold text-udemy-text-muted uppercase flex items-center gap-1 mb-2">
                                <Download className="w-3 h-3" />
                                Downloaded Models
                              </h4>
                              <div className="space-y-1">
                                {ollamaModels.downloaded_models.map((m) => (
                                  <button
                                    key={m.name}
                                    onClick={() => settings.setModel(m.name)}
                                    className={`w-full text-left px-3 py-2 rounded text-sm flex items-center justify-between transition-colors ${
                                      settings.model === m.name
                                        ? "bg-udemy-purple/10 text-udemy-purple font-medium"
                                        : "hover:bg-gray-50 text-udemy-text"
                                    }`}
                                  >
                                    <span className="truncate">{m.name}</span>
                                    {m.size && (
                                      <span className="text-xs text-udemy-text-muted flex-shrink-0 ml-2">
                                        {m.size}
                                      </span>
                                    )}
                                  </button>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Cloud models suggestions */}
                          <div>
                            <h4 className="text-xs font-bold text-udemy-text-muted uppercase flex items-center gap-1 mb-2">
                              <Cloud className="w-3 h-3" />
                              Available to Pull
                            </h4>
                            <div className="flex flex-wrap gap-1.5">
                              {ollamaModels.cloud_models
                                .filter(
                                  (cm) =>
                                    !ollamaModels.downloaded_models.some((dm) =>
                                      dm.name.startsWith(cm),
                                    ),
                                )
                                .map((m) => (
                                  <span
                                    key={m}
                                    className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded"
                                  >
                                    {m}
                                  </span>
                                ))}
                            </div>
                            <p className="text-xs text-udemy-text-muted mt-2">
                              Run{" "}
                              <code className="bg-gray-100 px-1 rounded">
                                ollama pull &lt;model&gt;
                              </code>{" "}
                              to download
                            </p>
                          </div>
                        </>
                      )}

                      {ollamaModels && !ollamaModels.connected && (
                        <p className="text-xs text-udemy-text-muted">
                          Start Ollama with{" "}
                          <code className="bg-gray-100 px-1 rounded">
                            ollama serve
                          </code>{" "}
                          to see available models.
                        </p>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>

          {/* Health & status panel */}
          <motion.div variants={cardVariants}>
            <div className="udemy-card p-6 space-y-6">
              <div>
                <h2 className="text-lg font-bold mb-4">Service Health</h2>
                <button
                  onClick={checkHealth}
                  disabled={loadingHealth}
                  className="btn-secondary w-full flex items-center justify-center gap-2 text-sm"
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
              </div>

              {/* Current config summary */}
              <div className="border-t border-udemy-border pt-4">
                <h3 className="text-sm font-bold text-udemy-text-muted uppercase mb-3">
                  Current Config
                </h3>
                <dl className="space-y-2 text-sm">
                  <ConfigRow label="Provider" value={settings.provider} />
                  <ConfigRow
                    label="Model"
                    value={settings.model || "(default)"}
                  />
                  <ConfigRow
                    label="Temperature"
                    value={settings.temperature.toFixed(1)}
                  />
                </dl>
              </div>

              {/* Danger zone */}
              <div className="border-t border-udemy-border pt-4">
                <h3 className="text-sm font-bold text-udemy-danger uppercase mb-3">
                  Danger Zone
                </h3>
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
          </motion.div>
        </motion.div>

        {/* ── Allowed Users Section ───────────────────────────── */}
        <motion.div
          variants={cardVariants}
          initial="hidden"
          animate="show"
          className="mt-6"
        >
          <div className="udemy-card p-6">
            <h2 className="text-lg font-bold mb-1 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-udemy-purple" />
              Allowed Users
            </h2>
            <p className="text-sm text-udemy-text-muted mb-5">
              Only these accounts can sign in to this app.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* GitHub users */}
              <div>
                <h3 className="text-sm font-bold flex items-center gap-2 mb-3">
                  <Github className="w-4 h-4" />
                  GitHub Usernames
                </h3>
                <div className="space-y-2 mb-3">
                  {allowedUsers?.github_users.map((u) => (
                    <div
                      key={u}
                      className="flex items-center justify-between bg-udemy-bg rounded px-3 py-2 text-sm"
                    >
                      <span className="font-medium">{u}</span>
                      <button
                        onClick={() => handleRemoveUser("github", u)}
                        disabled={loadingAllowed}
                        className="text-red-400 hover:text-red-600 transition-colors p-1"
                        title="Remove"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                  {allowedUsers?.github_users.length === 0 && (
                    <p className="text-xs text-udemy-text-muted italic">
                      No GitHub users allowed
                    </p>
                  )}
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="github-username"
                    value={newGithubUser}
                    onChange={(e) => setNewGithubUser(e.target.value)}
                    onKeyDown={(e) =>
                      e.key === "Enter" && handleAddUser("github")
                    }
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

              {/* Google emails */}
              <div>
                <h3 className="text-sm font-bold flex items-center gap-2 mb-3">
                  <Mail className="w-4 h-4" />
                  Google Emails
                </h3>
                <div className="space-y-2 mb-3">
                  {allowedUsers?.google_emails.map((e) => (
                    <div
                      key={e}
                      className="flex items-center justify-between bg-udemy-bg rounded px-3 py-2 text-sm"
                    >
                      <span className="font-medium">{e}</span>
                      <button
                        onClick={() => handleRemoveUser("google", e)}
                        disabled={loadingAllowed}
                        className="text-red-400 hover:text-red-600 transition-colors p-1"
                        title="Remove"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                  {allowedUsers?.google_emails.length === 0 && (
                    <p className="text-xs text-udemy-text-muted italic">
                      No Google emails allowed
                    </p>
                  )}
                </div>
                <div className="flex gap-2">
                  <input
                    type="email"
                    placeholder="user@gmail.com"
                    value={newGoogleEmail}
                    onChange={(e) => setNewGoogleEmail(e.target.value)}
                    onKeyDown={(e) =>
                      e.key === "Enter" && handleAddUser("google")
                    }
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
          </div>
        </motion.div>
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

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-udemy-text-muted">{label}</dt>
      <dd className="font-medium capitalize">{value}</dd>
    </div>
  );
}
