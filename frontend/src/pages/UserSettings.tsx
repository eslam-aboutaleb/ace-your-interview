import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  KeyRound,
  UserCheck,
  Save,
  Loader2,
  Link as LinkIcon,
  Unlink,
  ShieldCheck,
} from "lucide-react";
import {
  connectGeminiAccount,
  deleteUserApiKey,
  disconnectGeminiAccount,
  fetchUserSettings,
  saveUserApiKey,
  updateUserPreferences,
} from "@/services/api";
import type {
  ProviderConnectionStatus,
  UserAuthMode,
  UserLLMSource,
  UserPreferences,
  UserSettingsProvider,
  UserSettingsResponse,
} from "@/types";
import { useSettingsStore } from "@/store/settingsStore";

function readPolicyErrorFromUrl(): string {
  const params = new URLSearchParams(window.location.search);
  const policy = params.get("policy");
  if (policy === "llm_service_approval_required") {
    return "You are not approved for backend LLM service. Ask admin approval or use Personal LLM mode.";
  }
  if (policy === "study_app_llm_not_assigned") {
    return "Study App LLM is selected but admin has not assigned a provider/model for your account yet.";
  }
  if (policy === "personal_credential_required") {
    return "Personal LLM mode requires your own API key/account. Add credentials or switch mode.";
  }
  return "";
}

export default function UserSettingsPage() {
  const settings = useSettingsStore();
  const [data, setData] = useState<UserSettingsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [savingPrefs, setSavingPrefs] = useState(false);
  const [savingKey, setSavingKey] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [authMode, setAuthMode] = useState<UserAuthMode>("api_key");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const selectedProvider = settings.provider as UserSettingsProvider;
  const selectedSource = settings.llmSource as UserLLMSource;
  const inStudyAppMode = selectedSource === "study_app";

  const providerStatus = useMemo<ProviderConnectionStatus | undefined>(() => {
    return data?.providers.find((p) => p.provider === selectedProvider);
  }, [data, selectedProvider]);

  const canUseAccountMode = selectedProvider === "google";
  const showNotApprovedBanner = !!data && !data.study_app_available;
  const showMissingAssignmentBanner = inStudyAppMode && !!data && !data.study_app_assignment;
  const showPersonalCredentialBanner =
    !inStudyAppMode &&
    !!providerStatus &&
    !providerStatus.api_key_connected &&
    !providerStatus.account_connected;

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetchUserSettings();
      setData(res);
      setAuthMode(res.preferences.auth_mode);

      const providerForStore =
        res.preferences.llm_source === "study_app" && res.study_app_assignment
          ? res.study_app_assignment.provider
          : res.preferences.provider;
      const modelForStore =
        res.preferences.llm_source === "study_app" && res.study_app_assignment
          ? res.study_app_assignment.model
          : res.preferences.model;

      settings.hydrateFromServer({
        llmSource: res.preferences.llm_source,
        provider: providerForStore,
        model: modelForStore,
        temperature: res.preferences.temperature,
        maxTokens: res.preferences.max_tokens,
        requireAnswerReveal: res.preferences.require_answer_reveal,
      });
    } catch (e) {
      console.error(e);
      setError("Failed to load user settings.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const policyError = readPolicyErrorFromUrl();
    if (policyError) {
      setError(policyError);
      window.history.replaceState({}, "", window.location.pathname);
    }
    if (params.get("google_connected") === "1") {
      setMessage("Gemini account connected.");
      window.history.replaceState({}, "", window.location.pathname);
    }
    if (params.get("error")) {
      setError("Could not connect Gemini account.");
      window.history.replaceState({}, "", window.location.pathname);
    }
    load();
  }, []);

  const handleSavePreferences = async () => {
    setSavingPrefs(true);
    setError("");
    setMessage("");
    try {
      if (selectedSource === "study_app" && !data?.study_app_available) {
        setError("You are not approved to use Study App LLM.");
        setSavingPrefs(false);
        return;
      }
      const effectiveAuthMode: UserAuthMode =
        selectedSource === "study_app"
          ? "api_key"
          : canUseAccountMode
            ? authMode
            : "api_key";
      const assignmentProvider = data?.study_app_assignment?.provider || settings.provider;
      const assignmentModel = data?.study_app_assignment?.model || settings.model;

      const payload: UserPreferences = {
        provider: (selectedSource === "study_app" ? assignmentProvider : selectedProvider) as UserSettingsProvider,
        model: selectedSource === "study_app" ? assignmentModel : settings.model,
        temperature: settings.temperature,
        max_tokens: settings.maxTokens,
        auth_mode: effectiveAuthMode,
        llm_source: selectedSource,
        require_answer_reveal: settings.requireAnswerReveal,
      };
      const saved = await updateUserPreferences(payload);
      setAuthMode(saved.auth_mode);
      setData((prev) => (prev ? { ...prev, has_saved_preferences: true, preferences: saved } : prev));
      setMessage("Preferences saved.");
      await load();
    } catch (e) {
      console.error(e);
      setError("Failed to save preferences.");
    } finally {
      setSavingPrefs(false);
    }
  };

  const handleSaveApiKey = async () => {
    if (!apiKeyInput.trim()) return;
    setSavingKey(true);
    setError("");
    setMessage("");
    try {
      const next = await saveUserApiKey(selectedProvider, apiKeyInput.trim());
      setData(next);
      setApiKeyInput("");
      setMessage("API key saved.");
    } catch (e) {
      console.error(e);
      setError("Failed to save API key.");
    } finally {
      setSavingKey(false);
    }
  };

  const handleDeleteApiKey = async () => {
    setSavingKey(true);
    setError("");
    setMessage("");
    try {
      const next = await deleteUserApiKey(selectedProvider);
      setData(next);
      setMessage("API key removed.");
    } catch (e) {
      console.error(e);
      setError("Failed to remove API key.");
    } finally {
      setSavingKey(false);
    }
  };

  const handleConnectGemini = () => {
    window.location.href = connectGeminiAccount();
  };

  const handleDisconnectGemini = async () => {
    setSavingPrefs(true);
    setError("");
    setMessage("");
    try {
      const next = await disconnectGeminiAccount();
      setData(next);
      setAuthMode("api_key");
      setMessage("Gemini account disconnected.");
    } catch (e) {
      console.error(e);
      setError("Failed to disconnect Gemini account.");
    } finally {
      setSavingPrefs(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <Loader2 className="w-7 h-7 animate-spin text-udemy-purple" />
      </div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-[1100px] mx-auto px-6 py-8">
      <div className="udemy-card p-6 mb-6">
        <h1 className="text-2xl font-bold mb-2">User Settings</h1>
        <p className="text-sm text-udemy-text-muted">
          Choose your runtime LLM source and manage personal credentials.
        </p>
        {message && <p className="mt-3 text-sm text-green-700">{message}</p>}
        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
        {showNotApprovedBanner && (
          <p className="mt-3 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
            You are not approved for Study App LLM. Ask admin approval to enable this mode.
          </p>
        )}
        {showMissingAssignmentBanner && (
          <p className="mt-3 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
            Study App LLM is selected but admin has not assigned your provider/model yet.
          </p>
        )}
        {showPersonalCredentialBanner && (
          <p className="mt-3 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
            Personal LLM mode needs your own API key/account for the selected provider.
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 udemy-card p-6">
          <h2 className="text-lg font-bold mb-4">LLM Source</h2>
          <div className="rounded-lg border border-udemy-border p-4 mb-6">
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="radio"
                  name="llm-source"
                  checked={selectedSource === "personal"}
                  onChange={() => settings.setLLMSource("personal")}
                />
                Personal LLM
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="radio"
                  name="llm-source"
                  checked={selectedSource === "study_app"}
                  disabled={!data?.study_app_available}
                  onChange={() => settings.setLLMSource("study_app")}
                />
                Study App LLM
              </label>
            </div>
          </div>

          {inStudyAppMode ? (
            <div className="rounded-lg border border-udemy-border p-4 mb-6">
              <h3 className="text-sm font-semibold mb-2">Assigned Study App LLM</h3>
              {data?.study_app_assignment ? (
                <div className="text-sm space-y-1">
                  <p>
                    Provider: <span className="font-medium capitalize">{data.study_app_assignment.provider === "google" ? "Gemini" : data.study_app_assignment.provider}</span>
                  </p>
                  <p>
                    Model: <span className="font-medium">{data.study_app_assignment.model}</span>
                  </p>
                </div>
              ) : (
                <p className="text-sm text-udemy-text-muted">
                  No assignment yet. Ask admin to assign your Study App provider/model.
                </p>
              )}
            </div>
          ) : (
            <>
              <h2 className="text-lg font-bold mb-4">Personal Provider & Model</h2>
              <div className="grid grid-cols-2 gap-3 mb-6">
                {data?.providers.map((p) => (
                  <button
                    key={p.provider}
                    onClick={() => settings.setProvider(p.provider)}
                    className={`p-3 rounded-lg border-2 text-left transition-colors ${
                      selectedProvider === p.provider
                        ? "border-udemy-purple bg-udemy-purple/5"
                        : "border-udemy-border hover:border-gray-400"
                    }`}
                  >
                    <div className="font-semibold capitalize">{p.provider === "google" ? "Gemini" : p.provider}</div>
                    <div className="text-xs text-udemy-text-muted mt-1">{p.models.length} models</div>
                  </button>
                ))}
              </div>

              <label className="block text-sm font-medium mb-2">Model</label>
              <select
                value={settings.model}
                onChange={(e) => settings.setModel(e.target.value)}
                className="w-full border border-udemy-border rounded px-3 py-2 mb-4"
              >
                <option value="">Use provider default</option>
                {(providerStatus?.models || []).map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </>
          )}

          <label className="flex items-center gap-2 text-sm font-medium mb-2">
            Temperature: {settings.temperature.toFixed(1)}
          </label>
          <input
            type="range"
            min="0"
            max="1.5"
            step="0.1"
            value={settings.temperature}
            onChange={(e) => settings.setTemperature(parseFloat(e.target.value))}
            className="w-full accent-udemy-purple mb-6"
          />

          <div className="mb-6 rounded-lg border border-udemy-border p-4">
            <h3 className="text-sm font-semibold mb-2">Study Behavior</h3>
            <label className="flex items-start gap-3 text-sm">
              <input
                type="checkbox"
                checked={settings.requireAnswerReveal}
                onChange={(e) => settings.setRequireAnswerReveal(e.target.checked)}
                className="mt-0.5 accent-udemy-purple"
              />
              <span>
                Require clicking <span className="font-medium">Reveal Official Answer</span> before showing answers
                in Topics study mode.
              </span>
            </label>
          </div>

          <button
            onClick={handleSavePreferences}
            disabled={savingPrefs}
            className="btn-primary inline-flex items-center gap-2"
          >
            {savingPrefs ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Save Preferences
          </button>
        </div>

        <div className="udemy-card p-6 space-y-6">
          {!inStudyAppMode && (
            <>
              <div>
                <h3 className="text-sm font-bold uppercase text-udemy-text-muted mb-2">Authentication Mode</h3>
                {canUseAccountMode ? (
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="radio"
                        name="auth-mode"
                        checked={authMode === "api_key"}
                        onChange={() => setAuthMode("api_key")}
                      />
                      API Key
                    </label>
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="radio"
                        name="auth-mode"
                        checked={authMode === "account"}
                        onChange={() => setAuthMode("account")}
                      />
                      Connected Account
                    </label>
                  </div>
                ) : (
                  <p className="text-sm text-udemy-text-muted">Only API key mode is available for this provider.</p>
                )}
              </div>

              {selectedProvider === "google" && (
                <div>
                  <h3 className="text-sm font-bold uppercase text-udemy-text-muted mb-2">Gemini Account</h3>
                  <div className="flex items-center gap-2 text-sm mb-3">
                    <UserCheck className="w-4 h-4 text-udemy-purple" />
                    {providerStatus?.account_connected ? "Connected" : "Not connected"}
                  </div>
                  {providerStatus?.account_connected ? (
                    <button
                      onClick={handleDisconnectGemini}
                      className="btn-secondary w-full inline-flex items-center justify-center gap-2"
                    >
                      <Unlink className="w-4 h-4" />
                      Disconnect Account
                    </button>
                  ) : (
                    <button
                      onClick={handleConnectGemini}
                      className="btn-secondary w-full inline-flex items-center justify-center gap-2"
                    >
                      <LinkIcon className="w-4 h-4" />
                      Connect Account
                    </button>
                  )}
                </div>
              )}

              <div>
                <h3 className="text-sm font-bold uppercase text-udemy-text-muted mb-2">API Key</h3>
                <input
                  type="password"
                  placeholder="Paste API key"
                  value={apiKeyInput}
                  onChange={(e) => setApiKeyInput(e.target.value)}
                  className="w-full border border-udemy-border rounded px-3 py-2 mb-3"
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleSaveApiKey}
                    disabled={savingKey || !apiKeyInput.trim()}
                    className="btn-secondary flex-1 inline-flex items-center justify-center gap-2"
                  >
                    {savingKey ? <Loader2 className="w-4 h-4 animate-spin" /> : <KeyRound className="w-4 h-4" />}
                    Save
                  </button>
                  <button
                    onClick={handleDeleteApiKey}
                    disabled={savingKey || !providerStatus?.api_key_connected}
                    className="btn-secondary flex-1"
                  >
                    Remove
                  </button>
                </div>
              </div>
            </>
          )}

          <div className="border-t border-udemy-border pt-4 text-sm">
            <h3 className="text-sm font-bold uppercase text-udemy-text-muted mb-2">Status</h3>
            <p className="mb-1">Current source: {inStudyAppMode ? "Study App LLM" : "Personal LLM"}</p>
            <p className="mb-1">API key: {providerStatus?.api_key_connected ? "Connected" : "Not connected"}</p>
            <p className="mb-1">Account: {providerStatus?.account_connected ? "Connected" : "Not connected"}</p>
            <p className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-udemy-purple" />
              Study App access: {data?.study_app_available ? "Enabled" : "Disabled"}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
