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
  UserPreferences,
  UserSettingsProvider,
  UserSettingsResponse,
} from "@/types";
import { useSettingsStore } from "@/store/settingsStore";

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

  const providerStatus = useMemo<ProviderConnectionStatus | undefined>(() => {
    return data?.providers.find((p) => p.provider === selectedProvider);
  }, [data, selectedProvider]);

  const canUseAccountMode = selectedProvider === "google";
  const showApprovalWarning =
    !!providerStatus &&
    !providerStatus.api_key_connected &&
    !providerStatus.account_connected &&
    !providerStatus.backend_fallback_eligible;

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetchUserSettings();
      setData(res);
      setAuthMode(res.preferences.auth_mode);
      settings.hydrateFromServer({
        provider: res.preferences.provider,
        model: res.preferences.model,
        temperature: res.preferences.temperature,
        maxTokens: res.preferences.max_tokens,
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
    if (params.get("policy") === "llm_service_approval_required") {
      setError(
        "You are not approved for backend LLM service. Connect your own API key/account or ask admin.",
      );
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
      const effectiveAuthMode: UserAuthMode = canUseAccountMode ? authMode : "api_key";
      const payload: UserPreferences = {
        provider: selectedProvider,
        model: settings.model,
        temperature: settings.temperature,
        max_tokens: settings.maxTokens,
        auth_mode: effectiveAuthMode,
      };
      const saved = await updateUserPreferences(payload);
      setAuthMode(saved.auth_mode);
      setData((prev) =>
        prev
          ? {
              ...prev,
              has_saved_preferences: true,
              preferences: saved,
            }
          : prev,
      );
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
          Configure your personal LLM access and model selection.
        </p>
        {message && <p className="mt-3 text-sm text-green-700">{message}</p>}
        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
        {showApprovalWarning && (
          <p className="mt-3 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
            You are not approved for backend LLM service. Connect your own API key/account or ask admin.
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 udemy-card p-6">
          <h2 className="text-lg font-bold mb-4">Provider & Model</h2>

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

          <div className="border-t border-udemy-border pt-4 text-sm">
            <h3 className="text-sm font-bold uppercase text-udemy-text-muted mb-2">Status</h3>
            <p className="mb-1">API key: {providerStatus?.api_key_connected ? "Connected" : "Not connected"}</p>
            <p className="mb-1">Account: {providerStatus?.account_connected ? "Connected" : "Not connected"}</p>
            <p className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-udemy-purple" />
              Backend fallback: {providerStatus?.backend_fallback_eligible ? "Active" : "Off"}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
