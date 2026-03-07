/**
 * VoiceAdminSettings — admin panel section for configuring voice agent.
 */

import { useEffect, useState } from "react";
import {
  Mic,
  Save,
  Loader2,
  Globe,
  Wifi,
  Zap,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import type { VoiceConfig, VoiceTier } from "../../types";
import { fetchVoiceConfig } from "../../services/voiceService";

const API_URL = import.meta.env.VITE_API_URL || "";

interface VoiceSettings {
  enable_voice_agent: boolean;
  voice_tiers_enabled: string;
  voice_default_tier: string;
  voice_stt_provider: string;
  voice_stt_model: string;
  voice_tts_provider: string;
  voice_tts_voice: string;
}

const STT_PROVIDERS = [
  { value: "groq", label: "Groq (Free)" },
  { value: "openai", label: "OpenAI" },
];

const TTS_PROVIDERS = [
  { value: "edge", label: "Edge TTS (Free)" },
  { value: "openai", label: "OpenAI TTS" },
];

const TIERS: { value: VoiceTier; label: string; icon: typeof Globe }[] = [
  { value: "browser", label: "Browser (Free)", icon: Globe },
  { value: "cloud", label: "Cloud (~$0.01)", icon: Wifi },
  { value: "realtime", label: "Realtime (~$0.50+)", icon: Zap },
];

export default function VoiceAdminSettings() {
  const [config, setConfig] = useState<VoiceConfig | null>(null);
  const [settings, setSettings] = useState<VoiceSettings>({
    enable_voice_agent: false,
    voice_tiers_enabled: "browser",
    voice_default_tier: "browser",
    voice_stt_provider: "groq",
    voice_stt_model: "whisper-large-v3",
    voice_tts_provider: "edge",
    voice_tts_voice: "en-US-AriaNeural",
  });
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    fetchVoiceConfig()
      .then((cfg) => {
        setConfig(cfg);
        // Hydrate form from current config
        setSettings((prev) => ({
          ...prev,
          enable_voice_agent: cfg.enabled,
          voice_tiers_enabled: cfg.available_tiers.join(","),
          voice_default_tier: cfg.default_tier,
          voice_stt_provider: cfg.stt_provider || prev.voice_stt_provider,
          voice_tts_provider: cfg.tts_provider || prev.voice_tts_provider,
        }));
      })
      .catch(() => {
        setError("Could not load voice config");
      });
  }, []);

  const handleTierToggle = (tier: VoiceTier) => {
    const current = settings.voice_tiers_enabled.split(",").filter(Boolean);
    const idx = current.indexOf(tier);
    if (idx >= 0) {
      current.splice(idx, 1);
    } else {
      current.push(tier);
    }
    setSettings((prev) => ({
      ...prev,
      voice_tiers_enabled: current.join(","),
    }));
  };

  const enabledTiers = settings.voice_tiers_enabled.split(",").filter(Boolean);

  const handleSave = async () => {
    setSaving(true);
    setMessage("");
    setError("");
    try {
      const res = await fetch(`${API_URL}/api/voice/settings`, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      setMessage("Voice settings saved successfully");
      setTimeout(() => setMessage(""), 3000);
    } catch (err: any) {
      setError(err.message || "Failed to save");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="udemy-card p-6">
      <h2 className="text-lg font-bold flex items-center gap-2 mb-4">
        <Mic className="w-5 h-5 text-udemy-purple" />
        Voice Agent Settings
      </h2>

      {error && (
        <div className="mb-4 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}
      {message && (
        <div className="mb-4 rounded-lg border border-green-300 bg-green-50 p-3 text-sm text-green-700">
          {message}
        </div>
      )}

      <div className="space-y-5">
        {/* Enable/Disable */}
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={settings.enable_voice_agent}
            onChange={(e) =>
              setSettings((prev) => ({
                ...prev,
                enable_voice_agent: e.target.checked,
              }))
            }
            className="rounded border-gray-300 text-udemy-purple focus:ring-udemy-purple w-5 h-5"
          />
          <div>
            <span className="font-medium">Enable Voice Agent</span>
            <p className="text-xs text-udemy-text-muted">
              Allow users to interact with the app via voice
            </p>
          </div>
        </label>

        {/* Available Tiers */}
        <div>
          <label className="block text-sm font-medium mb-2">
            Available Tiers
          </label>
          <div className="flex flex-wrap gap-3">
            {TIERS.map(({ value, label, icon: Icon }) => {
              const active = enabledTiers.includes(value);
              return (
                <button
                  key={value}
                  onClick={() => handleTierToggle(value)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                    border transition-colors ${
                      active
                        ? "bg-udemy-purple/10 border-udemy-purple text-udemy-purple"
                        : "bg-gray-50 border-gray-200 text-gray-500 hover:bg-gray-100"
                    }`}
                >
                  {active ? (
                    <CheckCircle2 className="w-4 h-4" />
                  ) : (
                    <XCircle className="w-4 h-4 opacity-50" />
                  )}
                  <Icon className="w-4 h-4" />
                  {label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Default Tier */}
        <div>
          <label className="block text-sm font-medium mb-1">Default Tier</label>
          <select
            value={settings.voice_default_tier}
            onChange={(e) =>
              setSettings((prev) => ({
                ...prev,
                voice_default_tier: e.target.value,
              }))
            }
            className="border border-udemy-border rounded px-3 py-2 text-sm w-full max-w-xs"
          >
            {enabledTiers.map((t) => (
              <option key={t} value={t}>
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </option>
            ))}
          </select>
        </div>

        {/* STT Provider */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              STT Provider
            </label>
            <select
              value={settings.voice_stt_provider}
              onChange={(e) =>
                setSettings((prev) => ({
                  ...prev,
                  voice_stt_provider: e.target.value,
                }))
              }
              className="border border-udemy-border rounded px-3 py-2 text-sm w-full"
            >
              {STT_PROVIDERS.map(({ value, label }) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">STT Model</label>
            <input
              type="text"
              value={settings.voice_stt_model}
              onChange={(e) =>
                setSettings((prev) => ({
                  ...prev,
                  voice_stt_model: e.target.value,
                }))
              }
              className="border border-udemy-border rounded px-3 py-2 text-sm w-full"
              placeholder="whisper-large-v3"
            />
          </div>
        </div>

        {/* TTS Provider */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              TTS Provider
            </label>
            <select
              value={settings.voice_tts_provider}
              onChange={(e) =>
                setSettings((prev) => ({
                  ...prev,
                  voice_tts_provider: e.target.value,
                }))
              }
              className="border border-udemy-border rounded px-3 py-2 text-sm w-full"
            >
              {TTS_PROVIDERS.map(({ value, label }) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">TTS Voice</label>
            <input
              type="text"
              value={settings.voice_tts_voice}
              onChange={(e) =>
                setSettings((prev) => ({
                  ...prev,
                  voice_tts_voice: e.target.value,
                }))
              }
              className="border border-udemy-border rounded px-3 py-2 text-sm w-full"
              placeholder="en-US-AriaNeural"
            />
          </div>
        </div>

        <button
          onClick={handleSave}
          disabled={saving}
          className="btn-primary inline-flex items-center gap-2"
        >
          {saving ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Save className="w-4 h-4" />
          )}
          Save Voice Settings
        </button>
      </div>
    </div>
  );
}
