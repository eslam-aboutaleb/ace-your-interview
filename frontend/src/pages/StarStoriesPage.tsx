import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  ChevronDown,
  Loader2,
  Pencil,
  Plus,
  Search,
  Sparkles,
  Star,
  Trash2,
  X,
} from "lucide-react";
import { pageVariants, pageTransition } from "@/utils/animations";
import {
  createStarStory,
  deleteStarStory,
  fetchStarStories,
  suggestStarStories,
  updateStarStory,
} from "@/services/api";
import { normalizeEscapedMultilineText } from "@/utils/textNormalization";
import type { StarStory, StarStoryCreateRequest } from "@/types";

const EMPTY_FORM: StarStoryCreateRequest = {
  title: "",
  situation: "",
  task: "",
  action: "",
  result: "",
  tags: [],
};

export default function StarStoriesPage() {
  const [stories, setStories] = useState<StarStory[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<StarStoryCreateRequest>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);

  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<StarStory[]>([]);
  const [suggesting, setSuggesting] = useState(false);

  const loadStories = useCallback(async () => {
    setLoading(true);
    setErrorMsg("");
    try {
      const res = await fetchStarStories();
      setStories(res.stories);
    } catch {
      setErrorMsg("Could not load STAR stories.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStories();
  }, [loadStories]);

  const handleSuggest = async () => {
    const q = query.trim();
    if (!q) return;
    setSuggesting(true);
    try {
      const res = await suggestStarStories(q, 3);
      setSuggestions(res.stories);
    } catch {
      setSuggestions([]);
    } finally {
      setSuggesting(false);
    }
  };

  const resetForm = () => {
    setForm(EMPTY_FORM);
    setEditingId(null);
    setShowForm(false);
  };

  const handleEdit = (story: StarStory) => {
    setForm({
      title: story.title,
      situation: story.situation,
      task: story.task,
      action: story.action,
      result: story.result,
      tags: story.tags,
    });
    setEditingId(story.story_id);
    setShowForm(true);
  };

  const handleSave = async () => {
    if (!form.title.trim() || !form.action.trim()) return;
    setSaving(true);
    setErrorMsg("");
    try {
      if (editingId) {
        const updated = await updateStarStory(editingId, form);
        setStories((prev) =>
          prev.map((s) => (s.story_id === editingId ? updated : s)),
        );
      } else {
        const created = await createStarStory(form);
        setStories((prev) => [created, ...prev]);
      }
      resetForm();
    } catch {
      setErrorMsg("Could not save the story.");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (storyId: string) => {
    if (!window.confirm("Delete this STAR story?")) return;
    try {
      await deleteStarStory(storyId);
      setStories((prev) => prev.filter((s) => s.story_id !== storyId));
      if (expandedId === storyId) setExpandedId(null);
    } catch {
      setErrorMsg("Could not delete the story.");
    }
  };

  const updateField = (
    field: keyof StarStoryCreateRequest,
    value: string,
  ) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
    >
      <div className="bg-udemy-dark text-white">
        <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-8">
          <h1 className="text-2xl font-bold flex items-center gap-3">
            <Star className="w-7 h-7 text-udemy-purple-light" />
            STAR Story Bank
          </h1>
          <p className="text-gray-400 mt-2">
            Save behavioral stories and surface the right one for any
            interview question.
          </p>
        </div>
      </div>

      <div className="max-w-[1340px] mx-auto px-4 sm:px-6 py-8 space-y-6">
        {errorMsg && (
          <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        <div className="udemy-card p-6">
          <h2 className="text-sm font-bold text-udemy-text-muted uppercase mb-3">
            Find a matching story
          </h2>
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-udemy-text-muted" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSuggest();
                }}
                placeholder="e.g. Tell me about a time you handled a conflict…"
                className="w-full border border-udemy-border rounded pl-9 pr-3 py-2.5 text-sm"
              />
            </div>
            <button
              onClick={handleSuggest}
              disabled={suggesting || !query.trim()}
              className="btn-secondary inline-flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {suggesting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              Match stories
            </button>
          </div>
          {suggestions.length > 0 && (
            <div className="mt-4 space-y-2">
              {suggestions.map((story) => (
                <button
                  key={story.story_id}
                  onClick={() => setExpandedId(story.story_id)}
                  className="w-full text-left bg-udemy-bg rounded-lg p-3 hover:border-udemy-purple border border-transparent transition-colors"
                >
                  <p className="text-sm font-medium">{story.title}</p>
                  <p className="text-xs text-udemy-text-muted mt-0.5 line-clamp-1">
                    {story.result}
                  </p>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold">
            Your stories ({stories.length})
          </h2>
          <button
            onClick={() => {
              resetForm();
              setShowForm(true);
            }}
            className="btn-primary inline-flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New story
          </button>
        </div>

        {showForm && (
          <div className="udemy-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-bold">
                {editingId ? "Edit story" : "New STAR story"}
              </h3>
              <button
                onClick={resetForm}
                className="text-udemy-text-muted hover:text-udemy-text"
                aria-label="Close form"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1.5">
                Title
              </label>
              <input
                value={form.title}
                onChange={(e) => updateField("title", e.target.value)}
                placeholder="e.g. Resolved a production outage"
                className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
              />
            </div>
            {(
              [
                ["situation", "Situation"],
                ["task", "Task"],
                ["action", "Action"],
                ["result", "Result"],
              ] as const
            ).map(([field, label]) => (
              <div key={field}>
                <label className="block text-sm font-medium mb-1.5">
                  {label}
                </label>
                <textarea
                  value={form[field]}
                  onChange={(e) => updateField(field, e.target.value)}
                  rows={3}
                  className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
                />
              </div>
            ))}
            <div>
              <label className="block text-sm font-medium mb-1.5">
                Tags (comma-separated)
              </label>
              <input
                value={(form.tags ?? []).join(", ")}
                onChange={(e) =>
                  setForm((prev) => ({
                    ...prev,
                    tags: e.target.value
                      .split(",")
                      .map((t) => t.trim())
                      .filter(Boolean)
                      .slice(0, 20),
                  }))
                }
                placeholder="leadership, debugging, postgres"
                className="w-full border border-udemy-border rounded px-3 py-2.5 text-sm"
              />
            </div>
            <div className="flex gap-3">
              <button
                onClick={handleSave}
                disabled={
                  saving || !form.title.trim() || !form.action.trim()
                }
                className="btn-primary inline-flex items-center gap-2 disabled:opacity-50"
              >
                {saving ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4" />
                )}
                {saving ? "Saving..." : "Save story"}
              </button>
              <button onClick={resetForm} className="btn-secondary">
                Cancel
              </button>
            </div>
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center h-40">
            <Loader2 className="w-6 h-6 animate-spin text-udemy-purple" />
          </div>
        ) : stories.length === 0 ? (
          <div className="udemy-card p-8 text-center">
            <Star className="w-8 h-8 text-udemy-text-muted mx-auto mb-3" />
            <p className="text-udemy-text-muted">
              No STAR stories yet. Add one to personalize behavioral
              interview answers.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {stories.map((story) => {
              const expanded = expandedId === story.story_id;
              return (
                <div key={story.story_id} className="udemy-card overflow-hidden">
                  <button
                    onClick={() =>
                      setExpandedId(expanded ? null : story.story_id)
                    }
                    className="w-full p-4 text-left flex items-start gap-3 hover:bg-gray-50 transition-colors"
                  >
                    <Star className="w-4 h-4 mt-0.5 text-udemy-purple flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium">{story.title}</p>
                      <div className="flex flex-wrap gap-1.5 mt-1.5">
                        {story.tags.map((tag) => (
                          <span
                            key={tag}
                            className="text-[11px] bg-udemy-bg text-udemy-text-muted rounded-full px-2 py-0.5"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                    <ChevronDown
                      className={`w-4 h-4 text-udemy-text-muted transition-transform flex-shrink-0 ${
                        expanded ? "rotate-180" : ""
                      }`}
                    />
                  </button>
                  {expanded && (
                    <div className="px-4 pb-4 border-t border-udemy-border pt-3 space-y-3">
                      {(
                        [
                          ["situation", "Situation"],
                          ["task", "Task"],
                          ["action", "Action"],
                          ["result", "Result"],
                        ] as const
                      ).map(([field, label]) => (
                        <div key={field}>
                          <p className="text-[11px] font-bold uppercase text-udemy-text-muted mb-0.5">
                            {label}
                          </p>
                          <p className="text-sm whitespace-pre-line">
                            {normalizeEscapedMultilineText(story[field])}
                          </p>
                        </div>
                      ))}
                      <div className="flex gap-2 pt-1">
                        <button
                          onClick={() => handleEdit(story)}
                          className="btn-secondary inline-flex items-center gap-1.5 text-sm"
                        >
                          <Pencil className="w-3.5 h-3.5" />
                          Edit
                        </button>
                        <button
                          onClick={() => handleDelete(story.story_id)}
                          className="inline-flex items-center gap-1.5 text-sm text-red-600 hover:underline"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                          Delete
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </motion.div>
  );
}
