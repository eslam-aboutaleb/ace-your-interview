import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Download,
  FileUp,
  Layers,
  Loader2,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
} from "lucide-react";
import { pageTransition, pageVariants } from "@/utils/animations";
import {
  addDeckCard,
  createDeck,
  deleteDeck,
  deleteDeckCard,
  exportDeckApkgUrl,
  fetchDeckCards,
  fetchDecks,
  fetchStudySession,
  findDuplicateCards,
  generateDeckCards,
  fetchCardGenerationJob,
  importDeck,
  reviewDeckCard,
  updateDeck,
  updateDeckCard,
} from "@/services/api";
import type {
  DeckResponse,
  FlashcardResponse,
  GeneratedCardItem,
  LearningReviewRating,
} from "@/types";

const MAX_IMPORT_BYTES = 50 * 1024 * 1024;
const RATINGS: Array<{ value: LearningReviewRating; label: string }> = [
  { value: "again", label: "Again" },
  { value: "hard", label: "Hard" },
  { value: "good", label: "Good" },
  { value: "easy", label: "Easy" },
];

function formatDateTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export default function DecksPage() {
  const [decks, setDecks] = useState<DeckResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");
  const [disabledNotice, setDisabledNotice] = useState("");
  const [selectedDeckId, setSelectedDeckId] = useState<string | null>(
    null,
  );
  const [cards, setCards] = useState<FlashcardResponse[]>([]);
  const [cardsLoading, setCardsLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newDeckName, setNewDeckName] = useState("");
  const [newDeckDescription, setNewDeckDescription] = useState("");
  const [importing, setImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDecks = useCallback(async () => {
    try {
      const res = await fetchDecks();
      setDecks(res.decks || []);
      setErrorMsg("");
      setDisabledNotice("");
      return res.decks || [];
    } catch (err: any) {
      if (err?.response?.status === 503) {
        setDisabledNotice(
          "Flashcards are not enabled on this server " +
            "(STUDY_ENABLE_FLASHCARDS_V1=false).",
        );
        setErrorMsg("");
      } else {
        setErrorMsg("Could not load decks. Please retry.");
      }
      return [];
    }
  }, []);

  useEffect(() => {
    (async () => {
      setLoading(true);
      const list = await loadDecks();
      setLoading(false);
      if (!selectedDeckId && list.length > 0) {
        setSelectedDeckId(list[0].deck_id);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadDecks]);

  const loadCards = useCallback(async (deckId: string) => {
    setCardsLoading(true);
    try {
      const res = await fetchDeckCards(deckId);
      setCards(res.cards || []);
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.detail || "Could not load cards.",
      );
    } finally {
      setCardsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedDeckId) {
      void loadCards(selectedDeckId);
    } else {
      setCards([]);
    }
  }, [selectedDeckId, loadCards]);

  const selectedDeck = decks.find((d) => d.deck_id === selectedDeckId) || null;

  const handleCreateDeck = async () => {
    const name = newDeckName.trim();
    if (!name || creating) return;
    setCreating(true);
    setErrorMsg("");
    try {
      const deck = await createDeck({
        name,
        description: newDeckDescription.trim(),
      });
      const list = await loadDecks();
      setSelectedDeckId(deck.deck_id);
      setNewDeckName("");
      setNewDeckDescription("");
      void list;
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.detail || "Could not create deck.",
      );
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteDeck = async (deckId: string) => {
    try {
      await deleteDeck(deckId);
      const list = await loadDecks();
      if (selectedDeckId === deckId) {
        setSelectedDeckId(list.length > 0 ? list[0].deck_id : null);
      }
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.detail || "Could not delete deck.",
      );
    }
  };

  const handleImport = async (files: FileList | null) => {
    if (!files || !files.length || importing) return;
    const file = files[0];
    if (file.size > MAX_IMPORT_BYTES) {
      setErrorMsg("File exceeds the 50 MB import cap.");
      return;
    }
    setImporting(true);
    setErrorMsg("");
    setImportProgress(0);
    try {
      const res = await importDeck(file, setImportProgress);
      await loadDecks();
      setSelectedDeckId(res.deck_id);
      setErrorMsg(
        res.errors.length > 0
          ? `Imported ${res.imported} card(s). ${res.errors.join("; ")}`
          : "",
      );
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail;
      if (status === 413) {
        setErrorMsg("File exceeds the 50 MB import cap.");
      } else {
        setErrorMsg(
          typeof detail === "string"
            ? detail
            : "Import failed. Please retry.",
        );
      }
    } finally {
      setImporting(false);
      setImportProgress(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={pageTransition}
      className="max-w-[1200px] mx-auto px-4 sm:px-6 py-8"
    >
      <div className="udemy-card p-6">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold">Flashcard Decks</h1>
            <p className="text-sm text-udemy-text-muted mt-1">
              Build decks, generate cards from topics or documents,
              study with FSRS scheduling, and import/export Anki
              <code className="mx-1 rounded bg-udemy-bg px-1">.apkg</code>
              files.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => void loadDecks()}
              disabled={loading}
              className="btn-secondary flex items-center gap-2"
            >
              <RefreshCw
                className={`w-4 h-4 ${loading ? "animate-spin" : ""}`}
              />
              Refresh
            </button>
            <label
              className={`btn-secondary flex items-center gap-2 ${
                importing ? "opacity-60 pointer-events-none" : ""
              }`}
            >
              <FileUp className="w-4 h-4" />
              {importing ? `Importing… ${importProgress}%` : "Import .apkg"}
              <input
                ref={fileInputRef}
                type="file"
                accept=".apkg"
                className="hidden"
                onChange={(event) =>
                  void handleImport(event.target.files)
                }
              />
            </label>
          </div>
        </div>

        {disabledNotice && (
          <div className="mt-4 rounded-lg border border-udemy-border bg-udemy-bg p-3 text-sm text-udemy-text-muted">
            {disabledNotice}
          </div>
        )}

        {errorMsg && (
          <div className="mt-4 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Create deck */}
        <div className="mt-5 rounded-xl border border-udemy-border bg-udemy-bg p-4">
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              value={newDeckName}
              onChange={(event) => setNewDeckName(event.target.value)}
              placeholder="New deck name"
              maxLength={200}
              className="flex-1 rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm"
              onKeyDown={(event) => {
                if (event.key === "Enter") void handleCreateDeck();
              }}
            />
            <input
              value={newDeckDescription}
              onChange={(event) => setNewDeckDescription(event.target.value)}
              placeholder="Description (optional)"
              maxLength={2000}
              className="flex-1 rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm"
            />
            <button
              onClick={() => void handleCreateDeck()}
              disabled={creating || !newDeckName.trim()}
              className="btn-primary flex items-center justify-center gap-2"
            >
              {creating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Plus className="w-4 h-4" />
              )}
              Create deck
            </button>
          </div>
        </div>

        {loading && decks.length === 0 ? (
          <div className="mt-6 space-y-3">
            {[0, 1, 2].map((i) => (
              <div key={i} className="skeleton h-20 w-full" />
            ))}
          </div>
        ) : decks.length === 0 ? (
          <div className="mt-6 rounded-lg border border-udemy-border bg-udemy-bg p-6 text-center">
            <Layers className="w-8 h-8 text-udemy-text-muted mx-auto mb-2" />
            <p className="text-sm text-udemy-text-muted">
              No decks yet. Create a deck or import an Anki{" "}
              <code className="rounded bg-white px-1">.apkg</code> file.
            </p>
          </div>
        ) : (
          <div className="mt-6 grid gap-3 md:grid-cols-2">
            {decks.map((deck) => (
              <div
                key={deck.deck_id}
                className={`rounded-xl border bg-white p-4 transition-colors ${
                  selectedDeckId === deck.deck_id
                    ? "border-udemy-purple ring-1 ring-udemy-purple"
                    : "border-udemy-border"
                }`}
              >
                <button
                  type="button"
                  onClick={() => setSelectedDeckId(deck.deck_id)}
                  className="w-full text-left"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="font-medium text-[15px] truncate">
                      {deck.name}
                    </span>
                    <span className="text-xs text-udemy-text-muted shrink-0">
                      {deck.card_count} card{deck.card_count === 1 ? "" : "s"}
                      {deck.due_count > 0 && (
                        <span className="text-udemy-purple">
                          {" "}
                          · {deck.due_count} due
                        </span>
                      )}
                    </span>
                  </div>
                  {deck.description && (
                    <p className="text-xs text-udemy-text-muted mt-1 line-clamp-2">
                      {deck.description}
                    </p>
                  )}
                  <p className="text-[11px] text-udemy-text-muted mt-2">
                    Updated {formatDateTime(deck.updated_at)}
                  </p>
                </button>
                <div className="mt-3 flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => setSelectedDeckId(deck.deck_id)}
                    className="text-xs font-medium text-udemy-purple hover:underline"
                  >
                    Open
                  </button>
                  <span className="text-udemy-text-muted">·</span>
                  <button
                    type="button"
                    onClick={() => void handleDeleteDeck(deck.deck_id)}
                    className="text-xs font-medium text-red-600 hover:underline"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {selectedDeck && (
        <DeckDetail
          deck={selectedDeck}
          cards={cards}
          cardsLoading={cardsLoading}
          onCardsChanged={() => void loadCards(selectedDeck.deck_id)}
          onDeckChanged={() => void loadDecks()}
        />
      )}
    </motion.div>
  );
}

function DeckDetail({
  deck,
  cards,
  cardsLoading,
  onCardsChanged,
  onDeckChanged,
}: {
  deck: DeckResponse;
  cards: FlashcardResponse[];
  cardsLoading: boolean;
  onCardsChanged: () => void;
  onDeckChanged: () => void;
}) {
  const [tab, setTab] = useState<"cards" | "study" | "generate" | "duplicates">("cards");
  const [errorMsg, setErrorMsg] = useState("");
  const [notice, setNotice] = useState("");
  const [adding, setAdding] = useState(false);
  const [front, setFront] = useState("");
  const [back, setBack] = useState("");
  const [tags, setTags] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editFront, setEditFront] = useState("");
  const [editBack, setEditBack] = useState("");
  const [editTags, setEditTags] = useState("");

  const handleAddCard = async () => {
    const f = front.trim();
    const b = back.trim();
    if (!f || !b || adding) return;
    setAdding(true);
    setErrorMsg("");
    try {
      await addDeckCard(deck.deck_id, {
        front: f,
        back: b,
        tags: tags.split(",").map((t) => t.trim()).filter(Boolean),
      });
      setFront("");
      setBack("");
      setTags("");
      onCardsChanged();
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.detail || "Could not add card.",
      );
    } finally {
      setAdding(false);
    }
  };

  const handleDeleteCard = async (cardId: string) => {
    try {
      await deleteDeckCard(cardId);
      onCardsChanged();
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.detail || "Could not delete card.",
      );
    }
  };

  const startEdit = (card: FlashcardResponse) => {
    setEditingId(card.card_id);
    setEditFront(card.front);
    setEditBack(card.back);
    setEditTags(card.tags.join(", "));
  };

  const cancelEdit = () => {
    setEditingId(null);
  };

  const saveEdit = async (cardId: string) => {
    const f = editFront.trim();
    const b = editBack.trim();
    if (!f || !b) return;
    try {
      await updateDeckCard(cardId, {
        front: f,
        back: b,
        tags: editTags.split(",").map((t) => t.trim()).filter(Boolean),
      });
      setEditingId(null);
      onCardsChanged();
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.detail || "Could not update card.",
      );
    }
  };

  const handleExport = () => {
    window.open(exportDeckApkgUrl(deck.deck_id), "_blank");
  };

  return (
    <div className="max-w-[1200px] mx-auto px-4 sm:px-6 mt-6">
      <div className="udemy-card p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h2 className="text-xl font-bold flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-udemy-purple" />
              {deck.name}
            </h2>
            <p className="text-sm text-udemy-text-muted mt-1">
              {deck.card_count} cards · {deck.due_count} due
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => setTab("cards")}
              className={`btn-secondary ${tab === "cards" ? "ring-1 ring-udemy-purple" : ""}`}
            >
              Cards
            </button>
            <button
              onClick={() => setTab("study")}
              className={`btn-secondary ${tab === "study" ? "ring-1 ring-udemy-purple" : ""}`}
            >
              Study
            </button>
            <button
              onClick={() => setTab("generate")}
              className={`btn-secondary ${tab === "generate" ? "ring-1 ring-udemy-purple" : ""}`}
            >
              <Sparkles className="w-4 h-4" />
              Generate
            </button>
            <button
              onClick={() => setTab("duplicates")}
              className={`btn-secondary ${tab === "duplicates" ? "ring-1 ring-udemy-purple" : ""}`}
            >
              <Search className="w-4 h-4" />
              Duplicates
            </button>
            <button onClick={handleExport} className="btn-secondary flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export .apkg
            </button>
          </div>
        </div>

        {errorMsg && (
          <div className="mt-4 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}
        {notice && (
          <div className="mt-4 rounded-lg border border-udemy-border bg-udemy-bg p-3 text-sm text-udemy-text-muted flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{notice}</span>
          </div>
        )}

        {tab === "cards" && (
          <div className="mt-5">
            <div className="rounded-xl border border-udemy-border bg-udemy-bg p-4">
              <div className="flex flex-col gap-2">
                <input
                  value={front}
                  onChange={(event) => setFront(event.target.value)}
                  placeholder="Front (question)"
                  maxLength={8000}
                  className="rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm"
                />
                <textarea
                  value={back}
                  onChange={(event) => setBack(event.target.value)}
                  placeholder="Back (answer)"
                  maxLength={16000}
                  rows={3}
                  className="rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm"
                />
                <div className="flex flex-col sm:flex-row gap-2">
                  <input
                    value={tags}
                    onChange={(event) => setTags(event.target.value)}
                    placeholder="Tags (comma separated, optional)"
                    className="flex-1 rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm"
                  />
                  <button
                    onClick={() => void handleAddCard()}
                    disabled={adding || !front.trim() || !back.trim()}
                    className="btn-primary flex items-center justify-center gap-2"
                  >
                    {adding ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Plus className="w-4 h-4" />
                    )}
                    Add card
                  </button>
                </div>
              </div>
            </div>

            {cardsLoading ? (
              <div className="mt-4 space-y-3">
                {[0, 1].map((i) => (
                  <div key={i} className="skeleton h-24 w-full" />
                ))}
              </div>
            ) : cards.length === 0 ? (
              <div className="mt-4 rounded-lg border border-udemy-border bg-udemy-bg p-6 text-center">
                <p className="text-sm text-udemy-text-muted">
                  No cards yet. Add one above, generate from a topic, or
                  import an Anki file.
                </p>
              </div>
            ) : (
              <div className="mt-4 space-y-3">
                {cards.map((card) => (
                  <div
                    key={card.card_id}
                    className="rounded-xl border border-udemy-border bg-white p-4"
                  >
                    {editingId === card.card_id ? (
                      <div className="flex flex-col gap-2">
                        <input
                          value={editFront}
                          onChange={(event) => setEditFront(event.target.value)}
                          maxLength={8000}
                          className="rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm"
                        />
                        <textarea
                          value={editBack}
                          onChange={(event) => setEditBack(event.target.value)}
                          maxLength={16000}
                          rows={3}
                          className="rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm"
                        />
                        <input
                          value={editTags}
                          onChange={(event) => setEditTags(event.target.value)}
                          placeholder="Tags (comma separated)"
                          className="rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm"
                        />
                        <div className="flex gap-2">
                          <button
                            onClick={() => void saveEdit(card.card_id)}
                            className="btn-primary"
                          >
                            Save
                          </button>
                          <button onClick={cancelEdit} className="btn-secondary">
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <p className="text-sm font-medium">{card.front}</p>
                        <p className="text-sm text-udemy-text-muted mt-1 whitespace-pre-wrap">
                          {card.back}
                        </p>
                        <div className="mt-2 flex flex-wrap items-center gap-2">
                          {card.tags.map((tag) => (
                            <span
                              key={tag}
                              className="rounded-full bg-udemy-purple/10 px-2 py-0.5 text-xs text-udemy-purple"
                            >
                              {tag}
                            </span>
                          ))}
                          {card.fsrs && (
                            <span className="text-[11px] text-udemy-text-muted">
                              {card.fsrs.state} · reps {card.fsrs.reps} · due{" "}
                              {formatDateTime(card.fsrs.due_at)}
                            </span>
                          )}
                        </div>
                        <div className="mt-3 flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => startEdit(card)}
                            className="p-2 rounded-lg text-udemy-text-muted hover:text-udemy-purple hover:bg-udemy-purple/5 transition-colors"
                            aria-label={`Edit card ${card.card_id}`}
                          >
                            <Pencil className="w-4 h-4" />
                          </button>
                          <button
                            type="button"
                            onClick={() => void handleDeleteCard(card.card_id)}
                            className="p-2 rounded-lg text-udemy-text-muted hover:text-red-600 hover:bg-red-50 transition-colors"
                            aria-label={`Delete card ${card.card_id}`}
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === "study" && (
          <StudyPanel
            deckId={deck.deck_id}
            onReviewed={() => {
              onCardsChanged();
              onDeckChanged();
            }}
            setError={setErrorMsg}
          />
        )}

        {tab === "generate" && (
          <GeneratePanel
            deckId={deck.deck_id}
            onGenerated={() => {
              onCardsChanged();
              onDeckChanged();
            }}
            setError={setErrorMsg}
            setNotice={setNotice}
          />
        )}

        {tab === "duplicates" && (
          <DuplicatesPanel
            deckId={deck.deck_id}
            setError={setErrorMsg}
          />
        )}
      </div>
    </div>
  );
}

function StudyPanel({
  deckId,
  onReviewed,
  setError,
}: {
  deckId: string;
  onReviewed: () => void;
  setError: (message: string) => void;
}) {
  const [session, setSession] = useState<FlashcardResponse[]>([]);
  const [totalDue, setTotalDue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [index, setIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [startedAt, setStartedAt] = useState(0);
  const [submitting, setSubmitting] = useState(false);

  const loadSession = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetchStudySession(deckId, 50);
      setSession(res.cards || []);
      setTotalDue(res.total_due || 0);
      setIndex(0);
      setRevealed(false);
      setStartedAt(Date.now());
    } catch (err: any) {
      setError(
        err?.response?.data?.detail || "Could not load study session.",
      );
    } finally {
      setLoading(false);
    }
  }, [deckId, setError]);

  useEffect(() => {
    void loadSession();
  }, [loadSession]);

  const current = session[index];

  const handleRate = async (rating: LearningReviewRating) => {
    if (!current || submitting) return;
    setSubmitting(true);
    try {
      await reviewDeckCard(deckId, {
        card_id: current.card_id,
        rating,
        response_time_ms: Date.now() - startedAt,
      });
      const next = index + 1;
      if (next >= session.length) {
        setSession([]);
        setTotalDue(0);
        setIndex(0);
        setRevealed(false);
        onReviewed();
      } else {
        setIndex(next);
        setRevealed(false);
        setStartedAt(Date.now());
      }
    } catch (err: any) {
      setError(
        err?.response?.data?.detail || "Review failed. Please retry.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div className="mt-5 flex items-center justify-center py-10">
        <Loader2 className="w-6 h-6 text-udemy-purple animate-spin" />
      </div>
    );
  }

  if (session.length === 0) {
    return (
      <div className="mt-5 rounded-lg border border-udemy-border bg-udemy-bg p-6 text-center">
        <CheckCircle2 className="w-8 h-8 text-udemy-success mx-auto mb-2" />
        <p className="text-sm font-medium">
          {totalDue === 0 ? "All caught up!" : "No due cards."}
        </p>
        <p className="text-xs text-udemy-text-muted mt-1">
          {totalDue === 0
            ? "No cards are due for review right now."
            : "Cards become due based on FSRS scheduling."}
        </p>
        <button onClick={() => void loadSession()} className="btn-secondary mt-4">
          <RefreshCw className="w-4 h-4" />
          Check again
        </button>
      </div>
    );
  }

  return (
    <div className="mt-5">
      <div className="flex items-center justify-between text-xs text-udemy-text-muted">
        <span>
          Card {index + 1} of {session.length}
        </span>
        <span>{totalDue} due in deck</span>
      </div>
      <div className="mt-3 rounded-xl border border-udemy-border bg-white p-6 min-h-[180px]">
        <p className="text-base font-medium">{current.front}</p>
        {revealed ? (
          <p className="text-sm text-udemy-text-muted mt-4 whitespace-pre-wrap border-t border-udemy-border pt-4">
            {current.back}
          </p>
        ) : (
          <button
            onClick={() => setRevealed(true)}
            className="btn-secondary mt-6"
          >
            Show answer
          </button>
        )}
      </div>
      {revealed && (
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2">
          {RATINGS.map((rating) => (
            <button
              key={rating.value}
              onClick={() => void handleRate(rating.value)}
              disabled={submitting}
              className={`rounded-xl border px-3 py-2 text-sm font-medium transition-colors disabled:opacity-50 ${
                rating.value === "again"
                  ? "border-red-300 bg-red-50 text-red-700 hover:bg-red-100"
                  : rating.value === "hard"
                    ? "border-amber-300 bg-amber-50 text-amber-700 hover:bg-amber-100"
                    : rating.value === "good"
                      ? "border-udemy-purple bg-udemy-purple/10 text-udemy-purple hover:bg-udemy-purple/20"
                      : "border-green-300 bg-green-50 text-green-700 hover:bg-green-100"
              }`}
            >
              {submitting ? (
                <Loader2 className="w-4 h-4 animate-spin mx-auto" />
              ) : (
                rating.label
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function GeneratePanel({
  deckId,
  onGenerated,
  setError,
  setNotice,
}: {
  deckId: string;
  onGenerated: () => void;
  setError: (message: string) => void;
  setNotice: (message: string) => void;
}) {
  const [sourceType, setSourceType] = useState<"topic" | "section" | "document">("topic");
  const [topicId, setTopicId] = useState("");
  const [sectionTitle, setSectionTitle] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [count, setCount] = useState(10);
  const [submitting, setSubmitting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState("");
  const [generated, setGenerated] = useState<GeneratedCardItem[]>([]);
  const [polling, setPolling] = useState(false);
  const pollRef = useRef<number | null>(null);

  const stopPolling = () => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
    setPolling(false);
  };

  useEffect(() => {
    return stopPolling;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pollJob = useCallback(
    async (id: string) => {
      try {
        const res = await fetchCardGenerationJob(id);
        setJobStatus(res.status);
        if (res.status === "done") {
          stopPolling();
          setGenerated(res.cards || []);
          setNotice(
            `Generated ${res.cards.length} card(s)` +
              (res.malformed_items_dropped > 0
                ? ` (${res.malformed_items_dropped} malformed dropped)`
                : "") +
              ".",
          );
        } else if (res.status === "failed") {
          stopPolling();
          setError(res.error || "Generation failed.");
        }
      } catch (err: any) {
        stopPolling();
        setError(
          err?.response?.data?.detail || "Could not check generation status.",
        );
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const handleGenerate = async () => {
    if (submitting) return;
    if (sourceType !== "document" && !topicId.trim()) {
      setError("Enter a topic id (or pick a topic from Topics).");
      return;
    }
    if (sourceType === "document" && !documentId.trim()) {
      setError("Enter a document id (from Documents).");
      return;
    }
    if (sourceType === "section" && !sectionTitle.trim()) {
      setError("Enter a section title.");
      return;
    }
    setSubmitting(true);
    setError("");
    setNotice("");
    setGenerated([]);
    try {
      const res = await generateDeckCards(deckId, {
        source_type: sourceType,
        topic_id: topicId.trim() || undefined,
        section_title: sectionTitle.trim() || undefined,
        document_id: documentId.trim() || undefined,
        count,
      });
      setJobId(res.job_id);
      setJobStatus("queued");
      setPolling(true);
      pollRef.current = window.setInterval(() => {
        void pollJob(res.job_id);
      }, 1000);
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail;
      if (status === 404) {
        setError(
          typeof detail === "string"
            ? detail
            : "Topic or document not found.",
        );
      } else {
        setError(
          typeof detail === "string"
            ? detail
            : "Generation request failed.",
        );
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mt-5">
      <div className="rounded-xl border border-udemy-border bg-udemy-bg p-4">
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-2">
            {(["topic", "section", "document"] as const).map((type) => (
              <button
                key={type}
                type="button"
                onClick={() => setSourceType(type)}
                className={`rounded-lg border px-3 py-1.5 text-sm font-medium capitalize transition-colors ${
                  sourceType === type
                    ? "border-udemy-purple bg-udemy-purple/10 text-udemy-purple"
                    : "border-udemy-border bg-white text-udemy-text-muted hover:border-udemy-purple/50"
                }`}
              >
                {type}
              </button>
            ))}
          </div>

          {sourceType !== "document" && (
            <input
              value={topicId}
              onChange={(event) => setTopicId(event.target.value)}
              placeholder="Topic id (e.g. 01-backend-fundamentals-and-http)"
              maxLength={200}
              className="rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm"
            />
          )}
          {sourceType === "section" && (
            <input
              value={sectionTitle}
              onChange={(event) => setSectionTitle(event.target.value)}
              placeholder="Section title"
              maxLength={400}
              className="rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm"
            />
          )}
          {sourceType === "document" && (
            <input
              value={documentId}
              onChange={(event) => setDocumentId(event.target.value)}
              placeholder="Document id (from Documents page)"
              maxLength={200}
              className="rounded-lg border border-udemy-border bg-white px-3 py-2 text-sm"
            />
          )}
          <div className="flex flex-col sm:flex-row gap-2">
            <label className="flex items-center gap-2 text-sm text-udemy-text-muted">
              Count
              <input
                type="number"
                min={1}
                max={50}
                value={count}
                onChange={(event) =>
                  setCount(
                    Math.max(
                      1,
                      Math.min(50, Number(event.target.value) || 1),
                    ),
                  )
                }
                className="w-20 rounded-lg border border-udemy-border bg-white px-2 py-1.5 text-sm"
              />
            </label>
            <button
              onClick={() => void handleGenerate()}
              disabled={submitting || polling}
              className="btn-primary flex items-center justify-center gap-2"
            >
              {submitting || polling ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              Generate cards
            </button>
          </div>
        </div>
      </div>

      {jobId && (
        <div className="mt-4 rounded-lg border border-udemy-border bg-white p-3 text-sm">
          <div className="flex items-center gap-2">
            {polling ? (
              <Loader2 className="w-4 h-4 animate-spin text-udemy-purple" />
            ) : (
              <CheckCircle2 className="w-4 h-4 text-udemy-success" />
            )}
            <span className="font-medium">Job {jobId}</span>
            <span className="text-udemy-text-muted">status: {jobStatus}</span>
          </div>
        </div>
      )}

      {generated.length > 0 && (
        <div className="mt-4 space-y-3">
          <p className="text-sm font-medium">
            {generated.length} generated card(s) — review and add them from
            the Cards tab.
          </p>
          {generated.map((card) => (
            <div
              key={card.card_id}
              className="rounded-xl border border-udemy-border bg-white p-4"
            >
              <p className="text-sm font-medium">{card.front}</p>
              <p className="text-sm text-udemy-text-muted mt-1 whitespace-pre-wrap">
                {card.back}
              </p>
              {card.source_section && (
                <p className="text-[11px] text-udemy-text-muted mt-2">
                  Source: {card.source_section}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function DuplicatesPanel({
  deckId,
  setError,
}: {
  deckId: string;
  setError: (message: string) => void;
}) {
  const [threshold, setThreshold] = useState(0.85);
  const [loading, setLoading] = useState(false);
  const [pairs, setPairs] = useState<
    Array<{
      card_a: { card_id: string; deck_id: string; front: string };
      card_b: { card_id: string; deck_id: string; front: string };
      similarity: number;
      jaccard: number;
      cosine: number | null;
    }>
  >([]);
  const [ran, setRan] = useState(false);

  const handleFind = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await findDuplicateCards({ deck_id: deckId, threshold });
      setPairs(res.pairs || []);
      setRan(true);
    } catch (err: any) {
      setError(
        err?.response?.data?.detail || "Could not search for duplicates.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-5">
      <div className="rounded-xl border border-udemy-border bg-udemy-bg p-4">
        <div className="flex flex-col sm:flex-row gap-2 items-start sm:items-end">
          <label className="flex flex-col gap-1 text-sm text-udemy-text-muted">
            Similarity threshold
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={threshold}
              onChange={(event) =>
                setThreshold(
                  Math.max(
                    0,
                    Math.min(1, Number(event.target.value) || 0),
                  ),
                )
              }
              className="w-24 rounded-lg border border-udemy-border bg-white px-2 py-1.5 text-sm"
            />
          </label>
          <button
            onClick={() => void handleFind()}
            disabled={loading}
            className="btn-primary flex items-center gap-2"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Search className="w-4 h-4" />
            )}
            Find duplicates
          </button>
        </div>
        <p className="text-xs text-udemy-text-muted mt-2">
          Jaccard similarity over normalized card text
          {pairs.length > 0 && pairs[0].cosine !== null
            ? " refined by cosine similarity"
            : ""}
          .
        </p>
      </div>

      {ran && pairs.length === 0 && (
        <div className="mt-4 rounded-lg border border-udemy-border bg-udemy-bg p-4 text-sm text-udemy-text-muted">
          No duplicate pairs above {threshold}.
        </div>
      )}

      {pairs.length > 0 && (
        <div className="mt-4 space-y-3">
          {pairs.map((pair, index) => (
            <div
              key={`${pair.card_a.card_id}-${pair.card_b.card_id}-${index}`}
              className="rounded-xl border border-amber-300 bg-amber-50 p-4"
            >
              <div className="flex items-center justify-between text-xs text-amber-700">
                <span className="font-medium">
                  {Math.round(pair.similarity * 100)}% similar
                </span>
                <span>
                  jaccard {pair.jaccard.toFixed(2)}
                  {pair.cosine !== null
                    ? ` · cosine ${pair.cosine.toFixed(2)}`
                    : ""}
                </span>
              </div>
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <div className="rounded-lg bg-white p-3">
                  <p className="text-[11px] text-udemy-text-muted">
                    {pair.card_a.card_id}
                  </p>
                  <p className="text-sm mt-1">{pair.card_a.front}</p>
                </div>
                <div className="rounded-lg bg-white p-3">
                  <p className="text-[11px] text-udemy-text-muted">
                    {pair.card_b.card_id}
                  </p>
                  <p className="text-sm mt-1">{pair.card_b.front}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
