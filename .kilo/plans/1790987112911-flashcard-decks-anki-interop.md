# Plan: Flashcard Decks + Anki Interop

## Goal
Add persistent flashcard decks, LLM card generation from topics/sections/documents, semantic duplicate detection, and full Anki `.apkg` import/export with FSRS scheduling preserved.

## Context
- No card library exists today; question progress is ephemeral per `question_id`.
- **Depends on** `1790987112911-fsrs-spaced-repetition-engine.md` (cards use `fsrs_cards` with `source_type='flashcard'`).
- **Depends on** `1790987112911-document-ingestion-rag-chat.md` for semantic duplicate detection (Jaccard fallback when sqlite-vec is unavailable).
- `.apkg` = zip containing `collection.anki2` (SQLite with `col`/`notes`/`cards`/`revlog` tables) plus a `media` JSON file and media blobs.

## Decisions (resolved)
- Import **and** export both shipped.
- Anki parsing: custom minimal reader/writer over the documented `.apkg` format (avoids heavy deps; `apyrus` is an acceptable alternative if it supports writing).
- Media: extracted media stored under `data/anki_media/{user_id}/`; caps: 50 MB per import, 2000 cards per import, 5000 zip entries, compressed-ratio ≤ 100x (zip-bomb guard).
- Decks are user-scoped; no sharing in v1.

## Data model
```sql
CREATE TABLE IF NOT EXISTS decks (
    user_id TEXT NOT NULL, deck_id TEXT NOT NULL,
    name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, deck_id)
);
CREATE TABLE IF NOT EXISTS cards (
    user_id TEXT NOT NULL, card_id TEXT NOT NULL,
    deck_id TEXT NOT NULL, front TEXT NOT NULL, back TEXT NOT NULL,
    source_ref TEXT NOT NULL DEFAULT '',        -- topic_id/section or document chunk id
    tags_json TEXT NOT NULL DEFAULT '[]',
    fsrs_card_id TEXT NOT NULL,                 -- links to fsrs_cards.card_id
    suspended INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, card_id)
);
CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(user_id, deck_id);
```

## API changes
- `POST /api/decks` `{name, description}` → 201 deck
- `GET /api/decks` → list with card counts + due counts
- `PUT /api/decks/{id}`, `DELETE /api/decks/{id}`
- `POST /api/decks/{id}/cards` `{front, back, tags[]}` → 201 (creates linked `fsrs_cards` row)
- `GET /api/decks/{id}/cards?due_only=true`
- `PUT /api/cards/{id}`, `DELETE /api/cards/{id}`
- `POST /api/decks/{id}/generate` `{source_type: topic|section|document, topic_id?, section_title?, document_id?, count}` → 202 `{job_id}`; async LLM generation reusing the v2 grounding machinery (`source_quote`/`source_section` validation + retry/repair) with card-format output
- `POST /api/decks/import` (multipart `.apkg`) → 201 `{deck_id, imported, skipped_duplicates, errors[]}`
- `GET /api/decks/{id}/export` → `application/apkg` download (streamed zip built from cards + `fsrs_review_log`)
- `POST /api/cards/find-duplicates` `{deck_id?, threshold}` → similar card pairs (Jaccard on normalized text; cosine on embeddings when sqlite-vec is available)
- `GET /api/decks/{id}/study-session` → due cards (join `fsrs_cards`); `POST /api/decks/{id}/review` `{card_id, rating, response_time_ms}` → delegates to the FSRS review endpoint (plan 2)

## Implementation tasks (ordered)
1. DDL + `backend/app/services/card_store.py` following the `run_async` pattern; register in `main.py` lifespan; new `backend/app/routers/cards.py`.
2. `backend/app/services/anki_archive.py`: read `.apkg` zip → parse `collection.anki2` (`notes`, `cards`, `revlog`) → map to app cards; write `.apkg` (build minimal `collection.anki2` with `col`/`notes`/`cards`/`revlog` from `fsrs_review_log`, zip with `media` JSON). Target Anki 2.1 format (`usn=-1`, `mod` timestamps); verify against real exports via fixtures in `backend/tests/fixtures/`.
3. Import endpoint: size/count caps, zip-bomb guard, duplicate detection on import (skip + count, never fail the whole file).
4. Export endpoint: `StreamingResponse` zip.
5. Card generation: extend `backend/app/services/question_generator.py` with `generate_cards()` producing `{front, back, source_quote, source_section}`; same validation/retry loop as v2 questions; `count` capped at 50; reuses LLM budget machinery from `llm_client.py`.
6. Duplicate detection: `backend/app/services/card_dedup.py` — Jaccard baseline; optional vec cosine when plan 3 has landed.
7. Frontend: new `frontend/src/pages/DecksPage.tsx` (route `/decks`), deck detail with study mode (flip animation via framer-motion), import/export buttons, generate-cards dialog; add routes to `App.tsx` and nav to `Header.tsx`.
8. Tests: `test_card_store.py`, `test_anki_archive.py` (round-trip with a fixture `.apkg` built in-test), `test_card_generation.py` (grounding validation), `test_cards_router.py`.

## Failure modes
- Malformed `.apkg` → per-file error capture, partial import with `errors[]`, never a bare 500.
- Zip bomb → entry-count/ratio caps enforced before extraction.
- Media bloat → 50 MB cap; media stored outside SQLite.
- FSRS linkage broken on import → every imported card gets a fresh `fsrs_cards` row; Anki revlog imported as `fsrs_review_log` history only (does not drive scheduling).
- Generation cost → `count` cap 50 + existing LLM budget machinery.

## Rollout
1. Ship behind `STUDY_ENABLE_FLASHCARDS_V1` (default `true`).
2. Import/export validated against real Anki exports (fixtures committed under `backend/tests/fixtures/`).
3. GA with decks nav entry.

## Validation
- Round-trip: export a deck → import into a fresh account → card count and front/back text identical.
- Import a real-world `.apkg` with media → media files present under `data/anki_media/`.
- `pytest tests/test_card_store.py tests/test_anki_archive.py tests/test_cards_router.py`
