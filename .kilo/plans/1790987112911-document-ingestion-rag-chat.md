# Plan: Document Ingestion + Citation-Grounded RAG Chat

## Goal
Let users upload study documents (PDF, TXT, MD, DOCX), chunk and embed them into sqlite-vec inside the existing SQLite DB, and ask questions grounded in those documents with verbatim citations — extending chat beyond highlighted handbook words.

## Context
- Chat today (`backend/app/routers/chat.py`) is scoped to a highlighted word + section context from the static curriculum.
- Decisions resolved: **sqlite-vec** (no new infrastructure), **provider-configurable embeddings** (OpenAI `text-embedding-3-small` default, Google `gemini-embedding` second), reusing the `LLMClient` credential policy (personal key or backend-funded assignment).
- sqlite-vec is a loadable extension (`vec0`); the Docker image must ship it; a pure-Python cosine fallback runs when the extension is unavailable.

## Decisions (resolved)
- Vector index: sqlite-vec `vec0` virtual table; dimension stored per-provider in a meta table; index rebuilt on provider/dimension change.
- Chunking: recursive character splitter, 800 chars, 80-char overlap, max 2000 chunks/document.
- Retrieval: top-k = 5, cosine via vec0 `distance` (default `vec_distance_cos`).
- Citations: every answer includes `citations: [{document_id, chunk_index, quote}]`; quotes verified as exact substrings of stored chunks, one repair retry on mismatch.

## Data model
```sql
CREATE TABLE IF NOT EXISTS documents (
    user_id TEXT NOT NULL, document_id TEXT NOT NULL,
    filename TEXT NOT NULL, mime_type TEXT NOT NULL,
    title TEXT NOT NULL, status TEXT NOT NULL
        CHECK(status IN ('uploaded','processing','ready','failed')),
    error TEXT NOT NULL DEFAULT '', chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, document_id)
);
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL, document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(user_id, document_id, chunk_index);
-- created at runtime when sqlite-vec is available:
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[<dim>]);
CREATE TABLE IF NOT EXISTS vec_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);  -- dimension, provider, index_ready
```

## API changes
- `POST /api/documents` (multipart file) → 202 `{document_id, status}`; caps: 25 MB, allowlist pdf/txt/md/docx/markdown.
- `GET /api/documents` → list with status/chunk_count.
- `GET /api/documents/{id}` → metadata + processing error.
- `DELETE /api/documents/{id}` → removes chunks + vectors.
- `POST /api/chat/ask` `{message, document_ids?: [], topic_id?, conversation_id?}` → `{answer, citations[], conversation_id}`. Retrieval scope: given documents, else all user documents. Reuses `untrusted_block` fencing from `prompt_blocks.py` for chunk context; system prompt requires verbatim quotes.
- Existing `POST /api/chat/follow-up` unchanged.

## Implementation tasks (ordered)
1. `backend/app/services/embedding_client.py`: provider-configurable embeddings via the `LLMClient` credential path; `embed(texts: list[str]) -> list[list[float]]`; batch size 32; dimension per provider.
2. `backend/app/services/document_store.py`: DDL, upload status machine, chunk insert, vec index create/rebuild, `search(user_id, query_embedding, k)` using vec0 `distance` with a pure-Python cosine fallback (`_vec_available` probe at startup, logged).
3. `backend/app/services/document_pipeline.py`: extract (pypdf for PDF, mammoth for DOCX, direct read for txt/md) → chunk → embed → persist; runs on the learning-store executor thread; updates document status; failure → `status=failed` with error.
4. `backend/app/services/rag_service.py`: retrieve → fence chunks → call LLM with citation instruction → verify quotes are exact substrings → repair once → return citations.
5. Docker: add sqlite-vec to `backend/Dockerfile` (apt `sqlite-vec` package or build from source); extend `/api/llm/health` with `vector_index: available|fallback`.
6. New `backend/app/routers/documents.py` + chat extension in `routers/chat.py`; register both in `main.py`.
7. Schemas: `DocumentUploadResponse`, `DocumentDetail`, `ChatAskRequest/Response`, `Citation`.
8. Frontend: new `frontend/src/pages/DocumentsPage.tsx` (route `/documents`) with upload/drag-drop and status list; `ChatAskPanel.tsx` embedded in topic study + standalone; citation chips that highlight the source chunk.
9. Tests: `test_embedding_client.py` (mock provider), `test_document_store.py` (chunking, search recall on fixture corpus, fallback cosine parity), `test_rag_service.py` (citation precision, injection-in-document neutralization, quote-verification retry), `test_documents_router.py` (caps, bad mime → 415).

## Failure modes
- sqlite-vec missing → fallback mode (correct but O(n) scan); surfaced in the health endpoint; never silent.
- Scanned/image-only PDFs → extraction yields empty text → document marked failed with a clear error ("scanned PDFs not supported in v1").
- Embedding cost → per-user daily document cap (default 20, `STUDY_DOCUMENTS_DAILY_CAP`), enforced via the `feature_events` table.
- Prompt injection inside documents → chunks fenced with `untrusted_block`; system prompt includes `UNTRUSTED_CLAUSE`.
- Dimension change on provider switch → `vec_meta` check → rebuild index (re-embed all chunks) before serving search.
- Large docs → 2000-chunk cap, 25 MB cap, 200k-char hard guard consistent with chat's `HARD_INPUT_GUARD_CHARS`.

## Rollout
1. Ship behind `STUDY_ENABLE_RAG_V1` (default `false` until sqlite-vec is verified in the Docker image).
2. Verify the extension loads under `docker compose up`; flip the flag.
3. Monitor embedding spend via the existing LLM budget machinery.

## Validation
- Recall@5 ≥ 0.8 on a 50-document fixture corpus with known answers.
- Citation precision: 100% of returned quotes are exact substrings of stored chunks.
- Injection test: a document containing "ignore previous instructions" cannot alter answer behavior.
- Fallback parity: identical top-5 results in vec0 and cosine modes on the fixture corpus.
