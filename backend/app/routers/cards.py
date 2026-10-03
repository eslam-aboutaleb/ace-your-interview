"""Flashcard decks router — decks, cards, Anki interop, study sessions.

Endpoints (STUDY_ENABLE_FLASHCARDS_V1):

- ``POST   /api/decks``                       create deck
- ``GET    /api/decks``                       list decks with counts
- ``POST   /api/decks/import``                import ``.apkg``
- ``GET    /api/decks/generate/jobs/{job_id}`` poll a generation job
- ``PUT    /api/decks/{deck_id}``             update deck
- ``DELETE /api/decks/{deck_id}``             delete deck
- ``POST   /api/decks/{deck_id}/cards``       add card (links FSRS row)
- ``GET    /api/decks/{deck_id}/cards``       list cards (``?due_only=true``)
- ``GET    /api/decks/{deck_id}/study-session`` due cards for study
- ``POST   /api/decks/{deck_id}/review``      FSRS review of a deck card
- ``POST   /api/decks/{deck_id}/generate``    async LLM card generation
- ``GET    /api/decks/{deck_id}/export``      export ``.apkg``
- ``PUT    /api/cards/{card_id}``             update card
- ``DELETE /api/cards/{card_id}``             delete card
- ``POST   /api/cards/find-duplicates``       similar card pairs
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    CardCreateRequest,
    CardListResponse,
    CardResponse,
    CardUpdateRequest,
    DeckCreateRequest,
    DeckListResponse,
    DeckResponse,
    DeckReviewRequest,
    DeckReviewResponse,
    DeckUpdateRequest,
    FindDuplicatesRequest,
    FindDuplicatesResponse,
    FSRSReviewLogResponse,
    GenerateCardsJobResponse,
    GenerateCardsRequest,
    GenerateCardsResponse,
    GeneratedCardItem,
    ImportDeckResponse,
    StudySessionResponse,
    TopicDetail,
)
from app.services.anki_archive import (
    AnkiArchiveError,
    MAX_IMPORT_BYTES,
    extract_media,
    read_apkg,
    write_apkg,
)
from app.services.card_dedup import (
    DEFAULT_THRESHOLD,
    find_duplicate_pairs,
)
from app.services.card_store import CardStore
from app.services.doc_parser import DocParser
from app.services.document_store import (
    DocumentStore,
    vector_index_status,
)
from app.services.embedding_client import EmbeddingClient
from app.services.learning_store import LearningStore
from app.services.llm_client import LLMClient
from app.services.mcp_gateway import MCPGateway
from app.services.question_generator import QuestionGenerator
from app.services.topic_catalog import (
    PROBLEM_SOLVING_DEFAULT_LANGUAGE,
    is_problem_solving_topic,
)

logger = logging.getLogger(__name__)

decks_router = APIRouter(prefix="/api/decks", tags=["decks"])
cards_router = APIRouter(prefix="/api/cards", tags=["cards"])

_card_store: CardStore | None = None
_learning_store: LearningStore | None = None
_llm_client: LLMClient | None = None
_parser: DocParser | None = None
_mcp_gateway: MCPGateway | None = None
_document_store: DocumentStore | None = None
_embedding_client: EmbeddingClient | None = None

#: In-memory generation job registry (single-instance deployment).
_generation_jobs: dict[str, dict[str, Any]] = {}
_JOB_TTL_SECONDS = 3600.0

#: Extracted Anki media root (plan decision).
ANKI_MEDIA_ROOT = os.path.join("data", "anki_media")


def init(
    card_store: CardStore,
    learning_store: LearningStore,
    llm_client: LLMClient,
    parser: DocParser,
    mcp_gateway: MCPGateway | None = None,
    document_store: DocumentStore | None = None,
    embedding_client: EmbeddingClient | None = None,
):
    global _card_store, _learning_store, _llm_client, _parser
    global _mcp_gateway, _document_store, _embedding_client
    _card_store = card_store
    _learning_store = learning_store
    _llm_client = llm_client
    _parser = parser
    _mcp_gateway = mcp_gateway
    _document_store = document_store
    _embedding_client = embedding_client


def _ensure_enabled() -> None:
    if not get_settings().enable_flashcards_v1:
        raise HTTPException(
            status_code=503,
            detail="Flashcards disabled (STUDY_ENABLE_FLASHCARDS_V1=false)",
        )


def _require_card_store() -> CardStore:
    if _card_store is None:
        raise HTTPException(status_code=503, detail="Card store not initialised")
    return _card_store


def _require_learning_store() -> LearningStore:
    if _learning_store is None:
        raise HTTPException(
            status_code=503, detail="Learning store not initialised"
        )
    return _learning_store


def _require_generation_services() -> tuple[LLMClient, DocParser, LearningStore]:
    if _llm_client is None or _parser is None or _learning_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return _llm_client, _parser, _learning_store


# ── Topic / document content resolution ─────────
async def _resolve_topic_detail(
    *,
    topic_id: str,
    user_id: str,
) -> TopicDetail | None:
    """Resolve the full topic detail (with sections) for a user."""
    _, parser, store = _require_generation_services()
    if is_problem_solving_topic(topic_id):
        settings = await store.run_async(
            store.resolve_topic_ai_settings,
            user_id=user_id,
            topic_id=topic_id,
            topic_detail=None,
        )
        preferred = (
            settings.get("preferred_language", "")
            or PROBLEM_SOLVING_DEFAULT_LANGUAGE
        )
        detail = await store.run_async(
            store.resolve_problem_solving_topic_detail,
            user_id=user_id,
            preferred_language=preferred,
        )
        return TopicDetail(**detail)

    static_topic = parser.get_topic(topic_id)
    if static_topic:
        return static_topic

    custom_topic = await store.run_async(
        store.get_custom_topic,
        user_id=user_id,
        topic_id=topic_id,
    )
    if custom_topic:
        return TopicDetail(**custom_topic)
    return None


def _section_content(topic: TopicDetail, section_title: str) -> str:
    for section in topic.sections:
        if str(section.get("heading", "")).strip() == section_title.strip():
            return str(section.get("content", ""))
    return ""


async def _resolve_document_content(
    *,
    document_id: str,
    user_id: str,
) -> tuple[str, str] | None:
    """Return ``(title, content)`` for an uploaded document."""
    if _document_store is None:
        return None
    document = await _document_store.run_async(
        _document_store.get_document,
        user_id=user_id,
        document_id=document_id,
    )
    if document is None:
        return None
    chunks = await _document_store.run_async(
        _document_store.get_document_chunks,
        user_id=user_id,
        document_id=document_id,
    )
    content = "\n\n".join(
        str(chunk.get("content", "")) for chunk in chunks
    )
    return str(document.get("title", "")), content


# ── Generation job registry ─────────────────────
def _prune_jobs() -> None:
    cutoff = time.time() - _JOB_TTL_SECONDS
    stale = [
        job_id
        for job_id, job in _generation_jobs.items()
        if job.get("finished_at", 0) and job["finished_at"] < cutoff
    ]
    for job_id in stale:
        _generation_jobs.pop(job_id, None)
    while len(_generation_jobs) > 200:
        oldest = min(
            _generation_jobs,
            key=lambda j: _generation_jobs[j].get("created_at", 0),
        )
        _generation_jobs.pop(oldest, None)


async def _run_generation_job(
    job_id: str,
    generator: QuestionGenerator,
    kwargs: dict[str, Any],
) -> None:
    job = _generation_jobs.get(job_id)
    if job is None:
        return
    job["status"] = "running"

    try:
        cards, stats = await generator.generate_cards(**kwargs)
        job["status"] = "done"
        job["cards"] = cards
        job["retries_used"] = int(stats.get("retries_used", 0) or 0)
        job["malformed_items_dropped"] = int(
            stats.get("malformed_items_dropped", 0) or 0
        )
    except Exception as e:  # noqa: BLE001 - captured into the job
        logger.exception("Card generation job %s failed", job_id)
        job["status"] = "failed"
        job["error"] = str(e).strip() or "generation failed"
    finally:
        job["finished_at"] = time.time()


# ── Deck CRUD ───────────────────────────────────
@decks_router.post("", response_model=DeckResponse, status_code=201)
async def create_deck(
    body: DeckCreateRequest, user: dict = Depends(require_auth)
):
    _ensure_enabled()
    store = _require_card_store()
    deck = await store.run_async(
        store.create_deck,
        user_id=user["user"],
        name=body.name,
        description=body.description,
    )
    return DeckResponse(**deck)


@decks_router.get("", response_model=DeckListResponse)
async def list_decks(user: dict = Depends(require_auth)):
    _ensure_enabled()
    store = _require_card_store()
    decks = await store.run_async(store.list_decks, user_id=user["user"])
    return DeckListResponse(decks=[DeckResponse(**d) for d in decks])


@decks_router.post("/import", response_model=ImportDeckResponse, status_code=201)
async def import_deck(
    file: UploadFile = File(...), user: dict = Depends(require_auth)
):
    """Import an Anki ``.apkg`` into a new deck."""
    _ensure_enabled()
    store = _require_card_store()
    filename = (file.filename or "deck.apkg").strip() or "deck.apkg"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_IMPORT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File exceeds the "
                f"{MAX_IMPORT_BYTES // (1024 * 1024)} MB import cap"
            ),
        )
    try:
        parsed = read_apkg(data)
    except AnkiArchiveError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    errors = [str(e) for e in parsed.get("errors", [])]
    apkg_decks = parsed.get("decks", [])
    if not apkg_decks:
        raise HTTPException(
            status_code=400,
            detail="No cards found in the .apkg file",
        )

    first = apkg_decks[0]
    deck_name = str(first.get("name", "") or "").strip() or filename
    deck = await store.run_async(
        store.create_deck,
        user_id=user["user"],
        name=deck_name,
        description=f"Imported from {filename}",
    )

    cards: list[dict[str, Any]] = []
    review_logs: dict[str, list[dict[str, Any]]] = {}
    index = 0
    for apkg_deck in apkg_decks:
        for card in apkg_deck.get("cards", []):
            cards.append(
                {
                    "front": str(card.get("front", "")),
                    "back": str(card.get("back", "")),
                    "tags": [str(t) for t in (card.get("tags") or [])],
                }
            )
            logs = card.get("review_logs") or []
            if logs:
                review_logs[str(index)] = logs
            index += 1

    result = await store.run_async(
        store.import_cards,
        user_id=user["user"],
        deck_id=deck["deck_id"],
        cards=cards,
        review_logs=review_logs,
    )

    # Media extraction is best-effort: failures are captured,
    # never fail the import.
    try:
        media_dir = os.path.join(ANKI_MEDIA_ROOT, str(user["user"]))
        extract_media(data, media_dir)
    except AnkiArchiveError as e:
        errors.append(f"media extraction failed: {e}")
    except OSError as e:
        errors.append(f"media extraction failed: {e}")

    return ImportDeckResponse(
        deck_id=deck["deck_id"],
        deck_name=deck["name"],
        imported=len(result["imported"]),
        skipped_duplicates=int(result["skipped_duplicates"]),
        errors=errors,
    )


@decks_router.get(
    "/generate/jobs/{job_id}", response_model=GenerateCardsJobResponse
)
async def get_generation_job(
    job_id: str, user: dict = Depends(require_auth)
):
    _ensure_enabled()
    job = _generation_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("user_id") != user["user"]:
        raise HTTPException(status_code=404, detail="job not found")
    return GenerateCardsJobResponse(
        job_id=job_id,
        status=str(job.get("status", "queued")),
        cards=[
            GeneratedCardItem(**card) for card in (job.get("cards") or [])
        ],
        retries_used=int(job.get("retries_used", 0) or 0),
        malformed_items_dropped=int(
            job.get("malformed_items_dropped", 0) or 0
        ),
        error=str(job.get("error", "") or ""),
    )


@decks_router.get("/{deck_id}", response_model=DeckResponse)
async def get_deck(deck_id: str, user: dict = Depends(require_auth)):
    _ensure_enabled()
    store = _require_card_store()
    deck = await store.run_async(
        store.get_deck, user_id=user["user"], deck_id=deck_id
    )
    if deck is None:
        raise HTTPException(status_code=404, detail="deck not found")
    return DeckResponse(**deck)


@decks_router.put("/{deck_id}", response_model=DeckResponse)
async def update_deck(
    deck_id: str,
    body: DeckUpdateRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    store = _require_card_store()
    deck = await store.run_async(
        store.update_deck,
        user_id=user["user"],
        deck_id=deck_id,
        name=body.name,
        description=body.description,
    )
    if deck is None:
        raise HTTPException(status_code=404, detail="deck not found")
    return DeckResponse(**deck)


@decks_router.delete("/{deck_id}", status_code=204)
async def delete_deck(
    deck_id: str, user: dict = Depends(require_auth)
):
    _ensure_enabled()
    store = _require_card_store()
    deleted = await store.run_async(
        store.delete_deck,
        user_id=user["user"],
        deck_id=deck_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="deck not found")
    return None


@decks_router.post(
    "/{deck_id}/cards", response_model=CardResponse, status_code=201
)
async def add_card(
    deck_id: str,
    body: CardCreateRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    store = _require_card_store()
    deck = await store.run_async(
        store.get_deck, user_id=user["user"], deck_id=deck_id
    )
    if deck is None:
        raise HTTPException(status_code=404, detail="deck not found")
    card = await store.run_async(
        store.create_card,
        user_id=user["user"],
        deck_id=deck_id,
        front=body.front,
        back=body.back,
        tags=body.tags,
    )
    return CardResponse(**card)


@decks_router.get("/{deck_id}/cards", response_model=CardListResponse)
async def list_deck_cards(
    deck_id: str,
    due_only: bool = Query(default=False),
    limit: int = Query(default=500, ge=1, le=5000),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    store = _require_card_store()
    deck = await store.run_async(
        store.get_deck, user_id=user["user"], deck_id=deck_id
    )
    if deck is None:
        raise HTTPException(status_code=404, detail="deck not found")
    cards = await store.run_async(
        store.list_cards,
        user_id=user["user"],
        deck_id=deck_id,
        due_only=due_only,
        limit=limit,
    )
    return CardListResponse(
        cards=[CardResponse(**c) for c in cards],
        total=len(cards),
    )


@decks_router.get(
    "/{deck_id}/study-session", response_model=StudySessionResponse
)
async def study_session(
    deck_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    store = _require_card_store()
    deck = await store.run_async(
        store.get_deck, user_id=user["user"], deck_id=deck_id
    )
    if deck is None:
        raise HTTPException(status_code=404, detail="deck not found")
    cards = await store.run_async(
        store.due_cards_for_deck,
        user_id=user["user"],
        deck_id=deck_id,
        limit=limit,
    )
    return StudySessionResponse(
        deck_id=deck_id,
        cards=[CardResponse(**c) for c in cards],
        total_due=len(cards),
    )


@decks_router.post(
    "/{deck_id}/review", response_model=DeckReviewResponse, status_code=201
)
async def review_deck_card(
    deck_id: str,
    body: DeckReviewRequest,
    user: dict = Depends(require_auth),
):
    """Apply an FSRS review to a deck card (delegates to the
    FSRS review machinery from the spaced-repetition plan)."""
    _ensure_enabled()
    store = _require_card_store()
    learning = _require_learning_store()
    card = await store.run_async(
        store.get_card, user_id=user["user"], card_id=body.card_id
    )
    if card is None or card["deck_id"] != deck_id:
        raise HTTPException(status_code=404, detail="card not found")
    result = await learning.run_async(
        learning.record_review,
        user_id=user["user"],
        card_id=card["fsrs_card_id"],
        topic_id=deck_id,
        rating=body.rating.value,
        response_time_ms=body.response_time_ms,
        source_type="flashcard",
    )
    updated = await store.run_async(
        store.get_card, user_id=user["user"], card_id=body.card_id
    )
    assert updated is not None
    return DeckReviewResponse(
        card=CardResponse(**updated),
        log=FSRSReviewLogResponse(**result["log"]),
        leech=bool(result["leech"]),
    )


@decks_router.post(
    "/{deck_id}/generate", response_model=GenerateCardsResponse, status_code=202
)
async def generate_cards(
    deck_id: str,
    body: GenerateCardsRequest,
    user: dict = Depends(require_auth),
):
    """Queue async LLM card generation for a deck."""
    _ensure_enabled()
    store = _require_card_store()
    llm_client, _parser, learning = _require_generation_services()
    deck = await store.run_async(
        store.get_deck, user_id=user["user"], deck_id=deck_id
    )
    if deck is None:
        raise HTTPException(status_code=404, detail="deck not found")

    source_type = body.source_type.value
    topic_title = ""
    doc_content = ""
    section_title = body.section_title
    section_content: str | None = None

    if source_type == "document":
        if not body.document_id:
            raise HTTPException(
                status_code=400, detail="document_id is required for document source"
            )
        resolved = await _resolve_document_content(
            document_id=body.document_id,
            user_id=user["user"],
        )
        if resolved is None:
            raise HTTPException(status_code=404, detail="document not found")
        topic_title, doc_content = resolved
        section_title = None
    else:
        if not body.topic_id:
            raise HTTPException(
                status_code=400, detail="topic_id is required for topic/section source"
            )
        topic = await _resolve_topic_detail(
            topic_id=body.topic_id,
            user_id=user["user"],
        )
        if topic is None:
            raise HTTPException(status_code=404, detail="topic not found")
        topic_title = topic.title
        doc_content = topic.raw_content
        if source_type == "section":
            if not section_title:
                raise HTTPException(
                    status_code=400,
                    detail="section_title is required for section source",
                )
            section_content = _section_content(topic, section_title)
            if not section_content:
                raise HTTPException(
                    status_code=404, detail="section not found in topic"
                )

    # Existing card fronts feed the no-repeat uniqueness block.
    existing = await store.run_async(
        store.list_card_texts, user_id=user["user"], deck_id=deck_id
    )
    existing_fronts = [str(c["front"]) for c in existing]

    job_id = uuid.uuid4().hex
    _prune_jobs()
    _generation_jobs[job_id] = {
        "job_id": job_id,
        "user_id": user["user"],
        "deck_id": deck_id,
        "status": "queued",
        "cards": [],
        "retries_used": 0,
        "malformed_items_dropped": 0,
        "error": "",
        "created_at": time.time(),
        "finished_at": 0.0,
    }
    generator = QuestionGenerator(llm_client, mcp_gateway=_mcp_gateway)
    kwargs: dict[str, Any] = {
        "topic_id": body.topic_id or f"deck:{deck_id}",
        "topic_title": topic_title or deck["name"],
        "doc_content": doc_content,
        "count": body.count,
        "section_title": section_title,
        "section_content": section_content,
        "response_detail": "very_detailed",
        "existing_cards": existing_fronts,
        "llm_config": body.llm_config,
        "user_identity": user,
    }
    asyncio.create_task(_run_generation_job(job_id, generator, kwargs))
    return GenerateCardsResponse(job_id=job_id, status="queued")


@decks_router.get("/{deck_id}/export")
async def export_deck(
    deck_id: str, user: dict = Depends(require_auth)
):
    """Export a deck as an Anki ``.apkg`` download."""
    _ensure_enabled()
    store = _require_card_store()
    deck = await store.run_async(
        store.get_deck, user_id=user["user"], deck_id=deck_id
    )
    if deck is None:
        raise HTTPException(status_code=404, detail="deck not found")
    cards = await store.run_async(
        store.list_cards,
        user_id=user["user"],
        deck_id=deck_id,
        due_only=False,
        limit=5000,
    )
    fsrs_ids = [str(c["fsrs_card_id"]) for c in cards]
    logs = await store.run_async(
        store.review_logs_for_cards,
        user_id=user["user"],
        fsrs_card_ids=fsrs_ids,
    )
    export_cards = [
        {
            "card_id": str(c["fsrs_card_id"]),
            "front": str(c["front"]),
            "back": str(c["back"]),
            "tags": list(c["tags"]),
            "fsrs": c.get("fsrs") or {},
        }
        for c in cards
    ]
    data = write_apkg(deck["name"], export_cards, logs)
    safe_name = "".join(
        ch if ch.isalnum() or ch in "-_ " else "_" for ch in deck["name"]
    ).strip() or "deck"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/apkg",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.apkg"',
            "Content-Length": str(len(data)),
        },
    )


# ── Card CRUD + duplicate detection ─────────────
@cards_router.put("/{card_id}", response_model=CardResponse)
async def update_card(
    card_id: str,
    body: CardUpdateRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    store = _require_card_store()
    card = await store.run_async(
        store.update_card,
        user_id=user["user"],
        card_id=card_id,
        front=body.front,
        back=body.back,
        tags=body.tags,
    )
    if card is None:
        raise HTTPException(status_code=404, detail="card not found")
    return CardResponse(**card)


@cards_router.delete("/{card_id}", status_code=204)
async def delete_card(
    card_id: str, user: dict = Depends(require_auth)
):
    _ensure_enabled()
    store = _require_card_store()
    deleted = await store.run_async(
        store.delete_card, user_id=user["user"], card_id=card_id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="card not found")
    return None


@cards_router.post("/find-duplicates", response_model=FindDuplicatesResponse)
async def find_duplicates(
    body: FindDuplicatesRequest, user: dict = Depends(require_auth)
):
    """Similar card pairs: Jaccard on normalized text, with
    cosine on embeddings when the vector index is available."""
    _ensure_enabled()
    store = _require_card_store()
    cards = await store.run_async(
        store.list_card_texts,
        user_id=user["user"],
        deck_id=body.deck_id,
    )
    threshold = (
        body.threshold if body.threshold is not None else DEFAULT_THRESHOLD
    )

    vectors: list[list[float]] | None = None
    # The cosine accelerator only runs when sqlite-vec is
    # actually available; otherwise the Jaccard baseline
    # is the complete result.
    if (
        _document_store is not None
        and _embedding_client is not None
        and vector_index_status() == "available"
        and cards
    ):
        try:
            texts = [
                f"{c['front']}\n{c['back']}" for c in cards
            ]
            vectors = await _embedding_client.embed(
                texts, user_identity=user
            )
        except Exception as e:  # noqa: BLE001 - accelerator is optional
            logger.warning("Embedding accelerator unavailable: %s", e)
            vectors = None

    pairs = find_duplicate_pairs(cards, threshold=threshold, vectors=vectors)
    return FindDuplicatesResponse(pairs=pairs, threshold=threshold)
