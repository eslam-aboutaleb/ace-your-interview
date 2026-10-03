"""Mock interview router — personalized text-first interview sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    CreateInterviewSessionRequest,
    InterviewAnswerAndNextResponse,
    InterviewQuestionResponse,
    InterviewReportResponse,
    InterviewSessionResponse,
    InterviewSessionsListResponse,
    InterviewStatsResponse,
    InterviewTrendsResponse,
    InterviewTurn,
    InterviewTurnResponse,
    NextInterviewQuestionRequest,
    RubricScore,
    SubmitInterviewAnswerRequest,
)
from app.services.doc_parser import DocParser
from app.services.interview_generator import InterviewGenerator
from app.services.interview_store import (
    InterviewStore,
    TurnConflictError,
    memory_answer_excerpt_limit,
    memory_summary_limit,
    memory_verbatim_turn_count,
)
from app.services.learning_store import LearningStore
from app.services.llm_client import LLMClient
from app.services.llm_policy import (
    LLMServiceApprovalRequiredError,
    PersonalCredentialRequiredError,
    StudyAppLLMNotAssignedError,
    policy_error_detail,
)
from app.services.mcp_gateway import MCPGateway
from app.services.prompt_blocks import UNTRUSTED_CLAUSE, untrusted_block

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/interview-sessions", tags=["interview-sessions"])

_store: InterviewStore | None = None
_generator: InterviewGenerator | None = None
_parser: DocParser | None = None
_learning_store: LearningStore | None = None

#: Marker line every interview-memory prompt starts with. The prompt asks the
#: model to revise the previous note, exactly as ``progress_summarizer`` does for
#: study progress, instead of appending to a capped string.
_MEMORY_DIGEST_MARKER = "You maintain one compact memory note per mock-interview session"
_MEMORY_DIGEST_SYSTEM = (
    "You are an interview memory summariser. You maintain one short, factual memory "
    "note per mock-interview session so the next interviewer turn can build on it."
)
_MEMORY_DIGEST_MAX_WORDS = 180
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+\-]*\s*\n(?P<body>[\s\S]*?)\n?```$")


def init(
    llm_client: LLMClient,
    parser: DocParser,
    learning_store: LearningStore,
    db_path: str,
    mcp_gateway: MCPGateway | None = None,
):
    global _store, _generator, _parser, _learning_store
    _store = InterviewStore(db_path)
    _generator = InterviewGenerator(llm_client, parser, mcp_gateway=mcp_gateway)
    _parser = parser
    _learning_store = learning_store


def _ensure_enabled() -> None:
    if not get_settings().enable_mock_interview_v1:
        raise HTTPException(status_code=404, detail="Mock interviews disabled")


def _ensure_ready() -> None:
    if _store is None or _generator is None or _parser is None:
        raise HTTPException(status_code=503, detail="Mock interview services not initialised")


def _to_session_response(user_id: str, session_id: str) -> InterviewSessionResponse:
    assert _store is not None
    session = _store.get_session(user_id=user_id, session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")

    turns = _store.get_turns(user_id=user_id, session_id=session_id)
    parsed_turns = [InterviewTurn(**t) for t in turns]
    return InterviewSessionResponse(session=session, turns=parsed_turns)


def _track_topic_id(track: str) -> str:
    if _parser is None:
        return ""
    topics = _parser.list_topics(track=track)
    return topics[0].id if topics else ""


def _turn_conflict_detail() -> dict[str, str]:
    return {
        "code": TurnConflictError.code,
        "message": (
            "This interview turn was already submitted. Refresh the session and answer the current question."
        ),
    }


def _record_attempt_blocked(eval_payload: dict[str, Any]) -> bool:
    """A degraded evaluation has no rubric, so it must not reach the SRS store."""
    return bool(eval_payload.get("degraded"))


def _cancel_task(task: "asyncio.Task[Any] | None") -> None:
    """Drop an in-flight provider call when a stream goes away.

    A client disconnect closes the generator mid-LLM-call; without this the task
    keeps running (and keeps spending) against a response nobody reads.
    """
    if task is not None and not task.done():
        task.cancel()


def _pending_answer(session_id: str, user: dict) -> tuple[dict[str, Any], str, int]:
    """Shared pre-flight for the answer endpoints: session, question, turn index.

    Raises the same 404/400/409 trio for every endpoint that records a turn, so
    the combined endpoint cannot drift from ``/answer``.
    """
    assert _store is not None
    session = _store.get_session_context(user_id=user["user"], session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Interview session already completed")
    question = (session.get("current_question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="No active question. Generate next question first.")
    turn_index = int(session["turns_completed"]) + 1
    # A repeat submission of an already recorded turn is rejected before the body
    # starts, so it gets a real HTTP status.
    if _store.turn_exists(user_id=user["user"], session_id=session_id, turn_index=turn_index):
        raise HTTPException(status_code=409, detail=_turn_conflict_detail())
    return session, question, turn_index


# ── Interview memory (Stage 3.2) ────────────────────────────
def _memory_prompt_block(session: dict[str, Any]) -> str:
    """Digest plus the newest verbatim turns, capped at the one shared limit.

    The digest gets at most half the budget so the rolling window can never be
    squeezed out by a long note, and the window is emitted newest-first so the
    generator's own prefix slice keeps the most recent turns.
    """
    limit = memory_summary_limit()
    digest = str(session.get("memory_digest") or session.get("memory_summary") or "").strip()
    lines: list[str] = []
    if digest:
        lines.append(f"Session digest: {digest[: limit // 2]}")
    window = session.get("memory_window") or []
    size = memory_verbatim_turn_count()
    # Newest first, and never `[-0:]`, which would keep the whole window when
    # the rolling window is disabled.
    entries = list(reversed(list(window)[-size:])) if size else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        excerpt = str(entry.get("answer_excerpt") or "").strip()
        question = str(entry.get("question") or "").strip()
        if not (question or excerpt):
            continue
        lines.append(f"Turn {entry.get('turn_index', '?')} Q: {question} | A: {excerpt}")
    return "\n".join(lines)[:limit]


def _prompt_session(session: dict[str, Any]) -> dict[str, Any]:
    """The session as the prompt builder should see it.

    ``interview_generator`` reads ``memory_summary`` for both the question and
    the evaluation prompt, so the composed digest + window is handed over under
    that key instead of the bare digest.
    """
    return {**session, "memory_summary": _memory_prompt_block(session)}


def _digest_turn_line(turn: dict[str, Any], *, excerpt_chars: int) -> str:
    strengths = "; ".join(str(x).strip() for x in (turn.get("strengths") or [])[:2] if str(x).strip())
    improvements = "; ".join(str(x).strip() for x in (turn.get("improvements") or [])[:2] if str(x).strip())
    return (
        f"Turn {turn.get('turn_index', '?')} | "
        f"Q: {str(turn.get('question') or '').strip()[:220]} | "
        f"A: {str(turn.get('user_answer') or '').strip()[:excerpt_chars]}"
        + (f" | strengths: {strengths}" if strengths else "")
        + (f" | gaps: {improvements}" if improvements else "")
    )


def _deterministic_digest(turns: list[dict[str, Any]]) -> str:
    """Offline compaction: newest turns first, oldest folded into a marker.

    Mirrors ``progress_summarizer.build_fallback_summary``: used whenever the
    provider path is unavailable, so the note stays useful instead of the turn
    being silently dropped.
    """
    limit = memory_summary_limit()
    excerpt_chars = memory_answer_excerpt_limit()
    lines = [_digest_turn_line(turn, excerpt_chars=excerpt_chars) for turn in turns]
    lines = [line for line in lines if line.strip()]
    if not lines:
        return ""

    kept: list[str] = []
    used = 0
    folded = 0
    for line in reversed(lines):
        cost = len(line) + 1
        if kept and used + cost > max(0, limit - 120):
            folded += 1
            continue
        kept.append(line)
        used += cost
    kept.reverse()
    header = f"Interview memory through turn {turns[-1].get('turn_index', len(turns))}."
    if folded:
        header += f" {folded} earlier turn(s) compacted into this note."
    return "\n".join([header, *kept])[:limit]


def _clean_digest_text(text: str) -> str:
    """Strip a wrapping code fence, mirroring ``progress_summarizer``'s cleaner."""
    cleaned = str(text or "").strip()
    match = _FENCE_RE.match(cleaned)
    if match:
        cleaned = match.group("body").strip()
    return cleaned[: memory_summary_limit()]


def _build_digest_prompt(
    *,
    previous_digest: str,
    turn: dict[str, Any],
) -> str:
    excerpt_chars = memory_answer_excerpt_limit()
    return f"""{_MEMORY_DIGEST_MARKER}.

An interviewer and an evaluator read this note before the next turn, so it must
carry forward every fact that still changes how they probe the candidate.

Previous memory note (update it — never restart from scratch):
{untrusted_block("previous_digest", previous_digest, memory_summary_limit() // 2) or "(no previous note — this is the first one)"}

Newest recorded turn:
{untrusted_block("new_turn_question", str(turn.get("question") or ""), 400)}
{untrusted_block("new_turn_answer", str(turn.get("user_answer") or ""), excerpt_chars)}
Strengths this turn: {"; ".join(str(x) for x in (turn.get("strengths") or [])[:2])}
Improvements this turn: {"; ".join(str(x) for x in (turn.get("improvements") or [])[:2])}

Write the updated memory note now, at most {_MEMORY_DIGEST_MAX_WORDS} words.
Keep every still-useful earlier fact; fold the oldest detail away rather than
dropping a topic outright. Cover: what has been asked and answered so far; the
candidate's demonstrated strengths; the weakest areas so far; and anything the
next interviewer must probe. Plain prose or short bullets. No preamble, no
meta-commentary, no headings other than the topic title.

{UNTRUSTED_CLAUSE}"""


async def _refresh_interview_memory(
    *,
    user: dict[str, Any],
    session_id: str,
    turn_index: int,
    question: str,
    user_answer: str,
    eval_payload: dict[str, Any],
    degraded: bool,
    llm_config: Any = None,
) -> None:
    """Refresh the digest and the rolling window once per recorded turn.

    Never raises: the turn is already saved, so a memory failure must not fail
    the request. The window is appended unconditionally and the digest falls
    back to deterministic compaction whenever the provider path is unavailable.
    """
    assert _store is not None
    excerpt_chars = memory_answer_excerpt_limit()
    try:
        _store.append_memory_turn(
            user_id=user["user"],
            session_id=session_id,
            entry={
                "turn_index": turn_index,
                "question": question,
                "answer_excerpt": user_answer[:excerpt_chars],
                "strengths": eval_payload.get("strengths") or [],
                "improvements": eval_payload.get("improvements") or [],
            },
        )

        turns = _store.get_turns(user_id=user["user"], session_id=session_id)
        fallback = _deterministic_digest(turns)
        session = _store.get_session_context(user_id=user["user"], session_id=session_id) or {}
        previous = str(session.get("memory_digest") or session.get("memory_summary") or "").strip()

        digest = fallback
        # A degraded turn means the provider already failed for this turn; one
        # more call would only repeat the failure.
        if not degraded and _generator is not None:
            turn = {
                "turn_index": turn_index,
                "question": question,
                "user_answer": user_answer,
                "strengths": eval_payload.get("strengths") or [],
                "improvements": eval_payload.get("improvements") or [],
            }
            try:
                result = await _generator.llm.completion(
                    _build_digest_prompt(
                        previous_digest=str(session.get("memory_digest") or ""),
                        turn=turn,
                    ),
                    llm_config,
                    user_identity=user,
                    task="final",
                    system=_MEMORY_DIGEST_SYSTEM,
                    max_tokens_cap=800,
                )
                if result.get("success"):
                    cleaned = _clean_digest_text(result.get("analysis", ""))
                    if cleaned:
                        digest = cleaned
                    else:
                        logger.info(
                            "Interview memory digest returned empty analysis session=%s; using compaction",
                            session_id,
                        )
                else:
                    logger.info(
                        "Interview memory digest fell back session=%s code=%s error=%s",
                        session_id,
                        result.get("error_code", ""),
                        str(result.get("error", ""))[:200],
                    )
            except Exception:
                logger.exception(
                    "Interview memory digest generation failed session=%s; using compaction",
                    session_id,
                )

        _store.set_memory_digest(
            user_id=user["user"],
            session_id=session_id,
            digest=digest or previous,
        )
    except Exception:
        logger.exception("Interview memory refresh failed session=%s", session_id)


async def _finalise_turn(
    *,
    user: dict[str, Any],
    session: dict[str, Any],
    session_id: str,
    turn_index: int,
    question: str,
    body: SubmitInterviewAnswerRequest,
    eval_payload: dict[str, Any],
    degraded: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record the turn, refresh memory, feed the SRS store, build the report.

    Raises :class:`TurnConflictError` so each caller can map it onto its own
    transport (HTTP 409, or a ``turn_conflict`` NDJSON event).
    """
    assert _store is not None and _generator is not None
    turn_dict, updated_session = _store.record_turn(
        user_id=user["user"],
        session_id=session_id,
        turn_index=turn_index,
        question=question,
        user_answer=body.user_answer,
        rubric=eval_payload["rubric"],
        strengths=eval_payload["strengths"],
        improvements=eval_payload["improvements"],
        follow_up_note=eval_payload["follow_up_note"],
        response_time_ms=body.response_time_ms,
        degraded=degraded,
    )

    await _refresh_interview_memory(
        user=user,
        session_id=session_id,
        turn_index=turn_index,
        question=question,
        user_answer=body.user_answer,
        eval_payload=eval_payload,
        degraded=degraded,
        llm_config=body.llm_config,
    )

    if get_settings().enable_adaptive_learning and _learning_store is not None and not degraded:
        topic_id = _track_topic_id(session["track"])
        if topic_id:
            rubric = eval_payload["rubric"]
            overall = int(rubric.get("overall") or 60)
            confidence_signal = max(1, min(5, int(rubric.get("confidence_signal") or 3)))
            await _learning_store.run_async(
                _learning_store.record_attempt,
                user_id=user["user"],
                question_id=f"interview:{session_id}:{turn_index}",
                topic_id=topic_id,
                user_answer=body.user_answer,
                is_correct=overall >= 70,
                confidence=confidence_signal,
                response_time_ms=body.response_time_ms,
                mode="study",
            )

    if updated_session and updated_session["status"] == "completed":
        turns = _store.get_turns(user_id=user["user"], session_id=session_id)
        report = _generator.build_report(session=updated_session, turns=turns)
        _store.save_report(user_id=user["user"], session_id=session_id, report=report)

    session_out = _store.get_session(user_id=user["user"], session_id=session_id)
    if not session_out:
        raise RuntimeError("Session update failed")
    return turn_dict, session_out


def _turn_response(
    *,
    session_id: str,
    session_out: dict[str, Any],
    turn_dict: dict[str, Any],
    turn_index: int,
    question: str,
    body: SubmitInterviewAnswerRequest,
    eval_payload: dict[str, Any],
    degraded: bool,
) -> InterviewTurnResponse:
    return InterviewTurnResponse(
        session=session_out,
        turn=InterviewTurn(
            session_id=session_id,
            turn_index=turn_index,
            question=question,
            user_answer=body.user_answer,
            rubric=RubricScore(**eval_payload["rubric"]),
            strengths=eval_payload["strengths"],
            improvements=eval_payload["improvements"],
            follow_up_note=eval_payload["follow_up_note"],
            response_time_ms=max(0, body.response_time_ms),
            created_at=turn_dict["created_at"],
            degraded=degraded,
        ),
        report_ready=bool(session_out.get("report_ready") or session_out["status"] == "completed"),
        degraded=degraded,
    )


async def _generate_next_question(
    *,
    user: dict[str, Any],
    session_id: str,
    llm_config: Any,
) -> tuple[dict[str, Any] | None, str]:
    """Generate and persist the next question from post-turn state.

    Returns ``(question_response, status)`` where status is ``generated``,
    ``session_completed``, or ``generation_failed``. Never raises: by the time
    this runs the turn is already recorded, so a failure here must degrade to a
    partial answer instead of losing the turn.
    """
    assert _store is not None and _generator is not None
    session = _store.get_session_context(user_id=user["user"], session_id=session_id)
    if not session:
        return None, "generation_failed"
    if session["status"] != "active":
        return None, "session_completed"

    turns = _store.get_turns(user_id=user["user"], session_id=session_id)
    try:
        generated = await _generator.generate_question(
            session=_prompt_session(session),
            turns=turns,
            llm_config=llm_config,
            user_identity=user,
        )
        updated = _store.set_current_question(
            user_id=user["user"],
            session_id=session_id,
            question=generated["question"],
        )
        if not updated:
            return None, "generation_failed"
        return (
            InterviewQuestionResponse(
                session_id=session_id,
                turn_index=int(updated["turns_completed"]) + 1,
                question=generated["question"],
                competency_focus=generated.get("competency_focus", ""),
                expected_signals=generated.get("expected_signals", []),
            ),
            "generated",
        )
    except (
        LLMServiceApprovalRequiredError,
        StudyAppLLMNotAssignedError,
        PersonalCredentialRequiredError,
    ) as exc:
        detail = policy_error_detail(exc)
        logger.warning(
            "Interview next-question blocked by policy session=%s code=%s",
            session_id,
            detail["code"],
        )
        return None, "generation_failed"
    except Exception as exc:
        logger.warning(
            "Interview next-question generation failed session=%s: %s",
            session_id,
            str(exc)[:200],
        )
        return None, "generation_failed"


async def _progress_events(
    task: "asyncio.Task[Any]",
    *,
    stage: str,
    message: str,
    interval_seconds: float = 0.8,
) -> AsyncIterator[dict[str, Any]]:
    elapsed = 0.0
    while True:
        done, _ = await asyncio.wait({task}, timeout=interval_seconds)
        if task in done:
            break
        elapsed += interval_seconds
        yield {
            "type": "progress",
            "stage": stage,
            "message": message,
            "elapsed_seconds": round(elapsed, 1),
        }


@router.post("", response_model=InterviewSessionResponse)
async def create_session(
    body: CreateInterviewSessionRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None and _generator is not None

    session = _store.create_session(
        user_id=user["user"],
        track=body.track.value,
        level=body.level.value,
        interview_type=body.interview_type.value,
        turn_count=body.turn_count,
        target_role=body.target_role.strip(),
        interviewer_style=body.interviewer_style.value,
        feedback_mode=body.feedback_mode.value,
        job_description_text=body.job_description_text.strip(),
        resume_summary_text=body.resume_summary_text.strip(),
        focus_areas=[x.strip() for x in body.focus_areas if x.strip()],
    )

    context = _store.get_session_context(user_id=user["user"], session_id=session["session_id"])
    if not context:
        raise HTTPException(status_code=500, detail="Failed to create session context")

    first_q = await _generator.generate_question(
        session=_prompt_session(context),
        turns=[],
        llm_config=body.llm_config,
        user_identity=user,
    )
    _store.set_current_question(
        user_id=user["user"],
        session_id=session["session_id"],
        question=first_q["question"],
    )

    return _to_session_response(user["user"], session["session_id"])


@router.post("/stream")
async def create_session_stream(
    body: CreateInterviewSessionRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None and _generator is not None

    async def _event_stream():
        yield json.dumps(
            {
                "type": "start",
                "stage": "creating_session",
                "message": "Creating interview session.",
            }
        ) + "\n"
        first_q_task: "asyncio.Task[Any] | None" = None
        try:
            session = _store.create_session(
                user_id=user["user"],
                track=body.track.value,
                level=body.level.value,
                interview_type=body.interview_type.value,
                turn_count=body.turn_count,
                target_role=body.target_role.strip(),
                interviewer_style=body.interviewer_style.value,
                feedback_mode=body.feedback_mode.value,
                job_description_text=body.job_description_text.strip(),
                resume_summary_text=body.resume_summary_text.strip(),
                focus_areas=[x.strip() for x in body.focus_areas if x.strip()],
            )
            context = _store.get_session_context(
                user_id=user["user"],
                session_id=session["session_id"],
            )
            if not context:
                raise RuntimeError("Failed to create session context")

            first_q_task = asyncio.create_task(
                _generator.generate_question(
                    session=_prompt_session(context),
                    turns=[],
                    llm_config=body.llm_config,
                    user_identity=user,
                )
            )
            async for progress in _progress_events(
                first_q_task,
                stage="generating_question",
                message="Generating first interview question.",
            ):
                yield json.dumps(progress) + "\n"
            first_q = await first_q_task
            first_q_task = None
            _store.set_current_question(
                user_id=user["user"],
                session_id=session["session_id"],
                question=first_q["question"],
            )

            payload = _to_session_response(user["user"], session["session_id"]).model_dump(mode="json")
            yield json.dumps({"type": "done", **payload}) + "\n"
        except asyncio.CancelledError:
            logger.info("Interview session-create stream cancelled")
            raise
        except (
            LLMServiceApprovalRequiredError,
            StudyAppLLMNotAssignedError,
            PersonalCredentialRequiredError,
        ) as exc:
            detail = policy_error_detail(exc)
            yield json.dumps(
                {
                    "type": "error",
                    "code": detail["code"] or "generation_failed",
                    "message": detail["message"] or "Interview generation blocked by policy.",
                }
            ) + "\n"
        except Exception as exc:
            yield json.dumps(
                {
                    "type": "error",
                    "code": "generation_failed",
                    "message": str(exc).strip() or "Interview generation failed.",
                }
            ) + "\n"
        finally:
            _cancel_task(first_q_task)

    return StreamingResponse(
        _event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("", response_model=InterviewSessionsListResponse)
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None

    payload = _store.list_sessions(user_id=user["user"], limit=limit, offset=offset)
    return InterviewSessionsListResponse(**payload)


@router.get("/stats", response_model=InterviewStatsResponse)
async def interview_stats(user: dict = Depends(require_auth)):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None
    return InterviewStatsResponse(**_store.get_stats(user_id=user["user"]))


@router.get("/trends", response_model=InterviewTrendsResponse)
async def interview_trends(
    limit: int = Query(default=50, ge=1, le=200),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None
    return InterviewTrendsResponse(**_store.get_trends(user_id=user["user"], limit=limit))


@router.get("/{session_id}", response_model=InterviewSessionResponse)
async def get_session(session_id: str, user: dict = Depends(require_auth)):
    _ensure_enabled()
    _ensure_ready()
    return _to_session_response(user["user"], session_id)


@router.post("/{session_id}/answer", response_model=InterviewTurnResponse)
async def submit_answer(
    session_id: str,
    body: SubmitInterviewAnswerRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None and _generator is not None

    session, question, turn_index = _pending_answer(session_id, user)
    eval_payload = await _generator.evaluate_answer(
        session=_prompt_session(session),
        question=question,
        user_answer=body.user_answer,
        turn_index=turn_index,
        llm_config=body.llm_config,
        user_identity=user,
    )
    # Internal drift signal for the eval harness; never part of an API payload.
    eval_payload.pop("follow_up_note_repaired", None)
    degraded = _record_attempt_blocked(eval_payload)

    try:
        turn_dict, session_out = await _finalise_turn(
            user=user,
            session=session,
            session_id=session_id,
            turn_index=turn_index,
            question=question,
            body=body,
            eval_payload=eval_payload,
            degraded=degraded,
        )
    except TurnConflictError as exc:
        raise HTTPException(status_code=409, detail=_turn_conflict_detail()) from exc

    return _turn_response(
        session_id=session_id,
        session_out=session_out,
        turn_dict=turn_dict,
        turn_index=turn_index,
        question=question,
        body=body,
        eval_payload=eval_payload,
        degraded=degraded,
    )


@router.post("/{session_id}/answer/stream")
async def submit_answer_stream(
    session_id: str,
    body: SubmitInterviewAnswerRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None and _generator is not None

    # A repeat submission of an already recorded turn can be rejected before the
    # response body starts, so it gets a real HTTP status. The race that slips
    # past this check is caught by record_turn and reported as a `turn_conflict`
    # error event, because a streaming response can no longer change its status.
    session, question, stream_turn_index = _pending_answer(session_id, user)

    async def _event_stream():
        yield json.dumps(
            {
                "type": "start",
                "stage": "evaluating_answer",
                "session_id": session_id,
                "message": "Evaluating interview answer.",
            }
        ) + "\n"
        eval_task: "asyncio.Task[Any] | None" = None
        try:
            turn_index = stream_turn_index
            eval_task = asyncio.create_task(
                _generator.evaluate_answer(
                    session=_prompt_session(session),
                    question=question,
                    user_answer=body.user_answer,
                    turn_index=turn_index,
                    llm_config=body.llm_config,
                    user_identity=user,
                )
            )
            async for progress in _progress_events(
                eval_task,
                stage="evaluating_answer",
                message="Evaluating answer and preparing coaching feedback.",
            ):
                yield json.dumps(progress) + "\n"
            eval_payload = await eval_task
            eval_task = None
            # Internal drift signal for the eval harness; never part of the wire payload.
            eval_payload.pop("follow_up_note_repaired", None)
            degraded = _record_attempt_blocked(eval_payload)

            try:
                turn_dict, session_out = await _finalise_turn(
                    user=user,
                    session=session,
                    session_id=session_id,
                    turn_index=turn_index,
                    question=question,
                    body=body,
                    eval_payload=eval_payload,
                    degraded=degraded,
                )
            except TurnConflictError as exc:
                yield json.dumps(
                    {
                        "type": "error",
                        "code": TurnConflictError.code,
                        "message": str(exc),
                    }
                ) + "\n"
                return

            payload = _turn_response(
                session_id=session_id,
                session_out=session_out,
                turn_dict=turn_dict,
                turn_index=turn_index,
                question=question,
                body=body,
                eval_payload=eval_payload,
                degraded=degraded,
            ).model_dump(mode="json")
            yield json.dumps({"type": "done", **payload}) + "\n"
        except asyncio.CancelledError:
            # Client disconnect: the provider call is dropped in the finally below.
            logger.info("Interview answer stream cancelled session=%s", session_id)
            raise
        except (
            LLMServiceApprovalRequiredError,
            StudyAppLLMNotAssignedError,
            PersonalCredentialRequiredError,
        ) as exc:
            detail = policy_error_detail(exc)
            yield json.dumps(
                {
                    "type": "error",
                    "code": detail["code"] or "generation_failed",
                    "message": detail["message"] or "Interview evaluation blocked by policy.",
                }
            ) + "\n"
        except Exception as exc:
            yield json.dumps(
                {
                    "type": "error",
                    "code": "generation_failed",
                    "message": str(exc).strip() or "Interview evaluation failed.",
                }
            ) + "\n"
        finally:
            _cancel_task(eval_task)

    return StreamingResponse(
        _event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{session_id}/answer-and-next", response_model=InterviewAnswerAndNextResponse)
async def answer_and_next(
    session_id: str,
    body: SubmitInterviewAnswerRequest,
    user: dict = Depends(require_auth),
):
    """Evaluate an answer, record the turn, then ask the next question.

    Sequenced server-side on purpose: the next question reads post-turn state,
    so a client-side parallel call would generate from the pre-answer session and
    race with the question persistence. A next-question failure degrades to
    ``next_question: null`` with the recorded turn, because the turn is already
    committed at that point.
    """
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None and _generator is not None

    session, question, turn_index = _pending_answer(session_id, user)
    eval_payload = await _generator.evaluate_answer(
        session=_prompt_session(session),
        question=question,
        user_answer=body.user_answer,
        turn_index=turn_index,
        llm_config=body.llm_config,
        user_identity=user,
    )
    eval_payload.pop("follow_up_note_repaired", None)
    degraded = _record_attempt_blocked(eval_payload)

    try:
        turn_dict, session_out = await _finalise_turn(
            user=user,
            session=session,
            session_id=session_id,
            turn_index=turn_index,
            question=question,
            body=body,
            eval_payload=eval_payload,
            degraded=degraded,
        )
    except TurnConflictError as exc:
        raise HTTPException(status_code=409, detail=_turn_conflict_detail()) from exc

    next_question_payload, next_status = await _generate_next_question(
        user=user,
        session_id=session_id,
        llm_config=body.llm_config,
    )
    if next_question_payload is not None:
        # set_current_question rewrote current_question, so the session in the
        # response must be re-read rather than reusing the pre-question snapshot.
        refreshed = _store.get_session(user_id=user["user"], session_id=session_id)
        if refreshed:
            session_out = refreshed

    turn_payload = _turn_response(
        session_id=session_id,
        session_out=session_out,
        turn_dict=turn_dict,
        turn_index=turn_index,
        question=question,
        body=body,
        eval_payload=eval_payload,
        degraded=degraded,
    )
    return InterviewAnswerAndNextResponse(
        session=turn_payload.session,
        turn=turn_payload.turn,
        report_ready=turn_payload.report_ready,
        degraded=turn_payload.degraded,
        next_question=next_question_payload,
        next_question_status=next_status,
    )


@router.post("/{session_id}/answer-and-next/stream")
async def answer_and_next_stream(
    session_id: str,
    body: SubmitInterviewAnswerRequest,
    user: dict = Depends(require_auth),
):
    """NDJSON twin of ``/answer-and-next``: same sequence, same partial success."""
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None and _generator is not None

    session, question, stream_turn_index = _pending_answer(session_id, user)

    async def _event_stream():
        yield json.dumps(
            {
                "type": "start",
                "stage": "evaluating_answer",
                "session_id": session_id,
                "message": "Evaluating interview answer.",
            }
        ) + "\n"
        eval_task: "asyncio.Task[Any] | None" = None
        question_task: "asyncio.Task[Any] | None" = None
        try:
            turn_index = stream_turn_index
            eval_task = asyncio.create_task(
                _generator.evaluate_answer(
                    session=_prompt_session(session),
                    question=question,
                    user_answer=body.user_answer,
                    turn_index=turn_index,
                    llm_config=body.llm_config,
                    user_identity=user,
                )
            )
            async for progress in _progress_events(
                eval_task,
                stage="evaluating_answer",
                message="Evaluating answer and preparing coaching feedback.",
            ):
                yield json.dumps(progress) + "\n"
            eval_payload = await eval_task
            eval_task = None
            eval_payload.pop("follow_up_note_repaired", None)
            degraded = _record_attempt_blocked(eval_payload)

            try:
                turn_dict, session_out = await _finalise_turn(
                    user=user,
                    session=session,
                    session_id=session_id,
                    turn_index=turn_index,
                    question=question,
                    body=body,
                    eval_payload=eval_payload,
                    degraded=degraded,
                )
            except TurnConflictError as exc:
                yield json.dumps(
                    {
                        "type": "error",
                        "code": TurnConflictError.code,
                        "message": str(exc),
                    }
                ) + "\n"
                return

            turn_payload = _turn_response(
                session_id=session_id,
                session_out=session_out,
                turn_dict=turn_dict,
                turn_index=turn_index,
                question=question,
                body=body,
                eval_payload=eval_payload,
                degraded=degraded,
            )

            # The turn is saved at this point, so every failure below is a
            # partial success rather than an error event.
            question_task = asyncio.create_task(
                _generate_next_question(
                    user=user,
                    session_id=session_id,
                    llm_config=body.llm_config,
                )
            )
            async for progress in _progress_events(
                question_task,
                stage="generating_question",
                message="Generating the next interview question.",
            ):
                yield json.dumps(progress) + "\n"
            next_payload, next_status = await question_task
            question_task = None

            response_session = session_out
            if next_payload is not None:
                refreshed = _store.get_session(user_id=user["user"], session_id=session_id)
                if refreshed:
                    response_session = refreshed

            payload = InterviewAnswerAndNextResponse(
                session=response_session,
                turn=turn_payload.turn,
                report_ready=turn_payload.report_ready,
                degraded=turn_payload.degraded,
                next_question=next_payload,
                next_question_status=next_status,
            ).model_dump(mode="json")
            yield json.dumps({"type": "done", **payload}) + "\n"
        except asyncio.CancelledError:
            logger.info("Interview answer-and-next stream cancelled session=%s", session_id)
            raise
        except (
            LLMServiceApprovalRequiredError,
            StudyAppLLMNotAssignedError,
            PersonalCredentialRequiredError,
        ) as exc:
            detail = policy_error_detail(exc)
            yield json.dumps(
                {
                    "type": "error",
                    "code": detail["code"] or "generation_failed",
                    "message": detail["message"] or "Interview evaluation blocked by policy.",
                }
            ) + "\n"
        except Exception as exc:
            yield json.dumps(
                {
                    "type": "error",
                    "code": "generation_failed",
                    "message": str(exc).strip() or "Interview evaluation failed.",
                }
            ) + "\n"
        finally:
            _cancel_task(eval_task)
            _cancel_task(question_task)

    return StreamingResponse(
        _event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{session_id}/next-question", response_model=InterviewQuestionResponse)
async def next_question(
    session_id: str,
    body: NextInterviewQuestionRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None and _generator is not None

    llm_config = body.llm_config
    session = _store.get_session_context(user_id=user["user"], session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Interview session already completed")

    turns = _store.get_turns(user_id=user["user"], session_id=session_id)
    generated = await _generator.generate_question(
        session=_prompt_session(session),
        turns=turns,
        llm_config=llm_config,
        user_identity=user,
    )
    updated = _store.set_current_question(
        user_id=user["user"],
        session_id=session_id,
        question=generated["question"],
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Could not persist next question")

    return InterviewQuestionResponse(
        session_id=session_id,
        turn_index=int(updated["turns_completed"]) + 1,
        question=generated["question"],
        competency_focus=generated.get("competency_focus", ""),
        expected_signals=generated.get("expected_signals", []),
    )


@router.post("/{session_id}/next-question/stream")
async def next_question_stream(
    session_id: str,
    body: NextInterviewQuestionRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None and _generator is not None

    llm_config = body.llm_config
    session = _store.get_session_context(user_id=user["user"], session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Interview session already completed")

    turns = _store.get_turns(user_id=user["user"], session_id=session_id)

    async def _event_stream():
        yield json.dumps(
            {
                "type": "start",
                "stage": "generating_question",
                "session_id": session_id,
                "message": "Generating next interview question.",
            }
        ) + "\n"
        question_task: "asyncio.Task[Any] | None" = None
        try:
            question_task = asyncio.create_task(
                _generator.generate_question(
                    session=_prompt_session(session),
                    turns=turns,
                    llm_config=llm_config,
                    user_identity=user,
                )
            )
            async for progress in _progress_events(
                question_task,
                stage="generating_question",
                message="Generating next interview question.",
            ):
                yield json.dumps(progress) + "\n"
            generated = await question_task
            question_task = None

            updated = _store.set_current_question(
                user_id=user["user"],
                session_id=session_id,
                question=generated["question"],
            )
            if not updated:
                raise RuntimeError("Could not persist next question")

            payload = InterviewQuestionResponse(
                session_id=session_id,
                turn_index=int(updated["turns_completed"]) + 1,
                question=generated["question"],
                competency_focus=generated.get("competency_focus", ""),
                expected_signals=generated.get("expected_signals", []),
            ).model_dump(mode="json")
            yield json.dumps({"type": "done", **payload}) + "\n"
        except asyncio.CancelledError:
            logger.info("Interview next-question stream cancelled session=%s", session_id)
            raise
        except (
            LLMServiceApprovalRequiredError,
            StudyAppLLMNotAssignedError,
            PersonalCredentialRequiredError,
        ) as exc:
            detail = policy_error_detail(exc)
            yield json.dumps(
                {
                    "type": "error",
                    "code": detail["code"] or "generation_failed",
                    "message": detail["message"] or "Interview generation blocked by policy.",
                }
            ) + "\n"
        except Exception as exc:
            yield json.dumps(
                {
                    "type": "error",
                    "code": "generation_failed",
                    "message": str(exc).strip() or "Interview question generation failed.",
                }
            ) + "\n"
        finally:
            _cancel_task(question_task)

    return StreamingResponse(
        _event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{session_id}/report", response_model=InterviewReportResponse)
async def get_report(session_id: str, user: dict = Depends(require_auth)):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None and _generator is not None

    session = _store.get_session(user_id=user["user"], session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")

    report = _store.get_report(user_id=user["user"], session_id=session_id)
    if report is None:
        turns = _store.get_turns(user_id=user["user"], session_id=session_id)
        if not turns:
            raise HTTPException(status_code=400, detail="No completed turns for report")
        report = _generator.build_report(session=session, turns=turns)
        report = _store.save_report(user_id=user["user"], session_id=session_id, report=report)

    return InterviewReportResponse(session=session, report=report)
