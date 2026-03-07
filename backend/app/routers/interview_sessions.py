"""Mock interview router — personalized text-first interview sessions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    CreateInterviewSessionRequest,
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
from app.services.interview_store import InterviewStore
from app.services.learning_store import LearningStore
from app.services.llm_client import LLMClient
from app.services.mcp_gateway import MCPGateway

router = APIRouter(prefix="/api/interview-sessions", tags=["interview-sessions"])

_store: InterviewStore | None = None
_generator: InterviewGenerator | None = None
_parser: DocParser | None = None
_learning_store: LearningStore | None = None


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
        job_description_text=body.job_description_text.strip(),
        resume_summary_text=body.resume_summary_text.strip(),
        focus_areas=[x.strip() for x in body.focus_areas if x.strip()],
    )

    context = _store.get_session_context(user_id=user["user"], session_id=session["session_id"])
    if not context:
        raise HTTPException(status_code=500, detail="Failed to create session context")

    first_q = await _generator.generate_question(
        session=context,
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

    session = _store.get_session_context(user_id=user["user"], session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Interview session already completed")

    question = (session.get("current_question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="No active question. Generate next question first.")

    turn_index = int(session["turns_completed"]) + 1
    eval_payload = await _generator.evaluate_answer(
        session=session,
        question=question,
        user_answer=body.user_answer,
        turn_index=turn_index,
        llm_config=body.llm_config,
        user_identity=user,
    )

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
    )

    if get_settings().enable_adaptive_learning and _learning_store is not None:
        topic_id = _track_topic_id(session["track"])
        if topic_id:
            rubric = eval_payload["rubric"]
            overall = int(rubric.get("overall", 60))
            confidence_signal = max(1, min(5, int(rubric.get("confidence_signal", 3))))
            _learning_store.record_attempt(
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
        raise HTTPException(status_code=500, detail="Session update failed")

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
        ),
        report_ready=bool(session_out.get("report_ready") or session_out["status"] == "completed"),
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
        session=session,
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
