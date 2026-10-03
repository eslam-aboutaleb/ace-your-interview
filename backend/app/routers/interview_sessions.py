"""Mock interview router — personalized text-first interview sessions."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    CompanyPackInfo,
    CompanyPacksResponse,
    CreateInterviewSessionRequestExtended,
    InterviewQuestionResponse,
    InterviewReportResponse,
    InterviewSessionResponseExtended,
    InterviewSessionsListResponse,
    InterviewStatsResponse,
    InterviewTrendsResponse,
    InterviewTurnExtended,
    InterviewTurnResponseExtended,
    JDAnalysisRequest,
    JDAnalysisResponse,
    LLMConfigRequest,
    NextInterviewQuestionRequest,
    ResumeProfile,
    ResumeUploadResponse,
    RubricScoreExtended,
    SubmitInterviewAnswerRequestWithHints,
)
from app.services.doc_parser import DocParser
from app.services.company_packs import CompanyPackStore
from app.services.interview_generator import InterviewGenerator
from app.services.interview_store import InterviewStore, TurnConflictError
from app.services.jd_analyzer import JDAnalyzer
from app.services.learning_store import LearningStore
from app.services.llm_client import LLMClient
from app.services.llm_policy import (
    LLMServiceApprovalRequiredError,
    PersonalCredentialRequiredError,
    StudyAppLLMNotAssignedError,
    policy_error_detail,
)
from app.services.mcp_gateway import MCPGateway
from app.services.resume_parser import (
    MAX_RESUME_FILE_SIZE,
    ResumeParseError,
    ResumeParser,
    ResumeProfileStore,
)
from app.services.star_store import StarStore

router = APIRouter(prefix="/api/interview-sessions", tags=["interview-sessions"])

_store: InterviewStore | None = None
_generator: InterviewGenerator | None = None
_parser: DocParser | None = None
_learning_store: LearningStore | None = None
_resume_store: ResumeProfileStore | None = None
_resume_parser: ResumeParser | None = None
_jd_analyzer: JDAnalyzer | None = None
_company_packs: CompanyPackStore | None = None
_star_store: StarStore | None = None


def init(
    llm_client: LLMClient,
    parser: DocParser,
    learning_store: LearningStore,
    db_path: str,
    mcp_gateway: MCPGateway | None = None,
):
    global _store, _generator, _parser, _learning_store
    global _resume_store, _resume_parser, _jd_analyzer, _company_packs, _star_store
    _store = InterviewStore(db_path)
    _resume_store = ResumeProfileStore(db_path)
    _jd_analyzer = JDAnalyzer(llm_client)
    _company_packs = CompanyPackStore(db_path)
    _star_store = StarStore(db_path)
    _generator = InterviewGenerator(
        llm_client,
        parser,
        mcp_gateway=mcp_gateway,
        company_pack_store=_company_packs,
        star_store=_star_store,
    )
    _resume_parser = ResumeParser(llm_client, _resume_store)
    _parser = parser
    _learning_store = learning_store


def get_interview_services() -> tuple[InterviewStore, InterviewGenerator] | None:
    """Share the initialised interview store and generator.

    The voice router uses these same instances so a
    voice-to-voice interview and a REST interview hit the
    same persistence and generation pipeline.
    """
    if _store is None or _generator is None:
        return None
    return _store, _generator


def _ensure_enabled() -> None:
    if not get_settings().enable_mock_interview_v1:
        raise HTTPException(status_code=404, detail="Mock interviews disabled")


def _ensure_ready() -> None:
    if (
        _store is None
        or _generator is None
        or _parser is None
        or _resume_store is None
        or _resume_parser is None
        or _jd_analyzer is None
        or _company_packs is None
        or _star_store is None
    ):
        raise HTTPException(status_code=503, detail="Mock interview services not initialised")


def _profile_to_summary_text(profile: dict[str, Any]) -> str:
    """Render a stored resume profile as resume-summary text.

    The raw resume is never persisted, so the profile is the only
    thing that can be re-attached to a new session.
    """
    skills = ", ".join(profile.get("skills") or [])
    roles = ", ".join(profile.get("roles") or [])
    projects = "; ".join(profile.get("projects") or [])
    strengths = ", ".join(profile.get("strengths") or [])
    years = profile.get("experience_years")
    years_text = f"{years:g} years of experience" if isinstance(years, (int, float)) else ""
    parts = [
        f"Skills: {skills}" if skills else "",
        f"Roles: {roles}" if roles else "",
        f"Experience: {years_text}" if years_text else "",
        f"Projects: {projects}" if projects else "",
        f"Strengths: {strengths}" if strengths else "",
    ]
    return "\n".join(part for part in parts if part)


def _to_session_response(user_id: str, session_id: str) -> InterviewSessionResponseExtended:
    assert _store is not None
    session = _store.get_session(user_id=user_id, session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")

    turns = _store.get_turns(user_id=user_id, session_id=session_id)
    parsed_turns = [InterviewTurnExtended(**t) for t in turns]
    return InterviewSessionResponseExtended(session=session, turns=parsed_turns)


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


async def personalize_session_inputs(
    body: CreateInterviewSessionRequestExtended,
    *,
    user_id: str,
    llm_config: Any = None,
    user_identity: dict | None = None,
) -> tuple[str, list[str]]:
    """Resolve resume auto-attach and JD auto-analysis.

    Returns ``(resume_summary_text, focus_areas)``.
    """
    resume_summary_text = body.resume_summary_text.strip()
    focus_areas = [x.strip() for x in body.focus_areas if x.strip()]

    profile = None
    if _resume_store is not None:
        profile = _resume_store.get(user_id)
    if body.resume_profile and profile is not None:
        profile_text = _profile_to_summary_text(profile)
        if profile_text:
            resume_summary_text = (
                f"{resume_summary_text}\n{profile_text}".strip()
                if resume_summary_text
                else profile_text
            )

    jd_text = body.jd_text.strip()
    if jd_text and _jd_analyzer is not None:
        analysis = await _jd_analyzer.analyze(
            jd_text=jd_text,
            profile=profile,
            llm_config=llm_config,
            user_identity=user_identity,
        )
        for area in analysis.get("recommended_focus_areas") or []:
            area = str(area).strip()
            if area and area not in focus_areas:
                focus_areas.append(area)

    return resume_summary_text, focus_areas


@router.post("", response_model=InterviewSessionResponseExtended)
async def create_session(
    body: CreateInterviewSessionRequestExtended,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None and _generator is not None

    resume_summary_text, focus_areas = await personalize_session_inputs(
        body,
        user_id=user["user"],
        llm_config=body.llm_config,
        user_identity=user,
    )
    job_description_text = body.job_description_text.strip()
    if not job_description_text:
        job_description_text = body.jd_text.strip()

    session = _store.create_session(
        user_id=user["user"],
        track=body.track.value,
        level=body.level.value,
        interview_type=body.interview_type.value,
        turn_count=body.turn_count,
        target_role=body.target_role.strip(),
        interviewer_style=body.interviewer_style.value,
        feedback_mode=body.feedback_mode.value,
        job_description_text=job_description_text,
        resume_summary_text=resume_summary_text,
        focus_areas=focus_areas,
        company=body.company,
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


@router.post("/stream")
async def create_session_stream(
    body: CreateInterviewSessionRequestExtended,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    _ensure_ready()
    assert _store is not None and _generator is not None

    resume_summary_text, focus_areas = await personalize_session_inputs(
        body,
        user_id=user["user"],
        llm_config=body.llm_config,
        user_identity=user,
    )
    job_description_text = body.job_description_text.strip()
    if not job_description_text:
        job_description_text = body.jd_text.strip()

    async def _event_stream():
        yield json.dumps(
            {
                "type": "start",
                "stage": "creating_session",
                "message": "Creating interview session.",
            }
        ) + "\n"
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
                job_description_text=job_description_text,
                resume_summary_text=resume_summary_text,
                focus_areas=focus_areas,
                company=body.company,
            )
            context = _store.get_session_context(
                user_id=user["user"],
                session_id=session["session_id"],
            )
            if not context:
                raise RuntimeError("Failed to create session context")

            first_q_task = asyncio.create_task(
                _generator.generate_question(
                    session=context,
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
            _store.set_current_question(
                user_id=user["user"],
                session_id=session["session_id"],
                question=first_q["question"],
            )

            payload = _to_session_response(user["user"], session["session_id"]).model_dump(mode="json")
            yield json.dumps({"type": "done", **payload}) + "\n"
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


@router.post("/resume", response_model=ResumeUploadResponse, status_code=201)
async def upload_resume(
    file: UploadFile = File(...),
    llm_config_json: str = Form(default=""),
    user: dict = Depends(require_auth),
):
    """Upload a resume (PDF/DOCX/TXT) for structured profile extraction.

    The raw file content is never persisted; only the extracted
    profile (encrypted at rest) and its SHA-256 hash are stored.
    """
    _ensure_enabled()
    _ensure_ready()
    assert _resume_parser is not None and _resume_store is not None

    if not get_settings().enable_interview_plus_v1:
        raise HTTPException(status_code=404, detail="Interview personalization disabled")

    filename = (file.filename or "").strip() or "resume"
    llm_config = None
    if llm_config_json.strip():
        try:
            llm_config = LLMConfigRequest.model_validate_json(llm_config_json)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty resume file")
    if len(content) > MAX_RESUME_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Resume exceeds the {MAX_RESUME_FILE_SIZE // (1024 * 1024)} MB limit",
        )

    try:
        profile = await _resume_parser.parse_and_store(
            user_id=user["user"],
            filename=filename,
            content=content,
            llm_config=llm_config,
            user_identity=user,
        )
    except ResumeParseError as exc:
        detail = {"code": "resume_parse_failed", "message": exc.detail}
        if exc.field:
            detail["field"] = exc.field
        raise HTTPException(status_code=422, detail=detail) from exc

    return ResumeUploadResponse(profile=ResumeProfile(**profile), source="llm")


@router.delete("/resume", status_code=204)
async def delete_resume(user: dict = Depends(require_auth)):
    """GDPR delete: remove the stored resume profile."""
    _ensure_enabled()
    _ensure_ready()
    assert _resume_store is not None

    if not get_settings().enable_interview_plus_v1:
        raise HTTPException(status_code=404, detail="Interview personalization disabled")

    _resume_store.delete(user["user"])
    return None


@router.post("/jd-analysis", response_model=JDAnalysisResponse)
async def analyze_job_description(
    body: JDAnalysisRequest,
    user: dict = Depends(require_auth),
):
    """Compare a job description against the stored resume profile."""
    _ensure_enabled()
    _ensure_ready()
    assert _jd_analyzer is not None

    if not get_settings().enable_interview_plus_v1:
        raise HTTPException(status_code=404, detail="Interview personalization disabled")

    profile = _resume_store.get(user["user"]) if _resume_store is not None else None
    analysis = await _jd_analyzer.analyze(
        jd_text=body.jd_text,
        profile=profile,
    )
    return JDAnalysisResponse(**analysis)


@router.get("/companies", response_model=CompanyPacksResponse)
async def list_company_packs(user: dict = Depends(require_auth)):
    """Available company-specific interview packs."""
    _ensure_enabled()
    _ensure_ready()
    assert _company_packs is not None

    if not get_settings().enable_interview_plus_v1:
        raise HTTPException(status_code=404, detail="Interview personalization disabled")

    packs = _company_packs.list_packs()
    return CompanyPacksResponse(
        packs=[
            CompanyPackInfo(
                company=pack["company"],
                track=pack["track"],
                style_config=pack["style_config"],
                updated_at=pack["updated_at"],
            )
            for pack in packs
        ]
    )


@router.get("/{session_id}", response_model=InterviewSessionResponseExtended)
async def get_session(session_id: str, user: dict = Depends(require_auth)):
    _ensure_enabled()
    _ensure_ready()
    return _to_session_response(user["user"], session_id)


@router.post("/{session_id}/answer", response_model=InterviewTurnResponseExtended)
async def submit_answer(
    session_id: str,
    body: SubmitInterviewAnswerRequestWithHints,
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
    # Cheap pre-check: a turn index that is already taken is a duplicate
    # submission, so reject it before spending an evaluation call.
    if _store.turn_exists(user_id=user["user"], session_id=session_id, turn_index=turn_index):
        raise HTTPException(status_code=409, detail=_turn_conflict_detail())
    eval_payload = await _generator.evaluate_answer(
        session=session,
        question=question,
        user_answer=body.user_answer,
        turn_index=turn_index,
        llm_config=body.llm_config,
        user_identity=user,
        hint_level=body.hint_level,
    )
    # Internal drift signal for the eval harness; never part of an API payload.
    eval_payload.pop("follow_up_note_repaired", None)
    degraded = _record_attempt_blocked(eval_payload)

    try:
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
    except TurnConflictError as exc:
        raise HTTPException(status_code=409, detail=_turn_conflict_detail()) from exc

    prior_memory = str(session.get("memory_summary", "")).strip()
    memory_update = (
        f"{prior_memory}\n"
        f"Turn {turn_index} question: {question[:220]}\n"
        f"Candidate answer summary: {body.user_answer[:320]}\n"
        f"Strengths: {'; '.join(eval_payload['strengths'][:2])}\n"
        f"Improvements: {'; '.join(eval_payload['improvements'][:2])}"
    ).strip()
    _store.set_memory_summary(
        user_id=user["user"],
        session_id=session_id,
        summary=memory_update[:2400],
    )

    if (
        get_settings().enable_adaptive_learning
        and _learning_store is not None
        and not degraded
    ):
        topic_id = _track_topic_id(session["track"])
        if topic_id:
            rubric = eval_payload["rubric"]
            overall = int(rubric.get("overall", 60))
            confidence_signal = max(1, min(5, int(rubric.get("confidence_signal", 3))))
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
        raise HTTPException(status_code=500, detail="Session update failed")

    return InterviewTurnResponseExtended(
        session=session_out,
        turn=InterviewTurnExtended(
            session_id=session_id,
            turn_index=turn_index,
            question=question,
            user_answer=body.user_answer,
            rubric=RubricScoreExtended(**eval_payload["rubric"]),
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


@router.post("/{session_id}/answer/stream")
async def submit_answer_stream(
    session_id: str,
    body: SubmitInterviewAnswerRequestWithHints,
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

    # A repeat submission of an already recorded turn can be rejected before the
    # response body starts, so it gets a real HTTP status. The race that slips
    # past this check is caught by record_turn and reported as a `turn_conflict`
    # error event, because a streaming response can no longer change its status.
    stream_turn_index = int(session["turns_completed"]) + 1
    if _store.turn_exists(
        user_id=user["user"],
        session_id=session_id,
        turn_index=stream_turn_index,
    ):
        raise HTTPException(status_code=409, detail=_turn_conflict_detail())

    async def _event_stream():
        yield json.dumps(
            {
                "type": "start",
                "stage": "evaluating_answer",
                "session_id": session_id,
                "message": "Evaluating interview answer.",
            }
        ) + "\n"
        try:
            turn_index = stream_turn_index
            eval_task = asyncio.create_task(
                _generator.evaluate_answer(
                    session=session,
                    question=question,
                    user_answer=body.user_answer,
                    turn_index=turn_index,
                    llm_config=body.llm_config,
                    user_identity=user,
                    hint_level=body.hint_level,
                )
            )
            async for progress in _progress_events(
                eval_task,
                stage="evaluating_answer",
                message="Evaluating answer and preparing coaching feedback.",
            ):
                yield json.dumps(progress) + "\n"
            eval_payload = await eval_task
            # Internal drift signal for the eval harness; never part of the wire payload.
            eval_payload.pop("follow_up_note_repaired", None)
            degraded = _record_attempt_blocked(eval_payload)

            try:
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
            except TurnConflictError as exc:
                yield json.dumps(
                    {
                        "type": "error",
                        "code": TurnConflictError.code,
                        "message": str(exc),
                    }
                ) + "\n"
                return

            prior_memory = str(session.get("memory_summary", "")).strip()
            memory_update = (
                f"{prior_memory}\n"
                f"Turn {turn_index} question: {question[:220]}\n"
                f"Candidate answer summary: {body.user_answer[:320]}\n"
                f"Strengths: {'; '.join(eval_payload['strengths'][:2])}\n"
                f"Improvements: {'; '.join(eval_payload['improvements'][:2])}"
            ).strip()
            _store.set_memory_summary(
                user_id=user["user"],
                session_id=session_id,
                summary=memory_update[:2400],
            )

            if (
                get_settings().enable_adaptive_learning
                and _learning_store is not None
                and not degraded
            ):
                topic_id = _track_topic_id(session["track"])
                if topic_id:
                    rubric = eval_payload["rubric"]
                    overall = int(rubric.get("overall", 60))
                    confidence_signal = max(1, min(5, int(rubric.get("confidence_signal", 3))))
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

            payload = InterviewTurnResponseExtended(
                session=session_out,
                turn=InterviewTurnExtended(
                    session_id=session_id,
                    turn_index=turn_index,
                    question=question,
                    user_answer=body.user_answer,
                    rubric=RubricScoreExtended(**eval_payload["rubric"]),
                    strengths=eval_payload["strengths"],
                    improvements=eval_payload["improvements"],
                    follow_up_note=eval_payload["follow_up_note"],
                    response_time_ms=max(0, body.response_time_ms),
                    created_at=turn_dict["created_at"],
                    degraded=degraded,
                ),
                report_ready=bool(session_out.get("report_ready") or session_out["status"] == "completed"),
                degraded=degraded,
            ).model_dump(mode="json")
            yield json.dumps({"type": "done", **payload}) + "\n"
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
        try:
            question_task = asyncio.create_task(
                _generator.generate_question(
                    session=session,
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
