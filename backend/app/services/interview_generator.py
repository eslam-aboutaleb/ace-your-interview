"""LLM-powered question generation and rubric scoring for mock interviews."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from app.schemas.models import LLMConfigRequest
from app.services.doc_parser import DocParser
from app.services.llm_client import LLMClient
from app.services.llm_policy import raise_if_policy_blocked_result

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 4


def _extract_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\\s*\\n?(.*?)\\n?\\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

    return {}


def _normalise_question(text: str) -> str:
    return re.sub(r"\\s+", " ", (text or "").strip().lower())


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        n = int(value)
    except (ValueError, TypeError):
        return default
    return max(lo, min(hi, n))


class InterviewGenerator:
    """Handles interview question creation, evaluation, and report synthesis."""

    def __init__(self, llm: LLMClient, parser: DocParser):
        self.llm = llm
        self.parser = parser

    @staticmethod
    def _format_recent_turns(turns: list[dict[str, Any]]) -> str:
        if not turns:
            return "(none yet)"
        last = turns[-3:]
        lines: list[str] = []
        for t in last:
            lines.append(
                f"Turn {t.get('turn_index')}: Q={t.get('question')} | "
                f"A={str(t.get('user_answer', ''))[:500]}"
            )
        return "\n".join(lines)

    @staticmethod
    def _question_prompt(
        session: dict[str, Any],
        turns: list[dict[str, Any]],
    ) -> str:
        asked = session.get("asked_questions") or []
        return f"""You are a senior interviewer running a personalized mock interview.

Generate the NEXT interview question as strict JSON.

Session configuration:
- track: {session.get('track')}
- level: {session.get('level')}
- interview_type: {session.get('interview_type')}
- target_role: {session.get('target_role') or 'not specified'}
- focus_areas: {', '.join(session.get('focus_areas') or []) or 'none'}
- turn_count: {session.get('turn_count')}
- turns_completed: {session.get('turns_completed')}
- job_description_text: {(session.get('job_description_text') or '')[:2500]}
- resume_summary_text: {(session.get('resume_summary_text') or '')[:2500]}

Recent turns:
{InterviewGenerator._format_recent_turns(turns)}

Already asked questions (do not repeat semantically):
{json.dumps(asked[:30], ensure_ascii=True)}

Return ONLY a JSON object:
{{
  "question": "string",
  "competency_focus": "string",
  "expected_signals": ["signal1", "signal2", "signal3"]
}}

Rules:
- Question must be realistic for interviews.
- Match complexity to level: junior (fundamentals), mid (implementation tradeoffs), senior (architecture/risk).
- Keep question <= 45 words.
- expected_signals must contain 2-5 concise bullets.
- No markdown or extra text.
"""

    @staticmethod
    def _evaluate_prompt(
        session: dict[str, Any],
        question: str,
        answer: str,
        turn_index: int,
    ) -> str:
        return f"""You are an interview evaluator.

Evaluate the candidate answer and return strict JSON only.

Session:
- track: {session.get('track')}
- level: {session.get('level')}
- interview_type: {session.get('interview_type')}
- target_role: {session.get('target_role') or 'not specified'}
- turn_index: {turn_index}

Question:
{question}

Candidate answer:
{answer[:12000]}

Return ONLY this JSON object:
{{
  "rubric": {{
    "technical_accuracy": 0-5,
    "reasoning_depth": 0-5,
    "communication_clarity": 0-5,
    "completeness": 0-5,
    "confidence_signal": 0-5,
    "overall": 0-100
  }},
  "strengths": ["..."],
  "improvements": ["..."],
  "follow_up_note": "markdown coaching note"
}}

Rules:
- Keep scoring strict and evidence-based.
- strengths/improvements must each have 1-4 concise bullets.
- If answer is weak or vague, score low rather than guessing intent.
- strengths/improvements should remain plain short strings.
- follow_up_note should use adaptive markdown structure:
  - default to concise coaching prose,
  - use bullets only when giving multi-step action plans,
  - use headings only when sections improve clarity,
  - use tables only for direct option/tradeoff comparisons.
- If you use a table, output valid GFM table syntax:
  - one row per line,
  - include a separator row (e.g. `| --- | --- |`).
- You may include fenced code blocks when code clarifies a concrete fix.
- You may include fenced Mermaid diagrams when architecture/flow coaching is clearer visually.
- If you include fences, always use explicit language tags (for example: ```python, ```mermaid).
- Keep valid JSON string escaping for newlines, quotes, and backslashes.
- No extra keys.
"""

    @staticmethod
    def _validate_question_payload(
        payload: dict[str, Any],
        asked_questions: list[str],
    ) -> tuple[bool, str]:
        q = str(payload.get("question", "")).strip()
        if len(q) < 10:
            return False, "question_too_short"
        if len(q.split()) > 45:
            return False, "question_too_long"

        focus = str(payload.get("competency_focus", "")).strip()
        if len(focus) < 3:
            return False, "missing_competency_focus"

        signals = payload.get("expected_signals")
        if not isinstance(signals, list) or not (2 <= len(signals) <= 5):
            return False, "invalid_expected_signals"

        q_norm = _normalise_question(q)
        asked_norm = {_normalise_question(x) for x in asked_questions}
        if q_norm in asked_norm:
            return False, "duplicate_question"

        return True, ""

    @staticmethod
    def _validate_eval_payload(payload: dict[str, Any]) -> tuple[bool, str]:
        rubric = payload.get("rubric")
        if not isinstance(rubric, dict):
            return False, "missing_rubric"

        required = [
            "technical_accuracy",
            "reasoning_depth",
            "communication_clarity",
            "completeness",
            "confidence_signal",
            "overall",
        ]
        for key in required:
            if key not in rubric:
                return False, f"missing_{key}"

        strengths = payload.get("strengths")
        improvements = payload.get("improvements")
        note = str(payload.get("follow_up_note", "")).strip()

        if not isinstance(strengths, list) or not strengths:
            return False, "missing_strengths"
        if not isinstance(improvements, list) or not improvements:
            return False, "missing_improvements"
        if len(note) < 8:
            return False, "missing_follow_up_note"

        return True, ""

    def _normalise_eval_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        rubric = payload.get("rubric") if isinstance(payload.get("rubric"), dict) else {}
        norm = {
            "technical_accuracy": _clamp_int(rubric.get("technical_accuracy"), 0, 5, 3),
            "reasoning_depth": _clamp_int(rubric.get("reasoning_depth"), 0, 5, 3),
            "communication_clarity": _clamp_int(rubric.get("communication_clarity"), 0, 5, 3),
            "completeness": _clamp_int(rubric.get("completeness"), 0, 5, 3),
            "confidence_signal": _clamp_int(rubric.get("confidence_signal"), 0, 5, 3),
            "overall": _clamp_int(rubric.get("overall"), 0, 100, 60),
        }

        strengths_raw = payload.get("strengths") if isinstance(payload.get("strengths"), list) else []
        improvements_raw = (
            payload.get("improvements") if isinstance(payload.get("improvements"), list) else []
        )

        strengths = [str(x).strip() for x in strengths_raw if str(x).strip()][:4] or [
            "You provided an answer aligned to the prompt.",
        ]
        improvements = [str(x).strip() for x in improvements_raw if str(x).strip()][:4] or [
            "Use a clearer structure and include a concrete tradeoff.",
        ]

        note = str(payload.get("follow_up_note", "")).strip() or "Practice a concise STAR-style structure for stronger clarity."

        return {
            "rubric": norm,
            "strengths": strengths,
            "improvements": improvements,
            "follow_up_note": note,
        }

    async def generate_question(
        self,
        *,
        session: dict[str, Any],
        turns: list[dict[str, Any]],
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> dict[str, Any]:
        base_prompt = self._question_prompt(session, turns)
        asked = list(session.get("asked_questions") or [])
        prompt = base_prompt

        for _ in range(_MAX_ATTEMPTS):
            result = await self.llm.completion(
                prompt,
                llm_config,
                user_identity=user_identity,
            )
            raise_if_policy_blocked_result(result)
            payload = _extract_json(result.get("analysis", "")) if result.get("success") else {}
            ok, issue = self._validate_question_payload(payload, asked)
            if ok:
                signals = payload.get("expected_signals") if isinstance(payload.get("expected_signals"), list) else []
                return {
                    "question": str(payload.get("question", "")).strip(),
                    "competency_focus": str(payload.get("competency_focus", "")).strip(),
                    "expected_signals": [str(x).strip() for x in signals if str(x).strip()][:5],
                }
            prompt = (
                f"Your previous JSON was invalid ({issue}). Return only a corrected JSON object.\n\n"
                f"{base_prompt}"
            )

        logger.warning("Falling back to deterministic question for session=%s", session.get("session_id"))
        track = session.get("track", "technical")
        level = session.get("level", "mid")
        return {
            "question": f"For a {level} {track} interview, explain a recent design decision you would make and one tradeoff you would accept.",
            "competency_focus": "structured reasoning and tradeoff analysis",
            "expected_signals": [
                "Clear problem framing",
                "Explicit constraints",
                "Concrete tradeoff justification",
            ],
        }

    async def evaluate_answer(
        self,
        *,
        session: dict[str, Any],
        question: str,
        user_answer: str,
        turn_index: int,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> dict[str, Any]:
        base_prompt = self._evaluate_prompt(session, question, user_answer, turn_index)
        prompt = base_prompt

        for _ in range(_MAX_ATTEMPTS):
            result = await self.llm.completion(
                prompt,
                llm_config,
                user_identity=user_identity,
            )
            raise_if_policy_blocked_result(result)
            payload = _extract_json(result.get("analysis", "")) if result.get("success") else {}
            ok, issue = self._validate_eval_payload(payload)
            if ok:
                return self._normalise_eval_payload(payload)
            prompt = (
                f"Your previous JSON was invalid ({issue}). Return only corrected JSON.\n\n"
                f"{base_prompt}"
            )

        logger.warning(
            "Falling back to deterministic rubric evaluation for session=%s turn=%s",
            session.get("session_id"),
            turn_index,
        )
        return {
            "rubric": {
                "technical_accuracy": 3,
                "reasoning_depth": 3,
                "communication_clarity": 3,
                "completeness": 3,
                "confidence_signal": 3,
                "overall": 60,
            },
            "strengths": ["You attempted the question and provided a directionally relevant response."],
            "improvements": ["Add clearer structure and include one concrete example or metric."],
            "follow_up_note": (
                "### Coaching Focus\n"
                "Use a tighter answer structure so your reasoning is easier to evaluate.\n\n"
                "- State the problem in one sentence.\n"
                "- Describe your approach in two to three steps.\n"
                "- Name one tradeoff explicitly.\n"
                "- End with a measurable expected outcome."
            ),
        }

    def build_report(
        self,
        *,
        session: dict[str, Any],
        turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not turns:
            return {
                "overall_score": 0.0,
                "readiness_label": "Not Started",
                "completed_turns": 0,
                "rubric_averages": {
                    "technical_accuracy": 0.0,
                    "reasoning_depth": 0.0,
                    "communication_clarity": 0.0,
                    "completeness": 0.0,
                    "confidence_signal": 0.0,
                    "overall": 0.0,
                },
                "weak_competencies": ["No answers submitted yet"],
                "strengths": [],
                "recommended_topic_ids": [],
                "next_steps": ["Complete at least one interview turn to unlock a personalized report."],
                "summary": (
                    "### Interview Summary\n"
                    "No interview data is available yet.\n\n"
                    "- Complete at least one turn to generate a personalized readiness summary."
                ),
                "created_at": "",
            }

        dims = [
            "technical_accuracy",
            "reasoning_depth",
            "communication_clarity",
            "completeness",
            "confidence_signal",
            "overall",
        ]
        sums = {k: 0.0 for k in dims}
        strengths: list[str] = []
        improvements: list[str] = []

        for turn in turns:
            rubric = turn.get("rubric", {}) or {}
            for key in dims:
                value = rubric.get(key)
                if isinstance(value, (int, float)):
                    sums[key] += float(value)
            strengths.extend([str(s).strip() for s in turn.get("strengths", []) if str(s).strip()])
            improvements.extend([str(i).strip() for i in turn.get("improvements", []) if str(i).strip()])

        count = max(1, len(turns))
        avgs = {k: round(sums[k] / count, 2) for k in dims}
        overall = avgs["overall"]

        weak_dims = [
            key for key in [
                "technical_accuracy",
                "reasoning_depth",
                "communication_clarity",
                "completeness",
                "confidence_signal",
            ]
            if avgs[key] < 3.0
        ]

        if overall >= 85:
            readiness = "Strong"
        elif overall >= 70:
            readiness = "On Track"
        elif overall >= 50:
            readiness = "Developing"
        else:
            readiness = "Needs Work"

        topics = self.parser.list_topics(track=session.get("track"))
        recommended_topic_ids = [t.id for t in topics[:4]]

        next_steps = []
        if weak_dims:
            next_steps.append(
                "Prioritize weak competencies in your next mock interview round."
            )
        next_steps.append("Do one targeted quiz run on the recommended topics.")
        next_steps.append("Repeat a mock interview and aim for +10 overall score improvement.")

        top_strengths = []
        for item in strengths:
            if item not in top_strengths:
                top_strengths.append(item)
            if len(top_strengths) >= 4:
                break

        weak_labels = [w.replace("_", " ") for w in weak_dims] or ["No major weak competency detected"]
        summary_lines = [
            "### Interview Summary",
            (
                f"You completed **{len(turns)}** turns with an overall readiness score of "
                f"**{overall}** ({readiness})."
            ),
            "",
            "#### Focus Areas",
        ]
        for label in weak_labels[:3]:
            summary_lines.append(f"- {label}")
        summary_lines.extend(
            [
                "",
                "#### Score Snapshot",
                "| Dimension | Average |",
                "| --- | ---: |",
                f"| Technical accuracy | {avgs['technical_accuracy']} |",
                f"| Reasoning depth | {avgs['reasoning_depth']} |",
                f"| Communication clarity | {avgs['communication_clarity']} |",
                f"| Completeness | {avgs['completeness']} |",
                f"| Confidence signal | {avgs['confidence_signal']} |",
                f"| Overall | {avgs['overall']} |",
            ]
        )
        summary = "\n".join(summary_lines)

        return {
            "overall_score": overall,
            "readiness_label": readiness,
            "completed_turns": len(turns),
            "rubric_averages": avgs,
            "weak_competencies": weak_labels,
            "strengths": top_strengths,
            "recommended_topic_ids": recommended_topic_ids,
            "next_steps": next_steps,
            "summary": summary,
            "created_at": turns[-1].get("created_at", ""),
        }
