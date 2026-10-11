"""Tests for the real MCP stdio server (``app/mcp_server``).

Covers the JSON-RPC 2.0 round trip over the newline-delimited stdio wire
format (initialize → initialized → ping → tools/list → tools/call), the
single-user safety gate (refusals must be explicit), the shape of every
tool's MCP content payload, graceful empty/missing data, and the
``python -m app.mcp_server`` entry point acceptance path.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from app.config import Settings
from app.mcp_server import build_server
from app.mcp_server.protocol import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
)
from app.mcp_server.server import (
    MCPRefusalError,
    MCPServer,
    REFUSAL_EXIT_CODE,
    main,
)
from app.mcp_server.tools import InvalidToolParams, MCPToolkit
from app.services.interview_store import InterviewStore
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore

BACKEND_ROOT = Path(__file__).resolve().parents[1]
USER_ID = "mcp-learner"

TOPIC_HTTP = "01-backend-fundamentals-and-http"
TOPIC_API = "02-backend-api-design-and-contracts"

ALL_TOOL_NAMES = {
    "get_review_queue",
    "get_weak_areas",
    "get_next_objectives",
    "get_mastery",
    "get_study_plan",
    "get_interview_stats",
}

_PROGRESS_INSERT = """
INSERT INTO question_progress(
    user_id, question_id, topic_id, mastery_score, due_at, review_bucket,
    attempts, correct_attempts, avg_confidence, last_confidence, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_CARD_INSERT = """
INSERT INTO fsrs_cards(
    user_id, card_id, topic_id, source_type, state, stability, difficulty,
    due_at, last_review_at, reps, lapses, scheduled_days, elapsed_days,
    suspended, created_at, updated_at
) VALUES (?, ?, ?, 'question', 'review', ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
"""


def _settings(db_path: str, *, enabled: bool = True, user_id: str = USER_ID) -> Settings:
    return Settings(
        enable_mcp_server=enabled,
        mcp_server_user_id=user_id,
        learning_db_path=db_path,
    )


def _days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=abs(days))).isoformat()


def _days_ahead(days: int) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).isoformat()


def _insert_progress_rows(store: LearningStore, rows: list[tuple]) -> None:
    """Write ``question_progress`` rows the way the tools aggregate them.

    Rows go straight into the table (not through ``record_attempt``) so the
    fixture is deterministic and independent of which writer each backend
    revision routes attempts through: some revisions write
    ``question_progress``, others write FSRS cards, and the tools must be
    seeded for both readers.
    """
    payload = [
        (
            USER_ID,
            question_id,
            topic_id,
            mastery,
            _days_ago(2) if due_days < 0 else _days_ahead(due_days),
            0 if due_days < 0 else 4,
            attempts,
            correct,
            3.0,
            3,
            _days_ago(2),
        )
        for question_id, topic_id, mastery, attempts, correct, due_days in rows
    ]
    with store._lock, store._conn:
        store._conn.executemany(_PROGRESS_INSERT, payload)


def _insert_due_cards(store: LearningStore, rows: list[tuple]) -> None:
    """Write ``fsrs_cards`` rows, some due now and some scheduled ahead."""
    payload = [
        (
            USER_ID,
            card_id,
            topic_id,
            1.2 if lapses else 8.0,
            7.0 if lapses else 3.0,
            _days_ago(2) if due_days < 0 else _days_ahead(due_days),
            _days_ago(3),
            reps,
            lapses,
            1,
            1,
            _days_ago(3),
            _days_ago(3),
        )
        for card_id, topic_id, due_days, reps, lapses in rows
    ]
    with store._lock, store._conn:
        store._conn.executemany(_CARD_INSERT, payload)


def _seed_learner(db_path: str) -> None:
    """Seed one learner: topic evidence, a profile, and a scored interview."""
    store = LearningStore(db_path)
    _insert_progress_rows(
        store,
        [
            ("q-http-1", TOPIC_HTTP, 0.2, 2, 1, -1),
            ("q-http-2", TOPIC_HTTP, 0.25, 1, 0, -1),
            ("q-api-1", TOPIC_API, 0.8, 2, 2, 30),
        ],
    )
    _insert_due_cards(
        store,
        [
            ("q-http-1", TOPIC_HTTP, -1, 2, 1),
            ("q-http-2", TOPIC_HTTP, -1, 1, 1),
            ("q-api-1", TOPIC_API, 30, 2, 0),
        ],
    )

    planner = LearningPlannerStore(db_path)
    planner.upsert_profile(
        user_id=USER_ID,
        payload={
            "target_role": "Backend Engineer",
            "target_date": (datetime.now(UTC) + timedelta(days=45)).date().isoformat(),
            "weekly_minutes": 300,
            "preferred_session_minutes": 30,
            "current_level": "mid",
            "target_level": "senior",
            "primary_track": "backend",
            "focus_topic_ids": [TOPIC_HTTP],
            "preferred_modalities": ["study", "quiz", "interview"],
        },
    )

    interviews = InterviewStore(db_path)
    session = interviews.create_session(
        user_id=USER_ID,
        track="backend",
        level="mid",
        interview_type="mixed",
        turn_count=5,
        target_role="Backend Engineer",
        job_description_text="",
        resume_summary_text="",
        focus_areas=[],
    )
    interviews.save_report(
        user_id=USER_ID,
        session_id=session["session_id"],
        report={"overall_score": 71.5, "readiness_label": "developing"},
    )


class _ServerHarness:
    """Temporary SQLite db + a built MCPServer, one per test."""

    def __init__(self, *, seed: bool = True, enabled: bool = True, user_id: str = USER_ID):
        self._seed = seed
        self._enabled = enabled
        self._user_id = user_id

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._td.name, "learning.db")
        if self._seed:
            _seed_learner(self.db_path)
        self.settings = _settings(
            self.db_path, enabled=self._enabled, user_id=self._user_id
        )
        self.server = build_server(self.settings)
        return self

    def __exit__(self, *exc_info):
        self._td.cleanup()
        return False


def _run_session(server: MCPServer, messages: list) -> list[dict]:
    """Feed messages through the server's stdio loop, return parsed replies."""
    stdin = io.StringIO(
        "".join(json.dumps(message) + "\n" for message in messages)
    )
    stdout = io.StringIO()
    server.serve(stdin=stdin, stdout=stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def _request(request_id, method, params=None) -> dict:
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def _tool_call(request_id, name, arguments=None) -> dict:
    params = {"name": name}
    if arguments is not None:
        params["arguments"] = arguments
    return _request(request_id, "tools/call", params)


def _initialize(request_id=1, protocol_version="2025-06-18") -> dict:
    return _request(
        request_id,
        "initialize",
        {
            "protocolVersion": protocol_version,
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    )


_INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


class HandshakeTests(unittest.TestCase):
    def test_initialize_returns_valid_handshake(self):
        with _ServerHarness() as harness:
            [response] = _run_session(harness.server, [_initialize()])
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(response["id"], 1)
        result = response["result"]
        self.assertEqual(result["protocolVersion"], "2025-06-18")
        self.assertEqual(result["capabilities"], {"tools": {"listChanged": False}})
        self.assertEqual(result["serverInfo"]["name"], "ace-your-interview-mcp")
        self.assertIn("version", result["serverInfo"])
        self.assertIn("instructions", result)

    def test_initialize_falls_back_to_supported_version_when_not_requested(self):
        with _ServerHarness() as harness:
            [bare] = _run_session(
                harness.server, [_request(9, "initialize", {"capabilities": {}})]
            )
            [absent] = _run_session(harness.server, [_request(9, "initialize")])
            [blank] = _run_session(
                harness.server, [_initialize(9, protocol_version="  ")]
            )
        self.assertEqual(bare["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(absent["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(blank["result"]["protocolVersion"], "2025-03-26")

    def test_initialized_notification_is_silent_but_session_continues(self):
        with _ServerHarness() as harness:
            responses = _run_session(
                harness.server,
                [_initialize(), _INITIALIZED, _request(2, "ping"), _INITIALIZED],
            )
        self.assertEqual([response["id"] for response in responses], [1, 2])
        self.assertEqual(responses[1]["result"], {})

    def test_null_id_request_gets_a_null_id_response(self):
        with _ServerHarness() as harness:
            [response] = _run_session(
                harness.server,
                [{"jsonrpc": "2.0", "id": None, "method": "ping"}],
            )
        self.assertIsNone(response["id"])
        self.assertEqual(response["result"], {})


class ToolsListTests(unittest.TestCase):
    def test_tools_list_exposes_the_read_only_tool_catalog(self):
        with _ServerHarness() as harness:
            _, listed = _run_session(
                harness.server,
                [_initialize(), _request(2, "tools/list")],
            )
        tools = listed["result"]["tools"]
        self.assertEqual({tool["name"] for tool in tools}, ALL_TOOL_NAMES)
        for tool in tools:
            self.assertTrue(tool["description"])
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertFalse(tool["inputSchema"]["additionalProperties"])

    def test_parametric_tools_declare_their_bounds(self):
        with _ServerHarness() as harness:
            [response] = _run_session(harness.server, [_request(2, "tools/list")])
        by_name = {tool["name"]: tool for tool in response["result"]["tools"]}
        limit_schema = by_name["get_review_queue"]["inputSchema"]["properties"]["limit"]
        self.assertEqual(limit_schema["minimum"], 1)
        self.assertEqual(limit_schema["maximum"], 200)
        self.assertEqual(limit_schema["default"], 50)
        days_schema = by_name["get_study_plan"]["inputSchema"]["properties"]["days"]
        self.assertEqual(days_schema["minimum"], 1)
        self.assertEqual(days_schema["maximum"], 31)
        self.assertEqual(days_schema["default"], 7)


class ToolCallShapeTests(unittest.TestCase):
    """Every tool returns a valid MCP content array with a JSON text payload."""

    def _call_all_tools(self, server: MCPServer) -> dict:
        messages = [_initialize()]
        calls = [
            _tool_call(10, "get_review_queue", {"limit": 5}),
            _tool_call(11, "get_weak_areas"),
            _tool_call(12, "get_next_objectives"),
            _tool_call(13, "get_mastery"),
            _tool_call(14, "get_study_plan", {"days": 3}),
            _tool_call(15, "get_interview_stats"),
        ]
        responses = _run_session(server, messages + calls)
        payloads = {}
        for response in responses[1:]:
            self.assertIn("result", response, response)
            result = response["result"]
            self.assertIs(result["isError"], False)
            self.assertIsInstance(result["content"], list)
            self.assertTrue(result["content"])
            for block in result["content"]:
                self.assertIsInstance(block, dict)
                self.assertEqual(block["type"], "text")
                self.assertIsInstance(block["text"], str)
                payloads[response["id"]] = json.loads(block["text"])
        return payloads

    def test_every_tool_returns_valid_content_shape_with_seeded_data(self):
        with _ServerHarness() as harness:
            payloads = self._call_all_tools(harness.server)

        review = payloads[10]
        # Two of the three seeded cards are due; the third is scheduled ahead.
        self.assertEqual(review["total_due"], 2)
        self.assertTrue(review["items"])
        for item in review["items"]:
            self.assertIn("topic_id", item)
            self.assertIn("question_id", item)

        weak = payloads[11]
        self.assertTrue(weak["weak_areas"])

        objectives = payloads[12]
        self.assertTrue(objectives["items"])
        self.assertIn("profile", objectives)
        self.assertIn("days_until_target", objectives)

        mastery = payloads[13]
        self.assertEqual(
            {topic["topic_id"] for topic in mastery["topics"]},
            {TOPIC_HTTP, TOPIC_API},
        )

        plan = payloads[14]
        self.assertEqual(plan["days"], 3)
        self.assertEqual(len(plan["days_plan"]), 3)
        self.assertGreater(plan["total_tasks"], 0)

        stats = payloads[15]
        self.assertEqual(stats["total_sessions"], 1)
        self.assertEqual(stats["completed_sessions"], 0)
        self.assertEqual(stats["interview_readiness_score"], 71.5)

    def test_every_tool_handles_a_completely_empty_database(self):
        with _ServerHarness(seed=False) as harness:
            payloads = self._call_all_tools(harness.server)

        self.assertEqual(payloads[10], {"items": [], "total_due": 0})
        self.assertEqual(payloads[11], {"weak_areas": []})
        self.assertEqual(payloads[13], {"topics": []})
        self.assertEqual(payloads[15]["total_sessions"], 0)
        self.assertEqual(payloads[15]["interview_readiness_score"], 0.0)
        # Cold start still produces actionable objectives and a day skeleton.
        self.assertTrue(payloads[12]["items"])
        plan = payloads[14]
        self.assertEqual(plan["days"], 3)
        self.assertEqual(len(plan["days_plan"]), 3)
        self.assertEqual(
            plan["total_tasks"],
            sum(len(day["tasks"]) for day in plan["days_plan"]),
        )

    def test_review_queue_limit_is_clamped_to_the_declared_bounds(self):
        with _ServerHarness(seed=False) as harness:
            store = LearningStore(harness.db_path)
            _insert_due_cards(
                store,
                [(f"due-q-{idx:04d}", TOPIC_API, -1, 3, 1) for idx in range(210)],
            )
            responses = _run_session(
                harness.server,
                [
                    _tool_call(1, "get_review_queue", {"limit": 5000}),
                    _tool_call(2, "get_review_queue", {"limit": 0}),
                    _tool_call(3, "get_review_queue"),
                ],
            )
        default_payload = json.loads(responses[0]["result"]["content"][0]["text"])
        minimum_payload = json.loads(responses[1]["result"]["content"][0]["text"])
        omitted_payload = json.loads(responses[2]["result"]["content"][0]["text"])
        self.assertEqual(default_payload["total_due"], 200)
        self.assertEqual(minimum_payload["total_due"], 1)
        self.assertEqual(omitted_payload["total_due"], 50)

    def test_study_plan_days_is_clamped_to_the_declared_bounds(self):
        with _ServerHarness(seed=False) as harness:
            responses = _run_session(
                harness.server,
                [
                    _tool_call(1, "get_study_plan", {"days": 999}),
                    _tool_call(2, "get_study_plan", {"days": -5}),
                    _tool_call(3, "get_study_plan"),
                ],
            )
        maximum = json.loads(responses[0]["result"]["content"][0]["text"])
        minimum = json.loads(responses[1]["result"]["content"][0]["text"])
        default = json.loads(responses[2]["result"]["content"][0]["text"])
        self.assertEqual(maximum["days"], 31)
        self.assertEqual(minimum["days"], 1)
        self.assertEqual(default["days"], 7)


class ToolCallErrorTests(unittest.TestCase):
    def test_unknown_tool_is_invalid_params(self):
        with _ServerHarness() as harness:
            [response] = _run_session(
                harness.server, [_tool_call(1, "drop_everything")]
            )
        self.assertEqual(response["error"]["code"], INVALID_PARAMS)
        self.assertIn("drop_everything", response["error"]["message"])

    def test_missing_or_malformed_tool_params_are_invalid_params(self):
        with _ServerHarness() as harness:
            responses = _run_session(
                harness.server,
                [
                    _request(1, "tools/call", {"arguments": {}}),
                    _request(2, "tools/call", {"name": ""}),
                    _request(3, "tools/call", "not-an-object"),
                    _tool_call(4, "get_review_queue", "not-an-object"),
                    _tool_call(5, "get_review_queue", {"limit": "ten"}),
                    _tool_call(6, "get_review_queue", {"limit": True}),
                    _tool_call(7, "get_study_plan", {"days": 2.5}),
                ],
            )
        for response in responses:
            self.assertEqual(response["error"]["code"], INVALID_PARAMS, response)

    def test_tool_execution_failure_returns_an_is_error_result(self):
        with _ServerHarness() as harness:
            with mock.patch.object(
                harness.server._toolkit,
                "execute",
                side_effect=RuntimeError("db exploded"),
            ):
                [response] = _run_session(
                    harness.server, [_tool_call(1, "get_mastery")]
                )
        result = response["result"]
        self.assertIs(result["isError"], True)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertIn("db exploded", result["content"][0]["text"])


class ProtocolErrorTests(unittest.TestCase):
    def test_non_json_line_is_a_parse_error(self):
        with _ServerHarness() as harness:
            stdin = io.StringIO("this is not json\n")
            stdout = io.StringIO()
            harness.server.serve(stdin=stdin, stdout=stdout)
            [response] = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(response["error"]["code"], PARSE_ERROR)
        self.assertIsNone(response["id"])

    def test_non_object_message_is_invalid_request(self):
        with _ServerHarness() as harness:
            responses = _run_session(
                harness.server, [[1, 2, 3], "a string", {"jsonrpc": "2.0", "id": 4}]
            )
        self.assertEqual(responses[0]["error"]["code"], INVALID_REQUEST)
        self.assertIsNone(responses[0]["id"])
        self.assertEqual(responses[1]["error"]["code"], INVALID_REQUEST)
        self.assertEqual(responses[2]["error"]["code"], INVALID_REQUEST)

    def test_unknown_method_is_method_not_found(self):
        with _ServerHarness() as harness:
            [response] = _run_session(harness.server, [_request(1, "resources/list")])
        self.assertEqual(response["error"]["code"], METHOD_NOT_FOUND)

    def test_unknown_notification_is_ignored(self):
        with _ServerHarness() as harness:
            responses = _run_session(
                harness.server,
                [
                    {"jsonrpc": "2.0", "method": "notifications/cancelled"},
                    {"jsonrpc": "2.0", "method": "notifications/roots/list_changed"},
                    _request(1, "ping"),
                ],
            )
        self.assertEqual([response["id"] for response in responses], [1])

    def test_notification_without_method_is_ignored(self):
        with _ServerHarness() as harness:
            responses = _run_session(
                harness.server,
                [{"jsonrpc": "2.0", "params": {}}, _request(1, "ping")],
            )
        self.assertEqual([response["id"] for response in responses], [1])

    def test_notification_for_a_known_method_is_silent(self):
        with _ServerHarness() as harness:
            responses = _run_session(
                harness.server,
                [{"jsonrpc": "2.0", "method": "ping"}, _request(1, "ping")],
            )
        self.assertEqual([response["id"] for response in responses], [1])

    def test_explicit_null_arguments_fall_back_to_the_defaults(self):
        with _ServerHarness() as harness:
            responses = _run_session(
                harness.server,
                [
                    _request(1, "tools/call", {"name": "get_review_queue", "arguments": None}),
                    _tool_call(2, "get_review_queue", None),
                ],
            )
        for response in responses:
            self.assertIn("result", response, response)
        for response in responses:
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertLessEqual(payload["total_due"], 50)
            self.assertEqual(payload["total_due"], len(payload["items"]))

    def test_serve_defaults_to_the_process_stdio_streams(self):
        with _ServerHarness() as harness:
            with mock.patch("sys.stdin", io.StringIO(json.dumps(_request(1, "ping")) + "\n")):
                with mock.patch("sys.stdout", io.StringIO()) as stdout:
                    harness.server.serve()
        [line] = [line for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(json.loads(line)["result"], {})

    def test_blank_lines_are_skipped(self):
        with _ServerHarness() as harness:
            stdin = io.StringIO("\n   \n" + json.dumps(_request(1, "ping")) + "\n\n")
            stdout = io.StringIO()
            harness.server.serve(stdin=stdin, stdout=stdout)
            lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)

    def test_unexpected_handler_failure_is_an_internal_error(self):
        with _ServerHarness() as harness:
            with mock.patch.object(
                harness.server._toolkit,
                "definitions",
                side_effect=RuntimeError("boom"),
            ):
                [response] = _run_session(harness.server, [_request(1, "tools/list")])
        self.assertEqual(response["error"]["code"], -32603)
        self.assertIn("boom", response["error"]["message"])


class SafetyGateTests(unittest.TestCase):
    def test_refuses_startup_when_disabled(self):
        with self.assertRaises(MCPRefusalError) as ctx:
            build_server(_settings("/tmp/unused.db", enabled=False))
        self.assertIn("enable_mcp_server", str(ctx.exception))

    def test_refuses_startup_when_user_id_missing(self):
        with self.assertRaises(MCPRefusalError) as ctx:
            build_server(_settings("/tmp/unused.db", user_id=""))
        self.assertIn("mcp_server_user_id", str(ctx.exception))

    def test_refuses_startup_when_user_id_is_blank(self):
        with self.assertRaises(MCPRefusalError) as ctx:
            build_server(_settings("/tmp/unused.db", user_id="   "))
        self.assertIn("mcp_server_user_id", str(ctx.exception))

    def test_refusal_message_lists_every_reason(self):
        with self.assertRaises(MCPRefusalError) as ctx:
            build_server(_settings("/tmp/unused.db", enabled=False, user_id=""))
        message = str(ctx.exception)
        self.assertIn("enable_mcp_server", message)
        self.assertIn("mcp_server_user_id", message)

    def test_builds_server_and_strips_the_user_id_when_the_gate_passes(self):
        with _ServerHarness(user_id="  spaced-learner  ") as harness:
            self.assertEqual(harness.server.user_id, "spaced-learner")

    def test_main_refuses_and_never_serves(self):
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(os.path.join(td, "learning.db"), enabled=False)
            with mock.patch("app.mcp_server.server.get_settings", return_value=settings):
                with mock.patch.object(MCPServer, "serve") as serve:
                    with self.assertLogs("app.mcp_server.server", level="ERROR") as logs:
                        exit_code = main()
            serve.assert_not_called()
        self.assertEqual(exit_code, REFUSAL_EXIT_CODE)
        self.assertIn("startup refused", "\n".join(logs.output))

    def test_main_serves_when_the_gate_passes(self):
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(os.path.join(td, "learning.db"))
            with mock.patch("app.mcp_server.server.get_settings", return_value=settings):
                with mock.patch.object(MCPServer, "serve") as serve:
                    exit_code = main()
            serve.assert_called_once()
        self.assertEqual(exit_code, 0)


class EntryPointTests(unittest.TestCase):
    """``python -m app.mcp_server`` acceptance path over real stdin/stdout."""

    def _spawn_env(self, db_path: str, **extra) -> dict:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(BACKEND_ROOT),
            "STUDY_LEARNING_DB_PATH": db_path,
        }
        env.update(extra)
        return env

    def test_initialize_message_returns_a_valid_handshake_response(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            _seed_learner(db_path)
            proc = subprocess.run(
                [sys.executable, "-m", "app.mcp_server"],
                input=json.dumps(_initialize()) + "\n",
                capture_output=True,
                text=True,
                cwd=BACKEND_ROOT,
                env=self._spawn_env(
                    db_path,
                    STUDY_ENABLE_MCP_SERVER="true",
                    STUDY_MCP_SERVER_USER_ID=USER_ID,
                ),
                timeout=120,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        response = json.loads(lines[0])
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(response["id"], 1)
        self.assertEqual(response["result"]["serverInfo"]["name"], "ace-your-interview-mcp")
        self.assertIn("tools", response["result"]["capabilities"])

    def test_entry_point_refuses_when_the_gate_closes(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            proc = subprocess.run(
                [sys.executable, "-m", "app.mcp_server"],
                input=json.dumps(_initialize()) + "\n",
                capture_output=True,
                text=True,
                cwd=BACKEND_ROOT,
                env=self._spawn_env(db_path, STUDY_ENABLE_MCP_SERVER="false"),
                timeout=120,
            )
        self.assertEqual(proc.returncode, REFUSAL_EXIT_CODE)
        self.assertEqual(proc.stdout.strip(), "")
        self.assertIn("startup refused", proc.stderr)

    def test_entry_point_refuses_when_user_id_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            proc = subprocess.run(
                [sys.executable, "-m", "app.mcp_server"],
                input=json.dumps(_initialize()) + "\n",
                capture_output=True,
                text=True,
                cwd=BACKEND_ROOT,
                env=self._spawn_env(db_path, STUDY_ENABLE_MCP_SERVER="true"),
                timeout=120,
            )
        self.assertEqual(proc.returncode, REFUSAL_EXIT_CODE)
        self.assertEqual(proc.stdout.strip(), "")
        self.assertIn("mcp_server_user_id", proc.stderr)


class ToolkitUnitTests(unittest.TestCase):
    """Direct toolkit behaviour without the protocol loop."""

    def _toolkit(self, db_path: str) -> MCPToolkit:
        return MCPToolkit(
            learning_store=LearningStore(db_path),
            learning_planner=LearningPlannerStore(db_path),
            interview_store=InterviewStore(db_path),
            user_id=USER_ID,
        )

    def test_execute_rejects_unknown_tools_before_touching_a_store(self):
        with tempfile.TemporaryDirectory() as td:
            toolkit = self._toolkit(os.path.join(td, "learning.db"))
            with self.assertRaises(KeyError):
                toolkit.execute("not_a_tool", {})

    def test_execute_rejects_non_integer_arguments(self):
        with tempfile.TemporaryDirectory() as td:
            toolkit = self._toolkit(os.path.join(td, "learning.db"))
            with self.assertRaises(InvalidToolParams):
                toolkit.execute("get_review_queue", {"limit": "50"})
            with self.assertRaises(InvalidToolParams):
                toolkit.execute("get_study_plan", {"days": 2.5})

    def test_null_argument_falls_back_to_the_default(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "learning.db")
            store = LearningStore(db_path)
            _insert_due_cards(
                store,
                [(f"due-q-{idx:04d}", TOPIC_API, -1, 3, 1) for idx in range(60)],
            )
            toolkit = self._toolkit(db_path)
            [block] = toolkit.execute("get_review_queue", {"limit": None})
        payload = json.loads(block["text"])
        self.assertEqual(payload["total_due"], 50)


if __name__ == "__main__":
    unittest.main()
