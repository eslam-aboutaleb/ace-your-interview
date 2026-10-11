"""MCP stdio server: handshake, method dispatch, and the safety gate.

Transport: newline-delimited JSON-RPC 2.0 messages over stdin/stdout (see
:mod:`app.mcp_server.protocol` for the wire-format rationale). Implemented
handoff surface: ``initialize`` → ``initialized`` notification → ``ping``,
``tools/list``, ``tools/call``. The declared capabilities advertise **tools
only** — resources, prompts and sampling are intentionally absent, so clients
never attempt surfaces this server does not implement.

Protocol-version negotiation: when the client requests a version we echo it
back, otherwise we answer with :data:`SUPPORTED_PROTOCOL_VERSION`. The
implemented method set is the stable core shared by every MCP revision, so
echoing cannot make a newer client drive an unimplemented surface.

Safety model — single user or nothing
-------------------------------------
This server exposes one learner's private study history, so it is a
**single-user local bridge**, never a multi-tenant service:

* it refuses to start unless ``enable_mcp_server`` is true **and**
  ``mcp_server_user_id`` names a non-empty user id. Refusals are logged with
  the reason and the process exits non-zero — there is no degraded mode and
  no anonymous fallback;
* every tool result is scoped to that configured user id. No request can name
  or swap the user, so one MCP client cannot read another learner's data;
* the transport is stdio: the only clients that can reach it are processes on
  the same machine that the operator launched, so there is no network exposure
  to harden. Bind nothing, listen on nothing, keep the flag off in shared or
  deployed environments.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, IO, Sequence

from app.config import Settings, get_settings
from app.mcp_server.protocol import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SUPPORTED_PROTOCOL_VERSION,
    encode,
    error_response,
    is_notification,
    success_response,
)
from app.mcp_server.tools import InvalidToolParams, MCPToolkit, UnknownToolError
from app.services.interview_store import InterviewStore
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore

logger = logging.getLogger(__name__)

#: Exit code used when the single-user safety gate refuses startup.
REFUSAL_EXIT_CODE = 2

_MCP_SERVER_NAME = "ace-your-interview-mcp"


class MCPRefusalError(RuntimeError):
    """The safety gate refused to expose the MCP server."""


def _gate_reasons(settings: Settings) -> list[str]:
    """Why the server may not start, or an empty list when it may."""
    reasons: list[str] = []
    if not getattr(settings, "enable_mcp_server", False):
        reasons.append(
            "enable_mcp_server is false (set STUDY_ENABLE_MCP_SERVER=true to opt in)"
        )
    if not str(getattr(settings, "mcp_server_user_id", "") or "").strip():
        reasons.append(
            "mcp_server_user_id is empty (set STUDY_MCP_SERVER_USER_ID to the "
            "single learner this bridge is allowed to read)"
        )
    return reasons


def build_server(settings: Settings) -> MCPServer:
    """Construct the server or raise :class:`MCPRefusalError` with the reasons.

    Stores are built from ``settings.learning_db_path`` exactly like the
    FastAPI lifespan wiring in ``app/main.py``, so the MCP view and the REST
    view of a learner always read the same database.
    """
    reasons = _gate_reasons(settings)
    if reasons:
        raise MCPRefusalError("; ".join(reasons))
    user_id = str(settings.mcp_server_user_id).strip()
    db_path = settings.learning_db_path
    logger.info(
        "MCP stdio server enabled for single user id %r (db=%s)", user_id, db_path
    )
    return MCPServer(
        settings=settings,
        user_id=user_id,
        toolkit=MCPToolkit(
            learning_store=LearningStore(db_path),
            learning_planner=LearningPlannerStore(db_path),
            interview_store=InterviewStore(db_path),
            user_id=user_id,
        ),
    )


class MCPServer:
    """Minimal, dependency-free MCP server over newline-delimited JSON-RPC."""

    def __init__(self, *, settings: Settings, user_id: str, toolkit: MCPToolkit) -> None:
        self._settings = settings
        self._user_id = user_id
        self._toolkit = toolkit

    @property
    def user_id(self) -> str:
        return self._user_id

    # ── session loop ────────────────────────────────────────────

    def serve(
        self,
        *,
        stdin: IO[str] | None = None,
        stdout: IO[str] | None = None,
    ) -> None:
        """Read JSON-RPC lines until EOF, writing one response line each.

        Protocol traffic goes to stdout; logs go to stderr, so piped clients
        only ever parse protocol messages.
        """
        if stdin is None:
            stdin = sys.stdin
        if stdout is None:
            stdout = sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            response = self.handle_line(line)
            if response is None:
                continue
            stdout.write(encode(response))
            stdout.flush()
        logger.info("MCP stdio server stopped (stdin closed)")

    def handle_line(self, line: str) -> dict[str, Any] | None:
        """Handle one raw input line, returning the response or ``None``."""
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            logger.warning("MCP server received a non-JSON line")
            return error_response(None, PARSE_ERROR, "Parse error: invalid JSON")
        if not isinstance(message, dict):
            return error_response(None, INVALID_REQUEST, "Invalid Request: message must be a JSON object")
        return self.handle_message(message)

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one parsed JSON-RPC message."""
        request_id = message.get("id")
        method = message.get("method")
        notification = is_notification(message)

        if not isinstance(method, str) or not method:
            if notification:
                return None
            return error_response(request_id, INVALID_REQUEST, "Invalid Request: missing method")

        handler = {
            "initialize": self._handle_initialize,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
        }.get(method)
        if handler is None:
            if notification:
                # Unknown notifications (e.g. notifications/cancelled) are
                # simply ignored, like the initialized notification below.
                logger.debug("Ignoring unknown notification %s", method)
                return None
            return error_response(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")

        try:
            result = handler(message.get("params"))
        except InvalidToolParams as exc:
            return error_response(request_id, INVALID_PARAMS, str(exc))
        except UnknownToolError as exc:
            return error_response(request_id, INVALID_PARAMS, f"Unknown tool: {exc.args[0]}")
        except Exception as exc:  # defensive final net for the non-tool methods
            logger.exception("MCP %s failed", method)
            return error_response(request_id, -32603, f"Internal error: {exc}")

        if notification:
            # The only implemented notification is notifications/initialized;
            # acknowledged silently per the spec.
            return None
        return success_response(request_id, result)

    # ── method handlers ─────────────────────────────────────────

    def _handle_initialize(self, params: Any) -> dict[str, Any]:
        requested = None
        if isinstance(params, dict):
            requested = params.get("protocolVersion")
        protocol_version = requested if isinstance(requested, str) and requested.strip() else SUPPORTED_PROTOCOL_VERSION
        return {
            "protocolVersion": protocol_version.strip(),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": _MCP_SERVER_NAME,
                "version": str(getattr(self._settings, "app_version", "") or "0.0.0"),
            },
            "instructions": (
                "Read-only study tools for one configured learner. Call "
                "tools/list to discover them; results are scoped to the user id "
                "in STUDY_MCP_SERVER_USER_ID and contain no credentials."
            ),
        }

    def _handle_ping(self, params: Any) -> dict[str, Any]:
        return {}

    def _handle_tools_list(self, params: Any) -> dict[str, Any]:
        return {"tools": self._toolkit.definitions()}

    def _handle_tools_call(self, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise InvalidToolParams("tools/call params must be an object")
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise InvalidToolParams("tools/call requires a tool name")
        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise InvalidToolParams("tools/call arguments must be an object")
        try:
            content = self._toolkit.execute(name, arguments)
        except (InvalidToolParams, UnknownToolError):
            raise  # client-side mistakes are JSON-RPC errors, not tool errors
        except Exception as exc:
            # A store failure is a tool-execution error: report it in-band as
            # isError content so the agent can react, per the MCP spec.
            logger.exception("MCP tool %s failed", name)
            return {
                "content": [{"type": "text", "text": f"Tool {name} failed: {exc}"}],
                "isError": True,
            }
        return {"content": content, "isError": False}


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point: ``python -m app.mcp_server``.

    Refuses to serve unless the single-user gate passes; the refusal reason is
    logged and the exit code is :data:`REFUSAL_EXIT_CODE` so a supervisor can
    see the bridge never came up.
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    settings = get_settings()
    try:
        server = build_server(settings)
    except MCPRefusalError as exc:
        logger.error("MCP server startup refused: %s", exc)
        return REFUSAL_EXIT_CODE
    server.serve()
    return 0
