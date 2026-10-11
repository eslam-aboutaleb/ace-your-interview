"""JSON-RPC 2.0 wire format for the MCP stdio server.

Wire-format choice — newline-delimited JSON (NDJSON) over stdio
---------------------------------------------------------------
Every message is exactly one JSON object serialised onto a single line:
requests and responses are written as ``json.dumps(message) + "\\n"`` and the
server reads stdin one line at a time. This is the MCP stdio transport's own
convention (the MCP specification frames stdio messages with newlines, unlike
the ``Content-Length`` header framing LSP uses), so a hand-rolled loop needs no
framing state machine: a line is either a complete message or a parse error.
Because ``json.dumps`` escapes embedded newlines inside string values, one
object per line is unambiguous.

Deliberate protocol-level simplifications, all safe for a tools-only server:

* **No batch requests.** A JSON array is rejected with ``-32600``; no MCP
  client in practice sends batches, and a tools-only server has no call that
  benefits from them.
* **Notifications get no response** (no ``id`` member), per JSON-RPC 2.0.
* **Logs go to stderr, never stdout.** stdout carries protocol traffic only.
"""

from __future__ import annotations

import json
from typing import Any

JSONRPC_VERSION = "2.0"

#: Protocol revision reported when the client does not request one. The
#: implemented surface (initialize / initialized / ping / tools/list /
#: tools/call) is the stable core of every MCP revision, so a conservative
#: baseline is the safe default.
SUPPORTED_PROTOCOL_VERSION = "2025-03-26"

# JSON-RPC 2.0 error codes used by this server.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def is_notification(message: dict[str, Any]) -> bool:
    """True when a request object omits the ``id`` member (a notification)."""
    return "id" not in message


def success_response(request_id: Any, result: Any) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 success response."""
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def text_content(payload: Any) -> dict[str, Any]:
    """Wrap a JSON-serialisable payload as an MCP text content block."""
    return {"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=True)}


def encode(message: dict[str, Any]) -> str:
    """Serialise one message onto its stdout line (no embedded newline)."""
    return json.dumps(message, ensure_ascii=True) + "\n"
