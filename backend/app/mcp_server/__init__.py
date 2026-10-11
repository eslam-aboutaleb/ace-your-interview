"""Real MCP (Model Context Protocol) server for external agents.

This package speaks the actual **Model Context Protocol** over stdio, so
external agents (Claude Desktop, Kilo, and any other MCP client) can drive the
learner journey. It is deliberately unrelated to
``app.services.mcp_gateway`` — that module is a bounded *tool runner* over
Tavily / Firecrawl / GitHub with its own flags, not an MCP implementation.

Package layout:

* :mod:`app.mcp_server.protocol` — JSON-RPC 2.0 message helpers and the
  newline-delimited framing constants.
* :mod:`app.mcp_server.tools` — the read-only, single-user tool catalog
  backed by the learning stores.
* :mod:`app.mcp_server.server` — the handshake, method dispatch, and the
  single-user safety gate.
* ``python -m app.mcp_server`` — the stdio entry point.
"""

from app.mcp_server.protocol import (
    JSONRPC_VERSION,
    SUPPORTED_PROTOCOL_VERSION,
    error_response,
    is_notification,
    success_response,
    text_content,
)
from app.mcp_server.server import (
    MCPRefusalError,
    MCPServer,
    build_server,
    main,
)
from app.mcp_server.tools import MCPToolkit, UnknownToolError

__all__ = [
    "JSONRPC_VERSION",
    "SUPPORTED_PROTOCOL_VERSION",
    "MCPRefusalError",
    "MCPServer",
    "MCPToolkit",
    "UnknownToolError",
    "build_server",
    "error_response",
    "is_notification",
    "main",
    "success_response",
    "text_content",
]
