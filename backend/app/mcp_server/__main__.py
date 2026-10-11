"""``python -m app.mcp_server`` — MCP stdio server entry point.

Run from the ``backend/`` directory (or with ``backend`` on ``PYTHONPATH``):

.. code-block:: bash

    STUDY_ENABLE_MCP_SERVER=true \\
    STUDY_MCP_SERVER_USER_ID=alice \\
    python -m app.mcp_server

The process speaks newline-delimited JSON-RPC 2.0 on stdin/stdout and exits
non-zero when the single-user safety gate refuses startup (see
:mod:`app.mcp_server.server`).
"""

from app.mcp_server.server import main

if __name__ == "__main__":
    raise SystemExit(main())
