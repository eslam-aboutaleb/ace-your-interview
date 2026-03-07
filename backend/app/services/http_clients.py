"""Shared outbound HTTP clients."""

from __future__ import annotations

import threading

import httpx

_oauth_http_client: httpx.AsyncClient | None = None
_client_lock = threading.Lock()


def get_oauth_http_client() -> httpx.AsyncClient:
    global _oauth_http_client
    with _client_lock:
        if _oauth_http_client is None or _oauth_http_client.is_closed:
            _oauth_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(20.0, connect=10.0),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
        return _oauth_http_client


async def close_http_clients() -> None:
    global _oauth_http_client
    client: httpx.AsyncClient | None = None
    with _client_lock:
        client = _oauth_http_client
        _oauth_http_client = None
    if client is not None and not client.is_closed:
        await client.aclose()
