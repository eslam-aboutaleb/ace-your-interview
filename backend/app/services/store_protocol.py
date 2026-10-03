"""Async store protocol — the seam between routers and persistence.

Routers depend on this protocol, not on a concrete store, so the
SQLite-backed stores can later be replaced by Postgres-backed
implementations without touching call sites.

Postgres migration notes (deferred follow-up — deliberately NOT
implemented in this plan; this module only defines the contract):

1. Implement this protocol with an async driver (asyncpg or
   psycopg[async]). ``run_async`` becomes a native ``await`` on
   the driver call and the single-thread executor disappears.
2. Replace the ``CREATE TABLE IF NOT EXISTS`` DDL inside each
   store's private ``_init_db`` with versioned Alembic migrations.
   Keep ``init_tables`` as the idempotent "ensure schema exists"
   entry point for fresh deployments, and ``migrate`` as the
   versioned upgrade entry point that returns the number of
   applied migrations.
3. Preserve the composite primary keys (for example
   ``(user_id, question_id, topic_id)`` on question progress) so
   the existing per-user query shapes translate 1:1.
4. Multi-instance deployment additionally requires the Redis
   rate-limiter backend (``STUDY_RATE_LIMIT_BACKEND=redis`` with
   ``REDIS_URL``) and a shared database connection string; the
   in-memory limiter and the single SQLite connection do not
   share state across replicas.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")


class AsyncStoreProtocol(Protocol):
    """Interface every persistence store exposes to routers.

    The protocol is deliberately minimal: it captures the
    async-execution seam (``run_async``) and the schema lifecycle
    hooks (table init, migrations) that every store must provide.
    Domain-specific query methods stay on the concrete stores;
    routers that need them keep importing the concrete type until
    the Postgres port lands.
    """

    async def run_async(
        self, fn: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> T:
        """Run synchronous store work off the event loop."""
        ...

    def init_tables(self) -> None:
        """Idempotently create the schema on first use."""
        ...

    def migrate(self) -> int:
        """Apply pending schema migrations; return the count applied."""
        ...
