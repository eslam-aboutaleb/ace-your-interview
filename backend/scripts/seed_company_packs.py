#!/usr/bin/env python3
"""Seed company interview packs into the learning database.

Usage:
    uv run python -m scripts.seed_company_packs [--db-path PATH] [--reset]

Writes the top-10 launch packs (Google, Amazon, Meta, Microsoft,
Apple, Netflix, Stripe, Airbnb, Uber, LinkedIn) into the
``company_packs`` table. Idempotent: existing packs are updated
in place. ``--reset`` wipes the table first so removed packs
disappear.
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.services.company_packs import DEFAULT_COMPANY_PACKS, CompanyPackStore


def seed_company_packs(db_path: str, *, reset: bool = False) -> int:
    store = CompanyPackStore(db_path)
    if reset:
        for pack in store.list_packs():
            store.delete_pack(pack["company"])
    seeded = 0
    for pack in DEFAULT_COMPANY_PACKS:
        store.upsert_pack(
            pack["company"],
            track=pack["track"],
            style_config=pack["style_config"],
        )
        seeded += 1
    return seeded


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed company interview packs")
    parser.add_argument(
        "--db-path",
        default="",
        help="Learning database path (default: STUDY_LEARNING_DB_PATH)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing packs before seeding",
    )
    args = parser.parse_args()

    db_path = args.db_path or get_settings().learning_db_path
    count = seed_company_packs(db_path, reset=args.reset)
    print(f"Seeded {count} company packs into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
