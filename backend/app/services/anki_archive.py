"""Minimal Anki ``.apkg`` reader and writer.

An ``.apkg`` file is a zip containing ``collection.anki2``
(SQLite with ``col``/``notes``/``cards``/``revlog`` tables),
a ``media`` JSON manifest, and optional media blobs.

This module implements a custom minimal reader/writer over
the documented format (plan decision: avoid heavy deps). It
targets the Anki 2.1 collection format (``usn=-1``, ``mod``
as epoch seconds, collection ``ver=11``).

Import guards (plan decisions): 50 MB per import, 2000 cards
per import, 5000 zip entries, compressed-ratio <= 100x
(zip-bomb guard).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
import time
import zipfile
from typing import Any

# ── Import caps (plan decisions) ───────────────────
MAX_IMPORT_BYTES = 50 * 1024 * 1024
MAX_IMPORT_CARDS = 2000
MAX_ZIP_ENTRIES = 5000
MAX_COMPRESSION_RATIO = 100

FIELD_SEPARATOR = "\x1f"

# Anki ease: 1=Again, 2=Hard, 3=Good, 4=Easy
EASE_TO_RATING: dict[int, str] = {
    1: "again",
    2: "hard",
    3: "good",
    4: "easy",
}
RATING_TO_EASE: dict[str, int] = {
    "again": 1,
    "hard": 2,
    "good": 3,
    "easy": 4,
}

# Anki card/revlog type: 0=learn, 1=review, 2=relearn, 3=cram
TYPE_TO_STATE: dict[int, str] = {
    0: "learning",
    1: "review",
    2: "relearning",
    3: "learning",
}
STATE_TO_TYPE: dict[str, int] = {
    "new": 0,
    "learning": 0,
    "review": 1,
    "relearning": 2,
}

MEDIA_MANIFEST = "media"
COLLECTION_MEMBER = "collection.anki2"


class AnkiArchiveError(ValueError):
    """Raised when an ``.apkg`` cannot be parsed safely."""


# ── Writer ─────────────────────────────────────────
def _epoch_seconds(moment) -> int:
    return int(moment.timestamp())


def _epoch_millis(iso_text: str) -> int:
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(str(iso_text))
        return int(parsed.timestamp() * 1000)
    except (TypeError, ValueError):
        return int(time.time() * 1000)


def _field_checksum(text: str) -> int:
    """Anki-style field checksum (sum of char codes, mod 1000)."""
    return sum(ord(ch) for ch in text) % 1000


def _basic_model() -> dict[str, Any]:
    return {
        "1": {
            "id": 1,
            "name": "Basic",
            "did": "1",
            "usn": -1,
            "flds": [
                {
                    "name": "Front",
                    "ord": 0,
                    "sticky": False,
                    "rtl": False,
                    "font": "Arial",
                    "size": 20,
                    "media": [],
                },
                {
                    "name": "Back",
                    "ord": 1,
                    "sticky": False,
                    "rtl": False,
                    "font": "Arial",
                    "size": 20,
                    "media": [],
                },
            ],
            "tmpls": [
                {
                    "name": "Card 1",
                    "ord": 0,
                    "qfmt": "{{Front}}",
                    "afmt": (
                        "{{FrontSide}}\n\n<hr id=answer>\n\n{{Back}}"
                    ),
                    "did": None,
                    "bqfmt": "",
                    "bafmt": "",
                }
            ],
            "type": 0,
            "latexPre": "",
            "latexPost": "",
            "sortf": 0,
            "latexsvg": False,
            "req": [[0, "all", [0]]],
        }
    }


def _deck_config() -> dict[str, Any]:
    return {
        "1": {
            "id": 1,
            "name": "Default",
            "usn": -1,
            "newToday": [0, 0],
            "revToday": [0, 0],
            "lrnToday": [0, 0],
            "timeToday": [0, 0],
            "dyn": False,
            "desc": "",
            "mod": 0,
            "collapsed": False,
            "browserCollapsed": False,
            "new": {
                "perDay": 20,
                "delays": [1, 10],
                "separate": True,
                "order": 1,
                "initialFactor": 2500,
            },
            "rev": {
                "perDay": 200,
                "ease4": 1.3,
                "fuzz": 0.05,
                "ivlFct": 1.0,
                "maxIvl": 36500,
            },
            "lrn": {"perDay": 200, "delays": [1, 10]},
        }
    }


def write_apkg(
    deck_name: str,
    cards: list[dict[str, Any]],
    review_logs: list[dict[str, Any]] | None = None,
) -> bytes:
    """Build a minimal ``.apkg`` zip from app cards and FSRS review history.

    ``cards`` items: ``{front, back, tags?, fsrs?}`` where
    ``fsrs`` carries the scheduling counters (state, reps,
    lapses, scheduled_days, due_at). ``review_logs`` items are
    ``fsrs_review_log`` rows (``card_id``, ``rating``,
    ``state``, ``review_duration_ms``, ``scheduled_days``,
    ``elapsed_days``, ``created_at``).
    """
    name = (str(deck_name or "Deck").strip() or "Deck")[:200]
    logs_by_card: dict[str, list[dict[str, Any]]] = {}
    for entry in review_logs or []:
        logs_by_card.setdefault(str(entry.get("card_id", "")), []).append(
            entry
        )

    now = int(time.time())
    collection = sqlite3.connect(":memory:")
    collection.row_factory = sqlite3.Row
    collection.executescript(
        """
        CREATE TABLE col (
            id INTEGER PRIMARY KEY, crt INTEGER, mod INTEGER,
            scm INTEGER, ver INTEGER, dty INTEGER, usn INTEGER,
            ls INTEGER, conf TEXT, models TEXT, decks TEXT,
            dconf TEXT, tags TEXT
        );
        CREATE TABLE notes (
            id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER,
            mod INTEGER, usn INTEGER, tags TEXT, flds TEXT,
            sfld TEXT, csum INTEGER, flags INTEGER, data TEXT
        );
        CREATE TABLE cards (
            id INTEGER PRIMARY KEY, nid INTEGER, did INTEGER,
            ord INTEGER, mod INTEGER, usn INTEGER, type INTEGER,
            queue INTEGER, due INTEGER, ivl INTEGER, factor INTEGER,
            reps INTEGER, lapses INTEGER, left INTEGER, odue INTEGER,
            odid INTEGER, flags INTEGER, data TEXT
        );
        CREATE TABLE revlog (
            id INTEGER, cid INTEGER, usn INTEGER, ease INTEGER,
            ivl INTEGER, lastIvl INTEGER, factor INTEGER,
            time INTEGER, type INTEGER
        );
        """
    )
    collection.execute(
        """
        INSERT INTO col(
            id, crt, mod, scm, ver, dty, usn, ls, conf,
            models, decks, dconf, tags
        ) VALUES (1, ?, ?, ?, 11, 0, -1, 0, ?, ?, ?, ?, ?)
        """,
        (
            now,
            now,
            now,
            json.dumps({"nextPos": 0, "estTimes": True}),
            json.dumps(_basic_model()),
            json.dumps({"1": {"id": 1, "name": name, "mod": now, "usn": -1}}),
            json.dumps(_deck_config()),
            json.dumps({}),
        ),
    )

    for index, card in enumerate(cards[:MAX_IMPORT_CARDS]):
        front = str(card.get("front", ""))
        back = str(card.get("back", ""))
        tags = [
            re.sub(r"[^a-zA-Z0-9_-]", "_", str(t).strip())
            for t in (card.get("tags") or [])
            if str(t).strip()
        ]
        note_id = index + 1
        card_id = index + 1
        fsrs = card.get("fsrs") or {}
        state = str(fsrs.get("state", "new") or "new")
        reps = int(fsrs.get("reps", 0) or 0)
        lapses = int(fsrs.get("lapses", 0) or 0)
        scheduled_days = int(fsrs.get("scheduled_days", 0) or 0)
        flds = FIELD_SEPARATOR.join([front, back])
        collection.execute(
            """
            INSERT INTO notes(
                id, guid, mid, mod, usn, tags, flds, sfld,
                csum, flags, data
            ) VALUES (?, ?, 1, ?, -1, ?, ?, ?, ?, 0, '')
            """,
            (
                note_id,
                hashlib.sha1(flds.encode("utf-8")).hexdigest(),
                now,
                " ".join(tags),
                flds,
                front[:200],
                _field_checksum(flds),
            ),
        )
        card_type = STATE_TO_TYPE.get(state, 0)
        collection.execute(
            """
            INSERT INTO cards(
                id, nid, did, ord, mod, usn, type, queue, due,
                ivl, factor, reps, lapses, left, odue, odid,
                flags, data
            ) VALUES (?, ?, 1, 0, ?, -1, ?, ?, ?, ?, 0, ?, ?, 0, 0, 0, 0, '')
            """,
            (
                card_id,
                note_id,
                now,
                card_type,
                card_type,
                index + 1 if card_type == 0 else now // 86400,
                scheduled_days,
                reps,
                lapses,
            ),
        )
        for entry in logs_by_card.get(str(card.get("card_id", "")), []):
            rating = str(entry.get("rating", "again"))
            log_state = str(entry.get("state", state))
            collection.execute(
                """
                INSERT INTO revlog(
                    id, cid, usn, ease, ivl, lastIvl, factor,
                    time, type
                ) VALUES (?, ?, -1, ?, ?, 0, 0, ?, ?)
                """,
                (
                    _epoch_millis(entry.get("created_at", "")),
                    card_id,
                    RATING_TO_EASE.get(rating, 1),
                    int(entry.get("scheduled_days", 0) or 0),
                    int(entry.get("review_duration_ms", 0) or 0),
                    STATE_TO_TYPE.get(log_state, 0),
                ),
            )

    buffer = io.BytesIO()
    with tempfile.NamedTemporaryFile(
        suffix=".anki2", delete=False
    ) as tmp:
        tmp_path = tmp.name
    try:
        # The INSERTs above opened an implicit write
        # transaction; backup() deadlocks against an open
        # transaction on the source connection, so commit
        # first.
        collection.commit()
        disk = sqlite3.connect(tmp_path)
        collection.backup(disk)
        disk.close()
        with open(tmp_path, "rb") as f:
            collection_bytes = f.read()
    finally:
        os.unlink(tmp_path)
    with zipfile.ZipFile(
        buffer, "w", zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(COLLECTION_MEMBER, collection_bytes)
        archive.writestr(MEDIA_MANIFEST, json.dumps({}))
    collection.close()
    return buffer.getvalue()


# ── Reader ─────────────────────────────────────────
def _check_zip_guards(data: bytes) -> zipfile.ZipFile:
    if len(data) > MAX_IMPORT_BYTES:
        raise AnkiArchiveError(
            f"file exceeds the {MAX_IMPORT_BYTES // (1024 * 1024)} MB import cap"
        )
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise AnkiArchiveError(f"not a valid zip/apkg file: {e}") from e
    entries = archive.infolist()
    if len(entries) > MAX_ZIP_ENTRIES:
        raise AnkiArchiveError(
            f"zip has more than {MAX_ZIP_ENTRIES} entries"
        )
    total_uncompressed = sum(
        entry.file_size for entry in entries
    )
    total_compressed = sum(
        entry.compress_size for entry in entries
    )
    if (
        total_compressed > 0
        and total_uncompressed > total_compressed * MAX_COMPRESSION_RATIO
    ):
        raise AnkiArchiveError(
            "zip compression ratio exceeds the 100x zip-bomb guard"
        )
    return archive


def read_apkg(data: bytes) -> dict[str, Any]:
    """Parse ``.apkg`` bytes into decks, cards, revlog and media.

    Returns ``{"decks": [...], "media": {...}, "errors": [...]}``
    where each deck is ``{"name", "anki_deck_id", "cards"}`` and
    each card is ``{"front", "back", "tags", "review_logs"}``.
    Malformed pieces are captured per-file in ``errors`` rather
    than failing the whole archive.
    """
    archive = _check_zip_guards(data)
    errors: list[str] = []
    names = set(archive.namelist())
    if COLLECTION_MEMBER not in names:
        raise AnkiArchiveError("apkg is missing collection.anki2")

    collection_bytes = archive.read(COLLECTION_MEMBER)
    media_manifest: dict[str, str] = {}
    if MEDIA_MANIFEST in names:
        try:
            parsed = json.loads(archive.read(MEDIA_MANIFEST).decode("utf-8"))
            if isinstance(parsed, dict):
                media_manifest = {
                    str(k): str(v) for k, v in parsed.items()
                }
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            errors.append(f"media manifest unreadable: {e}")

    media_blobs: dict[str, bytes] = {}
    for key, filename in media_manifest.items():
        if filename in names:
            media_blobs[filename] = archive.read(filename)

    with tempfile.NamedTemporaryFile(
        suffix=".anki2", delete=False
    ) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as f:
            f.write(collection_bytes)
        conn = sqlite3.connect(tmp_path)
        conn.row_factory = sqlite3.Row
        try:
            return _parse_collection(conn, media_manifest, media_blobs, errors)
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        raise AnkiArchiveError(f"collection.anki2 is not readable: {e}") from e
    finally:
        os.unlink(tmp_path)


def _parse_collection(
    conn: sqlite3.Connection,
    media_manifest: dict[str, str],
    media_blobs: dict[str, bytes],
    errors: list[str],
) -> dict[str, Any]:
    decks: dict[str, dict[str, Any]] = {}
    deck_names: dict[int, str] = {}
    try:
        col_row = conn.execute("SELECT decks FROM col LIMIT 1").fetchone()
        if col_row is not None:
            parsed = json.loads(str(col_row["decks"] or "{}"))
            for key, value in parsed.items():
                if isinstance(value, dict):
                    deck_names[int(key)] = str(value.get("name", f"Deck {key}"))
    except (sqlite3.DatabaseError, json.JSONDecodeError, ValueError) as e:
        errors.append(f"col table unreadable: {e}")

    notes_by_id: dict[int, sqlite3.Row] = {}
    try:
        for row in conn.execute(
            "SELECT id, guid, mid, tags, flds, sfld FROM notes"
        ):
            notes_by_id[int(row["id"])] = row
    except sqlite3.DatabaseError as e:
        errors.append(f"notes table unreadable: {e}")

    revlog_by_card: dict[int, list[dict[str, Any]]] = {}
    try:
        for row in conn.execute(
            "SELECT id, cid, ease, ivl, lastIvl, factor, time, type FROM revlog"
        ):
            revlog_by_card.setdefault(int(row["cid"]), []).append(
                {
                    "id": int(row["id"]),
                    "ease": int(row["ease"]),
                    "ivl": int(row["ivl"] or 0),
                    "last_ivl": int(row["lastIvl"] or 0),
                    "factor": int(row["factor"] or 0),
                    "time_ms": int(row["time"] or 0),
                    "type": int(row["type"] or 0),
                }
            )
    except sqlite3.DatabaseError as e:
        errors.append(f"revlog table unreadable: {e}")

    card_count = 0
    try:
        for row in conn.execute(
            "SELECT id, nid, did, ord, type, queue, due, ivl, reps, lapses FROM cards"
        ):
            if card_count >= MAX_IMPORT_CARDS:
                errors.append(
                    f"import stopped at the {MAX_IMPORT_CARDS} card cap"
                )
                break
            card_count += 1
            note = notes_by_id.get(int(row["nid"]))
            if note is None:
                errors.append(
                    f"card {row['id']} references missing note {row['nid']}"
                )
                continue
            fields = str(note["flds"] or "").split(FIELD_SEPARATOR)
            front = fields[0] if fields else ""
            back = (
                fields[1]
                if len(fields) > 1
                else ""
            )
            if len(fields) > 2:
                back = "\n".join(fields[1:])
            tags = [
                t for t in re.split(r"\s+", str(note["tags"] or "").strip()) if t
            ]
            anki_deck_id = int(row["did"])
            deck_name = deck_names.get(anki_deck_id, f"Deck {anki_deck_id}")
            deck = decks.setdefault(
                str(anki_deck_id),
                {"name": deck_name, "anki_deck_id": str(anki_deck_id), "cards": []},
            )
            review_logs = []
            for entry in revlog_by_card.get(int(row["id"]), []):
                review_logs.append(
                    {
                        "rating": EASE_TO_RATING.get(entry["ease"], "again"),
                        "state": TYPE_TO_STATE.get(entry["type"], "learning"),
                        "review_duration_ms": entry["time_ms"],
                        "scheduled_days": entry["ivl"],
                        "elapsed_days": entry["last_ivl"],
                        "created_at": _iso_from_epoch_millis(entry["id"]),
                    }
                )
            deck["cards"].append(
                {
                    "front": front,
                    "back": back,
                    "tags": tags,
                    "review_logs": review_logs,
                }
            )
    except sqlite3.DatabaseError as e:
        errors.append(f"cards table unreadable: {e}")

    return {
        "decks": list(decks.values()),
        "media": {
            "manifest": media_manifest,
            "blobs": {
                name: blob
                for name, blob in media_blobs.items()
            },
        },
        "errors": errors,
    }


def _iso_from_epoch_millis(value: int) -> str:
    from datetime import UTC, datetime

    try:
        return datetime.fromtimestamp(
            int(value) / 1000.0, tz=UTC
        ).isoformat()
    except (OverflowError, OSError, ValueError):
        from datetime import datetime as _dt

        return _dt.now(UTC).isoformat()


def extract_media(
    archive_data: bytes,
    media_dir: str,
) -> list[str]:
    """Extract media blobs from an ``.apkg`` into ``media_dir``.

    Returns the list of written filenames. Only files named in
    the ``media`` manifest are extracted; everything else in the
    zip is ignored.
    """
    archive = _check_zip_guards(archive_data)
    written: list[str] = []
    try:
        manifest_raw = archive.read(MEDIA_MANIFEST)
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return written
    if not isinstance(manifest, dict):
        return written
    os.makedirs(media_dir, exist_ok=True)
    for filename in manifest.values():
        filename = str(filename)
        if not filename or filename in (".", ".."):
            continue
        if "/" in filename or "\\" in filename or filename.startswith("."):
            continue
        if filename not in archive.namelist():
            continue
        target = os.path.join(media_dir, filename)
        with open(target, "wb") as f:
            f.write(archive.read(filename))
        written.append(filename)
    return written
