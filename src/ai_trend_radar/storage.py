from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import Item


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS items (
                uid TEXT PRIMARY KEY,
                published_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            )"""
        )
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self.db.commit()
        self._migrate_legacy_github_trending()

    def upsert(self, items: list[Item]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        for item in items:
            self.db.execute(
                """INSERT INTO items(uid, published_at, payload, first_seen_at, last_seen_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(uid) DO UPDATE SET
                     published_at=excluded.published_at,
                     payload=excluded.payload,
                     last_seen_at=excluded.last_seen_at""",
                (item.uid, item.published_at.isoformat(), json.dumps(item.to_dict()), now, now),
            )
        self.db.commit()
        return len(items)

    def recent(self, as_of: datetime, days: int) -> list[Item]:
        start = (as_of - timedelta(days=days)).astimezone(timezone.utc).isoformat()
        end = as_of.astimezone(timezone.utc).isoformat()
        rows = self.db.execute(
            """SELECT payload, first_seen_at, last_seen_at
               FROM items WHERE published_at >= ? AND published_at <= ?
               ORDER BY published_at DESC""",
            (start, end),
        ).fetchall()
        items: list[Item] = []
        for payload, first_seen_at, last_seen_at in rows:
            item = Item.from_dict(json.loads(payload))
            item.metadata["first_seen_at"] = first_seen_at
            item.metadata["last_seen_at"] = last_seen_at
            items.append(item)
        return items

    def _migrate_legacy_github_trending(self) -> None:
        rows = self.db.execute(
            """SELECT uid, published_at, payload, first_seen_at, last_seen_at
               FROM items WHERE uid LIKE 'github-trending:%'"""
        ).fetchall()
        groups: dict[str, list[tuple[str, str, str, str, str]]] = {}
        pattern = re.compile(r"^github-trending:\d{4}-\d{2}-\d{2}:(.+)$")
        for row in rows:
            match = pattern.match(row[0])
            if match:
                groups.setdefault(f"github-trending:{match.group(1)}", []).append(row)
        for stable_uid, versions in groups.items():
            latest = max(versions, key=lambda row: row[1])
            item = Item.from_dict(json.loads(latest[2]))
            item.uid = stable_uid
            first_seen = min(row[3] for row in versions)
            last_seen = max(row[4] for row in versions)
            self.db.executemany("DELETE FROM items WHERE uid = ?", [(row[0],) for row in versions])
            self.db.execute(
                """INSERT INTO items(uid, published_at, payload, first_seen_at, last_seen_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(uid) DO UPDATE SET
                     published_at=excluded.published_at,
                     payload=excluded.payload,
                     first_seen_at=min(items.first_seen_at, excluded.first_seen_at),
                     last_seen_at=max(items.last_seen_at, excluded.last_seen_at)""",
                (stable_uid, latest[1], json.dumps(item.to_dict()), first_seen, last_seen),
            )
        if groups:
            self.db.commit()

    def get_embedding_blobs(self, cache_keys: list[str]) -> dict[str, bytes]:
        if not cache_keys:
            return {}
        result: dict[str, bytes] = {}
        # Stay below SQLite's host-parameter limit.
        for start in range(0, len(cache_keys), 500):
            batch = cache_keys[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self.db.execute(
                f"SELECT cache_key, vector FROM embeddings WHERE cache_key IN ({placeholders})",
                batch,
            ).fetchall()
            result.update((str(key), bytes(vector)) for key, vector in rows)
        return result

    def put_embedding_blobs(
        self,
        rows: list[tuple[str, str, int, bytes]],
    ) -> None:
        if not rows:
            return
        now = datetime.now(timezone.utc).isoformat()
        self.db.executemany(
            """INSERT INTO embeddings(cache_key, namespace, dimensions, vector, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(cache_key) DO UPDATE SET
                 vector=excluded.vector, created_at=excluded.created_at""",
            [(key, namespace, dimensions, vector, now) for key, namespace, dimensions, vector in rows],
        )
        self.db.commit()

    def embedding_cache_count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def close(self) -> None:
        self.db.close()
