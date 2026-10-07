"""Threads kept one per durable session file, the way pi keeps one per session, with an index to list them.

A session has a single owner (see `Kernel`): one process opens a thread's
file at a time, and processes working on different threads never contend.
Agents built with a `Threads` as their checkpointer open each thread's
kernel on first use.

Listing threads does not open their files. A small SQLite index beside them,
written in ordinary transactions that any process may make, holds one row
per thread: when it was created and last updated, the metadata of its latest
run, its message count, and its first message. Each row is refreshed when a
run of the thread ends or the thread is updated.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepagents_durable.kernel import Kernel

INDEX = "index.sqlite"
_SAFE_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    first_message TEXT
);
CREATE INDEX IF NOT EXISTS threads_by_update ON threads (updated_at);
"""


@dataclass(frozen=True)
class ThreadSummary:
    """One thread as the index lists it."""

    thread_id: str
    created_at: str
    """ISO time the thread was first recorded."""
    updated_at: str
    """ISO time the thread was last recorded."""
    metadata: dict[str, Any]
    """The metadata of the thread's latest run's config."""
    message_count: int
    first_message: str | None
    """The text of the thread's first human message."""


def _json_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """The JSON-encodable part of a run's metadata."""
    kept: dict[str, Any] = {}
    for key, value in metadata.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        kept[key] = value
    return kept


def _summary(row: sqlite3.Row) -> ThreadSummary:
    return ThreadSummary(
        thread_id=row["thread_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=json.loads(row["metadata"]),
        message_count=row["message_count"],
        first_message=row["first_message"],
    )


class Threads:
    """A directory of threads, one durable session file each, and their index."""

    def __init__(self, directory: str | Path) -> None:
        """Keep threads in `directory`, creating it when first needed."""
        self.directory = Path(directory)
        self._kernels: dict[str, Kernel] = {}
        self._opening: dict[str, asyncio.Lock] = {}

    def path(self, thread_id: str) -> Path:
        """The session file of a thread: named by its ID, or by a hash of an ID unfit for a file name."""
        name = thread_id if _SAFE_NAME.fullmatch(thread_id) else hashlib.sha256(thread_id.encode()).hexdigest()
        return self.directory / f"{name}.sqlite"

    def exists(self, thread_id: str) -> bool:
        """Whether a thread has a session file."""
        return self.path(thread_id).exists()

    async def kernel(self, thread_id: str) -> Kernel:
        """The open kernel of a thread, opening its file on first use.

        Raises:
            SessionLocked: Another process has the thread open.
        """
        kernel = self._kernels.get(thread_id)
        if kernel is not None:
            return kernel
        async with self._opening.setdefault(thread_id, asyncio.Lock()):
            if thread_id not in self._kernels:
                self.directory.mkdir(parents=True, exist_ok=True)
                self._kernels[thread_id] = await Kernel.open(self.path(thread_id))
            return self._kernels[thread_id]

    async def release(self, thread_id: str) -> None:
        """Close a thread's kernel, if open here; its unfinished runs resume when it is next opened."""
        kernel = self._kernels.pop(thread_id, None)
        if kernel is not None:
            await kernel.close()

    async def close(self) -> None:
        """Close every thread opened here."""
        for thread_id in list(self._kernels):
            await self.release(thread_id)

    # The index.

    def _connect(self) -> sqlite3.Connection:
        self.directory.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.directory / INDEX, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(_SCHEMA)
        return connection

    def _record(self, thread_id: str, metadata: dict[str, Any] | None, message_count: int, first_message: str | None) -> None:
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO threads (thread_id, created_at, updated_at, metadata, message_count, first_message)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (thread_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    metadata = CASE WHEN ? THEN excluded.metadata ELSE threads.metadata END,
                    message_count = excluded.message_count,
                    first_message = excluded.first_message
                """,
                (thread_id, now, now, json.dumps(_json_metadata(metadata or {})), message_count, first_message, metadata is not None),
            )

    async def record(self, thread_id: str, *, metadata: dict[str, Any] | None, message_count: int, first_message: str | None) -> None:
        """Refresh a thread's index row; `metadata=None` keeps the metadata it has."""
        await asyncio.to_thread(self._record, thread_id, metadata, message_count, first_message)

    def _recent(self, limit: int | None) -> list[ThreadSummary]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM threads ORDER BY updated_at DESC LIMIT ?", (-1 if limit is None else limit,)).fetchall()
        return [_summary(row) for row in rows]

    async def recent(self, *, limit: int | None = None) -> list[ThreadSummary]:
        """The indexed threads, most recently updated first."""
        return await asyncio.to_thread(self._recent, limit)

    def _get(self, thread_id: str) -> ThreadSummary | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        return None if row is None else _summary(row)

    async def get(self, thread_id: str) -> ThreadSummary | None:
        """A thread's index row."""
        return await asyncio.to_thread(self._get, thread_id)

    def _forget(self, thread_id: str) -> bool:
        with closing(self._connect()) as connection, connection:
            indexed = connection.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,)).rowcount > 0
        removed = False
        path = self.path(thread_id)
        for suffix in ("", "-wal", "-shm", ".lock"):
            file = path.with_name(path.name + suffix)
            if file.exists():
                file.unlink()
                removed = True
        return indexed or removed

    async def delete(self, thread_id: str) -> bool:
        """Delete a thread's file and index row; returns whether there was anything to delete.

        Raises:
            SessionLocked: Another process has the thread open.
        """
        await self.release(thread_id)
        if self.exists(thread_id):
            # Opening takes the thread's lock, proving no other process holds it.
            kernel = await Kernel.open(self.path(thread_id))
            await kernel.close()
        return await asyncio.to_thread(self._forget, thread_id)
