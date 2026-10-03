"""A LangGraph store kept in the kernel's SQLite file.

Items are session documents of kind `lc.store`, one per namespace and key, so
they outlive threads and processes like everything else in the file. Only
this process writes the file, so the store answers reads and searches from
memory, loaded when it opens. An async write is durable when it returns. A
synchronous write made on the kernel's event loop cannot wait for its commit
without blocking that loop, so it commits in the background.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from langgraph.store.base import Item, PutOp
from langgraph.store.memory import InMemoryStore

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langgraph.store.base import Op, Result

    from deepagents_durable import _core
    from deepagents_durable.kernel import Kernel

logger = logging.getLogger(__name__)

STORE = "lc.store"
SESSION = {"kind": "session"}

Write = tuple[tuple[str, ...], str, Item | None, bool]
"""`(namespace, key, item or None when deleted, whether a document stored it before)`."""


def _doc_key(namespace: tuple[str, ...], key: str) -> str:
    return json.dumps([*namespace, key])


class DurableStore(InMemoryStore):
    """A `BaseStore` whose items are documents in a kernel's session. Vector search is not supported."""

    def __init__(self, kernel: Kernel) -> None:
        """An empty store on `kernel`. Use `DurableStore.open` to load the stored items."""
        super().__init__()
        self.kernel = kernel
        self.loop = asyncio.get_running_loop()
        self._background: set[asyncio.Task[None]] = set()

    @classmethod
    async def open(cls, kernel: Kernel) -> DurableStore:
        """The store of `kernel`'s session, with every stored item loaded."""
        store = cls(kernel)
        session = kernel.session
        for record in await session.docs(SESSION, STORE):
            stored = await session.doc(STORE, SESSION, record["key"])
            if stored is not None:
                value = stored["value"]
                namespace = tuple(value["namespace"])
                created, updated = datetime.fromisoformat(value["createdAt"]), datetime.fromisoformat(value["updatedAt"])
                store._data[namespace][value["key"]] = Item(
                    value=value["value"], key=value["key"], namespace=namespace, created_at=created, updated_at=updated
                )
        return store

    def _existing(self, ops: list[Op]) -> set[tuple[tuple[str, ...], str]]:
        """Which put targets hold an item now, before the batch applies."""
        return {(op.namespace, op.key) for op in ops if isinstance(op, PutOp) and op.key in self._data.get(op.namespace, {})}

    def _writes(self, ops: list[Op], existing: set[tuple[tuple[str, ...], str]]) -> list[Write]:
        """The batch's puts as they ended up in memory, last one per item."""
        targets = dict.fromkeys((op.namespace, op.key) for op in ops if isinstance(op, PutOp))
        return [(namespace, key, self._data.get(namespace, {}).get(key), (namespace, key) in existing) for namespace, key in targets]

    async def _commit(self, writes: list[Write]) -> None:
        def change(tx: _core.Tx) -> None:
            for namespace, key, item, stored in writes:
                if item is None:
                    if stored:
                        tx.retire_doc(STORE, SESSION, _doc_key(namespace, key))
                    continue
                value = {
                    "namespace": list(namespace),
                    "key": key,
                    "value": item.value,
                    "createdAt": item.created_at.isoformat(),
                    "updatedAt": item.updated_at.isoformat(),
                }
                tx.put_doc(STORE, SESSION, value, key=_doc_key(namespace, key))

        if writes:
            await self.kernel.commit(change)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Apply `ops`; puts commit before returning, or in the background on the kernel's loop."""
        ops = list(ops)
        existing = self._existing(ops)
        results = super().batch(ops)
        writes = self._writes(ops, existing)
        if not writes:
            return results
        try:
            on_loop = asyncio.get_running_loop() is self.loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            task = self.loop.create_task(self._commit(writes))
            self._background.add(task)
            task.add_done_callback(self._committed)
        else:
            asyncio.run_coroutine_threadsafe(self._commit(writes), self.loop).result()
        return results

    def _committed(self, task: asyncio.Task[None]) -> None:
        self._background.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("A store write was not saved", exc_info=task.exception())

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Apply `ops`; puts are durable when this returns."""
        ops = list(ops)
        existing = self._existing(ops)
        results = await super().abatch(ops)
        await self._commit(self._writes(ops, existing))
        return results

    async def flush(self) -> None:
        """Wait for writes committing in the background."""
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
