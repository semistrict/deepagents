"""The durable session and the task scheduler that runs on it, for one process."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Self, TypeVar, overload

from deepagents_durable import _core

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from deepagents_durable.runtime import AgentRuntime
    from deepagents_durable.tasks import TaskDefinition

T = TypeVar("T")


class Kernel:
    """One open durable session plus the scheduler that runs its tasks.

    Everything durable goes through `commit`. Work the process starts is a
    task: the scheduler reserves it, runs its phases, and resumes it from its
    last checkpoint after a restart.
    """

    def __init__(self, session: _core.Session) -> None:
        """Wrap an open session. Use `Kernel.open` instead."""
        self.session = session
        self.scheduler = _core.Scheduler(session)
        self.agents: AgentRuntime | None = None
        """The agent runtime serving this kernel, once an agent ran on it."""

    @classmethod
    async def open(cls, path: str | Path | None = None) -> Kernel:
        """Open a session over a pi-format SQLite file, or in memory without a path."""
        session = await _core.Session.open(None if path is None else str(path))
        return cls(session)

    @overload
    async def commit(self, change: Callable[[_core.Tx], Awaitable[T]]) -> T: ...

    @overload
    async def commit(self, change: Callable[[_core.Tx], T]) -> T: ...

    async def commit(self, change: Callable[[_core.Tx], T | Awaitable[T]]) -> T:
        """Run `change` in one transaction and commit it atomically.

        Returns what `change` returned, awaited when it is awaitable. When it
        raises, nothing is written.
        """
        tx = self.session.begin()
        try:
            result = change(tx)
            value: T = await result if inspect.isawaitable(result) else result
        except BaseException:
            tx.rollback()
            raise
        await tx.commit()
        return value

    def register(self, definition: TaskDefinition) -> None:
        """Run tasks of `definition.kind` in this process; a later registration replaces an earlier one."""
        self.scheduler.register(definition.kind, definition.handler())

    async def close(self) -> None:
        """Stop every invocation without writing outcomes, then close storage.

        Interrupted work stays pending and resumes when the file is reopened.
        """
        await self.scheduler.stop()
        await self.session.close()

    async def __aenter__(self) -> Self:
        """Use the kernel as an async context manager that closes on exit."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the kernel."""
        await self.close()
