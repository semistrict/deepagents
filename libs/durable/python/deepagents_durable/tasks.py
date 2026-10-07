"""Durable tasks, scheduled by the Rust kernel and implemented in Python.

A task's checkpoint names its next phase. The kernel's scheduler reserves a
pending task, calls the matching phase handler, and keeps going while each
phase commits progress through `Invocation.step()`. After a crash the task is
pending again at its last checkpoint and that phase simply runs again.
Aborts flow down the ownership tree; the scheduler cancels a marked run
invocation's coroutine and later runs the abort handler, bottom-up.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

    from deepagents_durable import _core

Record = dict[str, Any]


class Invocation:
    """One reserved run or abort of a task, and the only way its code commits."""

    def __init__(self, core: _core.Invocation) -> None:
        """Wrap the kernel's invocation."""
        self._core = core
        self._task: Record | None = None

    @property
    def task(self) -> Record:
        """The task record as of this invocation's last commit."""
        if self._task is None:
            self._task = self._core.task
        return self._task

    @property
    def mode(self) -> str:
        """`"run"` or `"abort"`."""
        return self._core.mode

    @property
    def session(self) -> _core.Session:
        """The session the task lives in."""
        return self._core.session

    @property
    def id(self) -> int:
        """The task ID."""
        return self.task["id"]

    @property
    def conversation(self) -> int:
        """The conversation the task belongs to."""
        return self.task["conversationId"]

    @property
    def input(self) -> Any:  # noqa: ANN401  # any JSON value
        """The task's input."""
        return self.task["input"]

    @property
    def checkpoint(self) -> Record:
        """Where the task resumes."""
        return self.task["state"]["checkpoint"]

    @contextlib.asynccontextmanager
    async def step(self) -> AsyncIterator[_core.Step]:
        """Commit the block's writes and the step's new task state atomically.

        Raises:
            InvocationEnded: At commit, when the task is no longer running under
                this invocation, or a run invocation's task was marked for abort.
        """
        step = self._core.step()
        try:
            yield step
        except BaseException:
            step.rollback()
            raise
        try:
            await step.commit()
        finally:
            self._task = None


@dataclass(frozen=True)
class TaskDefinition:
    """A kind of durable task: one handler per checkpoint phase, plus an abort handler.

    `checkpoint["phase"]` selects the handler. A handler must commit progress
    through `Invocation.step()` before returning, or the task faults.
    `on_fault` runs inside the commit that faults a task of this kind, so it
    can settle whatever the task was responsible for.
    """

    kind: str
    phases: Mapping[str, Callable[[Invocation], Awaitable[None]]]
    abort: Callable[[Invocation], Awaitable[None]]
    on_fault: Callable[[_core.Tx, Record, str], Awaitable[None]] | None = None

    def handler(self) -> object:
        """The object the kernel's scheduler calls for tasks of this kind."""
        return _FaultingHandler(self) if self.on_fault is not None else _Handler(self)


class _Handler:
    def __init__(self, definition: TaskDefinition) -> None:
        self.definition = definition

    async def run(self, core: _core.Invocation) -> None:
        invocation = Invocation(core)
        phase = invocation.checkpoint.get("phase")
        handler = self.definition.phases.get(phase)
        if handler is None:
            msg = f"task kind {self.definition.kind} has no phase {phase!r}"
            raise LookupError(msg)
        await handler(invocation)

    async def abort(self, core: _core.Invocation) -> None:
        await self.definition.abort(Invocation(core))


class _FaultingHandler(_Handler):
    async def fault(self, tx: _core.Tx, task: Record, message: str) -> None:
        await self.definition.on_fault(tx, task, message)  # ty: ignore[call-non-callable]  # only built with on_fault
