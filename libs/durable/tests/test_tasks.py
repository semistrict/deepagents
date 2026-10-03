"""Durable task scheduling over the Rust kernel."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from deepagents_durable.kernel import Kernel
from deepagents_durable.tasks import Invocation, TaskDefinition

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_durable import _core


async def _no_abort(invocation: Invocation) -> None:
    async with invocation.step() as step:
        step.aborted()


async def _start(kernel: Kernel, kind: str, input_: object, checkpoint: dict) -> int:
    def create(tx: _core.Tx) -> int:
        root = tx.create_root()
        return tx.create_task(root, kind, input_, checkpoint)

    return await kernel.commit(create)


async def _tick(invocation: Invocation) -> None:
    n = invocation.checkpoint["n"]
    async with invocation.step() as step:
        if n == invocation.input:
            step.finish(n * 10)
        else:
            step.advance({"phase": "tick", "n": n + 1})


COUNTER = TaskDefinition(kind="test.counter", phases={"tick": _tick}, abort=_no_abort)


async def test_phases_run_until_the_task_finishes() -> None:
    async with await Kernel.open() as kernel:
        kernel.register(COUNTER)
        task = await _start(kernel, "test.counter", 3, {"phase": "tick", "n": 1})
        settled = await kernel.session.wait_task(task)
    assert settled["state"] == {"status": "terminal", "outcome": {"status": "completed", "result": 30}}


async def test_a_reopened_session_resumes_at_the_last_checkpoint(tmp_path: Path) -> None:
    file = tmp_path / "session.sqlite"
    reached = asyncio.Event()

    async def stall(invocation: Invocation) -> None:
        async with invocation.step() as step:
            step.advance({"phase": "effect", "saved": "before the crash"})
        reached.set()
        await asyncio.Event().wait()

    kernel = await Kernel.open(file)
    kernel.register(TaskDefinition(kind="test.work", phases={"start": stall}, abort=_no_abort))
    task = await _start(kernel, "test.work", None, {"phase": "start"})
    await reached.wait()
    await kernel.close()

    async def effect(invocation: Invocation) -> None:
        async with invocation.step() as step:
            step.finish(invocation.checkpoint["saved"])

    async with await Kernel.open(file) as kernel:
        kernel.register(TaskDefinition(kind="test.work", phases={"start": stall, "effect": effect}, abort=_no_abort))
        settled = await kernel.session.wait_task(task)
    assert settled["state"]["outcome"] == {"status": "completed", "result": "before the crash"}


async def test_a_parent_waits_for_its_children() -> None:
    async def spawn(invocation: Invocation) -> None:
        async with invocation.step() as step:
            children = [
                step.tx.create_task(invocation.conversation, "test.counter", n, {"phase": "tick", "n": 1}, owner=invocation.id) for n in (1, 2, 3)
            ]
            step.wait(children, {"phase": "sum", "children": children})

    async def total(invocation: Invocation) -> None:
        results = []
        for child in invocation.checkpoint["children"]:
            record = await invocation.session.task(child)
            results.append(record["state"]["outcome"]["result"])
        async with invocation.step() as step:
            step.finish(sum(results))

    async with await Kernel.open() as kernel:
        kernel.register(COUNTER)
        kernel.register(TaskDefinition(kind="test.parent", phases={"spawn": spawn, "sum": total}, abort=_no_abort))
        task = await _start(kernel, "test.parent", None, {"phase": "spawn"})
        settled = await kernel.session.wait_task(task)
    assert settled["state"]["outcome"] == {"status": "completed", "result": 60}


async def test_abort_stops_running_work_and_runs_handlers_bottom_up() -> None:
    order: list[str] = []
    child_started = asyncio.Event()

    async def spawn(invocation: Invocation) -> None:
        async with invocation.step() as step:
            child = step.tx.create_task(invocation.conversation, "test.child", None, {"phase": "block"}, owner=invocation.id)
            step.wait([child], {"phase": "never"})

    async def block(invocation: Invocation) -> None:
        async with invocation.step() as step:
            step.advance({"phase": "block", "started": True})
        child_started.set()
        await asyncio.Event().wait()

    def recorder(name: str):
        async def abort(invocation: Invocation) -> None:
            order.append(name)
            async with invocation.step() as step:
                step.aborted("stopped")

        return abort

    async with await Kernel.open() as kernel:
        kernel.register(TaskDefinition(kind="test.parent", phases={"spawn": spawn}, abort=recorder("parent")))
        kernel.register(TaskDefinition(kind="test.child", phases={"block": block}, abort=recorder("child")))
        parent = await _start(kernel, "test.parent", None, {"phase": "spawn"})
        await child_started.wait()
        await kernel.commit(lambda tx: tx.abort_task(parent))
        settled = await kernel.session.wait_task(parent)
    assert order == ["child", "parent"]
    assert settled["state"]["outcome"] == {"status": "aborted", "reason": "stopped"}


async def test_a_raising_phase_faults_and_runs_the_fault_hook() -> None:
    async def explode(invocation: Invocation) -> None:
        msg = f"task {invocation.id} cannot continue"
        raise ValueError(msg)

    async def on_fault(tx: _core.Tx, task: dict, message: str) -> None:
        tx.append_entry(task["conversationId"], "test.fault", {"data": message})

    async with await Kernel.open() as kernel:
        kernel.register(TaskDefinition(kind="test.bad", phases={"start": explode}, abort=_no_abort, on_fault=on_fault))
        task = await _start(kernel, "test.bad", None, {"phase": "start"})
        settled = await kernel.session.wait_task(task)
        entries = await kernel.session.entries(settled["conversationId"])
    message = "ValueError('task 2 cannot continue')"
    assert settled["state"]["outcome"] == {"status": "faulted", "error": {"message": message}}
    assert [stored["entry"]["data"] for stored in entries] == [message]


async def test_a_phase_that_commits_nothing_faults() -> None:
    async def idle(_: Invocation) -> None:
        return

    async with await Kernel.open() as kernel:
        kernel.register(TaskDefinition(kind="test.idle", phases={"start": idle}, abort=_no_abort))
        task = await _start(kernel, "test.idle", None, {"phase": "start"})
        settled = await kernel.session.wait_task(task)
    assert settled["state"]["outcome"]["status"] == "faulted"
