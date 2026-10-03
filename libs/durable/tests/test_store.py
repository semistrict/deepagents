"""The LangGraph store kept in the kernel's file."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from deepagents_durable import DurableStore, Kernel

if TYPE_CHECKING:
    from pathlib import Path


async def test_items_outlive_the_process(tmp_path: Path) -> None:
    file = tmp_path / "store.sqlite"
    async with await Kernel.open(file) as kernel:
        store = await DurableStore.open(kernel)
        await store.aput(("approval", "t1"), "mode", {"mode": "auto"})
        await store.aput(("approval", "t2"), "mode", {"mode": "manual"})
        await store.aput(("counters",), "c", {"n": 1})
        await store.adelete(("counters",), "c")
    async with await Kernel.open(file) as kernel:
        store = await DurableStore.open(kernel)
        found = await store.aget(("approval", "t1"), "mode")
        searched = await store.asearch(("approval",), filter={"mode": "manual"})
        gone = await store.aget(("counters",), "c")
    assert found is not None
    assert found.value == {"mode": "auto"}
    assert [(item.namespace, item.key) for item in searched] == [(("approval", "t2"), "mode")]
    assert gone is None


async def test_a_synchronous_write_on_the_loop_is_saved_in_the_background(tmp_path: Path) -> None:
    file = tmp_path / "store.sqlite"
    async with await Kernel.open(file) as kernel:
        store = await DurableStore.open(kernel)
        store.put(("approval",), "mode", {"mode": "yolo"})
        assert store.get(("approval",), "mode").value == {"mode": "yolo"}  # ty: ignore[possibly-missing-attribute]
        await store.flush()
    async with await Kernel.open(file) as kernel:
        reopened = await DurableStore.open(kernel)
    assert reopened.get(("approval",), "mode").value == {"mode": "yolo"}  # ty: ignore[possibly-missing-attribute]


async def test_a_synchronous_write_off_the_loop_is_saved_before_returning(tmp_path: Path) -> None:
    file = tmp_path / "store.sqlite"
    async with await Kernel.open(file) as kernel:
        store = await DurableStore.open(kernel)
        await asyncio.to_thread(store.put, ("approval",), "mode", {"mode": "manual"})
    async with await Kernel.open(file) as kernel:
        reopened = await DurableStore.open(kernel)
    assert reopened.get(("approval",), "mode").value == {"mode": "manual"}  # ty: ignore[possibly-missing-attribute]
