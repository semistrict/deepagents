"""Threads kept one per session file, and their index."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import deepagents
import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from deepagents_durable import Threads, agent_factory, create_agent
from deepagents_durable._core import SessionLocked
from tests.fakes import ScriptedModel, ai

if TYPE_CHECKING:
    from pathlib import Path


def _config(thread_id: str, **metadata: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "metadata": metadata}


async def test_each_thread_has_its_own_file_and_an_index_row(tmp_path: Path) -> None:
    threads = Threads(tmp_path)
    agent = create_agent(ScriptedModel(script=[ai(text="Hi.")]), [], checkpointer=threads)
    try:
        await agent.ainvoke({"messages": [HumanMessage("first thread")]}, _config("a", cwd="/work/a"))
        await agent.ainvoke({"messages": [HumanMessage("second thread")]}, _config("b", cwd="/work/b"))
        await agent.ainvoke({"messages": [HumanMessage("again")]}, _config("a", cwd="/work/a2"))
        missing = await agent.aget_state(_config("never"))
        listed = await threads.recent()
    finally:
        await threads.close()
    files = await asyncio.to_thread(lambda: sorted(path.name for path in tmp_path.glob("*.sqlite")))
    assert files == ["a.sqlite", "b.sqlite", "index.sqlite"]
    assert [(row.thread_id, row.metadata["cwd"], row.message_count, row.first_message) for row in listed] == [
        ("a", "/work/a2", 4, "first thread"),
        ("b", "/work/b", 2, "second thread"),
    ]
    assert missing.values == {}
    assert not threads.exists("never")


async def test_a_thread_is_open_in_one_place_at_a_time(tmp_path: Path) -> None:
    mine, theirs = Threads(tmp_path), Threads(tmp_path)
    try:
        await mine.kernel("shared")
        with pytest.raises(SessionLocked):
            await theirs.kernel("shared")
        await theirs.kernel("other")
        await mine.release("shared")
        await theirs.kernel("shared")
    finally:
        await mine.close()
        await theirs.close()


async def test_deleting_a_thread_removes_its_file_and_row(tmp_path: Path) -> None:
    threads = Threads(tmp_path)
    agent = create_agent(ScriptedModel(script=[ai(text="Hi.")]), [], checkpointer=threads)
    await agent.ainvoke({"messages": [HumanMessage("hello")]}, _config("gone"))
    assert await threads.delete("gone")
    assert not threads.exists("gone")
    assert await threads.recent() == []
    assert not await threads.delete("gone")


async def test_an_app_building_its_own_deep_agent_keeps_threads_in_files(tmp_path: Path) -> None:
    threads, store = Threads(tmp_path), InMemoryStore()
    agent = deepagents.create_deep_agent(ScriptedModel(script=[ai(text="Hi.")]), agent_factory=agent_factory(threads, store=store))
    try:
        result = await agent.ainvoke({"messages": [HumanMessage("hello")]}, _config("app"))
    finally:
        await threads.close()
    assert [message.text for message in result["messages"]] == ["hello", "Hi."]
    assert agent.spec.store is store
    assert threads.exists("app")
