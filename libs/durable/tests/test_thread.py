"""Threads store messages as entries and edits, and read back exactly what was applied."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired

from langchain.agents.middleware.types import AgentState
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from deepagents_durable import thread
from deepagents_durable.kernel import Kernel
from deepagents_durable.schema import Schema
from deepagents_durable.thread import ThreadState


def _merge_counts(current: dict[str, int], update: dict[str, int]) -> dict[str, int]:
    return {key: current.get(key, 0) + update.get(key, 0) for key in current.keys() | update.keys()}


class Counted(AgentState):
    counts: NotRequired[Annotated[dict[str, int], _merge_counts]]


async def _roundtrip(updates: list[list[tuple[str, Any]]]) -> tuple[ThreadState, ThreadState, list[str]]:
    schema = Schema([Counted])
    state = ThreadState(schema)
    async with await Kernel.open() as kernel:

        def create(tx: Any) -> int:  # noqa: ANN401
            return thread.create(tx, "t")

        conversation = await kernel.commit(create)
        for step in updates:
            change = state.apply(step)
            await kernel.commit(lambda tx, change=change: state.persist(tx, conversation, change))
        loaded = await thread.load(kernel.session, conversation, schema)
        kinds = [stored["entry"]["kind"] for stored in await kernel.session.entries(conversation)]
    return state, loaded, kinds


async def test_replaced_and_removed_messages_are_edits_that_reload_identically() -> None:
    first, second = HumanMessage("hi", id="m1"), AIMessage("hello", id="m2")
    state, loaded, kinds = await _roundtrip(
        [
            [("messages", [first, second]), ("counts", {"a": 1})],
            [("messages", [AIMessage("hello, edited", id="m2")]), ("counts", {"a": 2, "b": 1})],
            [("messages", [HumanMessage("more", id="m3"), RemoveMessage(id="m1")])],
        ]
    )
    assert [(m.id, m.content) for m in loaded.messages] == [(m.id, m.content) for m in state.messages] == [("m2", "hello, edited"), ("m3", "more")]
    assert loaded.fields == state.fields == {"counts": {"a": 3, "b": 1}}
    assert kinds == ["lc.message", "lc.message", "lc.edit", "lc.message", "lc.edit"]


def test_state_model_context_matches_a_full_scan() -> None:
    call = {"name": "search", "args": {}, "id": "c1", "type": "tool_call"}
    state = ThreadState(Schema([Counted]))
    state.apply([("messages", [HumanMessage("go", id="h1"), AIMessage("", tool_calls=[call], id="a1")])])
    assert state.model_context() == thread.model_context(state.messages) == state.messages
    state.apply([("messages", [HumanMessage("never mind", id="h2")])])
    assert [m.type for m in state.model_context()] == ["human", "ai", "tool", "human"]
    state.apply([("messages", [ToolMessage("found", tool_call_id="c1", id="t1")])])
    assert state.model_context() == state.messages
    state.apply([("messages", [RemoveMessage(id="t1")])])
    assert [m.type for m in state.model_context()] == ["human", "ai", "tool", "human"]
