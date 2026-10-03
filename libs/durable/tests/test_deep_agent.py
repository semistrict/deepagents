"""The full Deep Agents middleware stack on the durable runtime, against the Pregel runtime."""

from __future__ import annotations

from typing import Any

import deepagents
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import HumanMessage

from deepagents_durable import Kernel, create_deep_agent
from tests.fakes import ScriptedModel, ai, call
from tests.test_agent import _shape


def _todos_and_files() -> list[Any]:
    return [
        ai(call("write_todos", "t1", todos=[{"content": "write notes", "status": "in_progress"}])),
        ai(call("write_file", "w1", file_path="/notes.md", content="hello\nworld\n")),
        ai(call("read_file", "r1", file_path="/notes.md")),
        ai(call("edit_file", "e1", file_path="/notes.md", old_string="world", new_string="durable world")),
        ai(text="Notes written."),
    ]


async def _both(script: list[Any], **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    async with await Kernel.open() as kernel:
        durable = create_deep_agent(ScriptedModel(script=script), checkpointer=kernel, **kwargs)
        ours = await durable.ainvoke({"messages": [HumanMessage("take notes")]})
    pregel = deepagents.create_deep_agent(ScriptedModel(script=script), **kwargs)
    theirs = await pregel.ainvoke({"messages": [HumanMessage("take notes")]})
    return ours, theirs


async def test_todos_and_state_backend_files_match_pregel() -> None:
    ours, theirs = await _both(_todos_and_files(), middleware=[TodoListMiddleware()])
    assert _shape(ours["messages"]) == _shape(theirs["messages"])
    assert ours["todos"] == theirs["todos"] == [{"content": "write notes", "status": "in_progress"}]
    assert ours["files"]["/notes.md"]["content"] == theirs["files"]["/notes.md"]["content"] == "hello\ndurable world\n"


async def test_general_purpose_subagent_matches_pregel() -> None:
    script = [
        ai(call("task", "s1", description="Write /sub.md saying hi", subagent_type="general-purpose")),
        ai(call("write_file", "w1", file_path="/sub.md", content="hi\n")),
        ai(text="Wrote it."),
        ai(text="The subagent wrote /sub.md."),
    ]
    ours, theirs = await _both(script)
    assert _shape(ours["messages"]) == _shape(theirs["messages"])
    assert ours["files"].keys() == theirs["files"].keys() == {"/sub.md"}
