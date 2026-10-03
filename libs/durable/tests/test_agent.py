"""The agent loop on the durable kernel, checked against LangChain's Pregel-based `create_agent`."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from langchain.agents import create_agent as create_pregel_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware, hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from deepagents_durable.agent import create_agent
from deepagents_durable.kernel import Kernel
from tests.fakes import ScriptedModel, ai, call

if TYPE_CHECKING:
    from pathlib import Path


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It is sunny in {city}."


def _shape(messages: list[BaseMessage]) -> list[tuple[str, Any, list[str]]]:
    """Messages without IDs, for comparing runtimes."""
    return [(message.type, message.content, [c["name"] for c in getattr(message, "tool_calls", [])]) for message in messages]


def _weather_script() -> list[AIMessage]:
    return [ai(call("get_weather", "c1", city="Paris"), call("get_weather", "c2", city="Rome")), ai(text="Sunny in both.")]


async def test_matches_pregel_for_a_parallel_tool_round() -> None:
    async with await Kernel.open() as kernel:
        durable = create_agent(ScriptedModel(script=_weather_script()), [get_weather], checkpointer=kernel)
        ours = await durable.ainvoke({"messages": [HumanMessage("weather?")]})
    pregel = create_pregel_agent(ScriptedModel(script=_weather_script()), [get_weather])
    theirs = await pregel.ainvoke({"messages": [HumanMessage("weather?")]})
    assert _shape(ours["messages"]) == _shape(theirs["messages"])
    assert ours["messages"][2].content == "It is sunny in Paris."


async def test_updates_stream_matches_pregel() -> None:
    async with await Kernel.open() as kernel:
        durable = create_agent(ScriptedModel(script=_weather_script()), [get_weather], checkpointer=kernel)
        ours = [list(chunk) async for chunk in durable.astream({"messages": [HumanMessage("weather?")]}, stream_mode="updates")]
    pregel = create_pregel_agent(ScriptedModel(script=_weather_script()), [get_weather])
    theirs = [list(chunk) async for chunk in pregel.astream({"messages": [HumanMessage("weather?")]}, stream_mode="updates")]
    assert ours == theirs == [["model"], ["tools"], ["tools"], ["model"]]


async def test_messages_stream_matches_pregel() -> None:
    script = [ai(text="The answer is 42.")]
    async with await Kernel.open() as kernel:
        agent = create_agent(ScriptedModel(script=script), [], checkpointer=kernel, name="oracle")
        ours = [chunk async for chunk in agent.astream({"messages": [HumanMessage("?")]}, stream_mode="messages")]
    pregel = create_pregel_agent(ScriptedModel(script=script), [], name="oracle")
    theirs = [chunk async for chunk in pregel.astream({"messages": [HumanMessage("?")]}, stream_mode="messages")]

    def tokens(chunks: list[Any]) -> list[tuple[str, str]]:
        return [(message.type, message.text) for message, _ in chunks]

    assert tokens(ours) == tokens(theirs)
    assert tokens(ours)[:4] == [("AIMessageChunk", "The "), ("AIMessageChunk", "answer "), ("AIMessageChunk", "is "), ("AIMessageChunk", "42.")]
    keys = ("langgraph_node", "langgraph_step", "lc_agent_name")
    assert [tuple(meta[key] for key in keys) for _, meta in ours] == [tuple(meta[key] for key in keys) for _, meta in theirs]


@pytest.mark.parametrize("subgraphs", [False, True])
@pytest.mark.parametrize("stream_mode", ["updates", ["updates"], ["updates", "values"]])
async def test_chunk_shapes_match_pregel(stream_mode: str | list[str], subgraphs: bool) -> None:  # noqa: FBT001
    def shapes(chunks: list[Any]) -> list[Any]:
        return [tuple(type(part).__name__ for part in chunk) if isinstance(chunk, tuple) else type(chunk).__name__ for chunk in chunks]

    async with await Kernel.open() as kernel:
        durable = create_agent(ScriptedModel(script=[ai(text="hi")]), [], checkpointer=kernel)
        ours = [chunk async for chunk in durable.astream({"messages": [HumanMessage("?")]}, stream_mode=stream_mode, subgraphs=subgraphs)]
    pregel = create_pregel_agent(ScriptedModel(script=[ai(text="hi")]), [])
    theirs = [chunk async for chunk in pregel.astream({"messages": [HumanMessage("?")]}, stream_mode=stream_mode, subgraphs=subgraphs)]
    assert shapes(ours) == shapes(theirs)


async def test_a_thread_continues_after_reopening_its_file(tmp_path: Path) -> None:
    file = tmp_path / "threads.sqlite"
    config = {"configurable": {"thread_id": "t1"}}
    async with await Kernel.open(file) as kernel:
        agent = create_agent(ScriptedModel(script=[ai(text="Hello Ada.")]), [], checkpointer=kernel)
        await agent.ainvoke({"messages": [HumanMessage("I am Ada.")]}, config)
    async with await Kernel.open(file) as kernel:
        model = ScriptedModel(script=[ai(text="You are Ada.")])
        agent = create_agent(model, [], checkpointer=kernel)
        result = await agent.ainvoke({"messages": [HumanMessage("Who am I?")]}, config)
    assert [message.content for message in result["messages"]] == ["I am Ada.", "Hello Ada.", "Who am I?", "You are Ada."]
    assert [message.content for message in model.calls[0]] == ["I am Ada.", "Hello Ada.", "Who am I?"]


async def test_a_run_resumes_after_a_crash_mid_tool(tmp_path: Path) -> None:
    file = tmp_path / "crash.sqlite"
    started = asyncio.Event()

    @tool
    def slow(query: str) -> str:
        """A tool that hangs the first time."""
        return f"done: {query}"

    @tool("slow")
    async def hanging(query: str) -> str:
        """A tool that hangs the first time."""
        started.set()
        await asyncio.Event().wait()
        return query

    script = [ai(call("slow", "c1", query="x")), ai(text="Finished.")]
    kernel = await Kernel.open(file)
    agent = create_agent(ScriptedModel(script=script), [hanging], checkpointer=kernel)
    run = asyncio.create_task(agent.ainvoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t"}}))
    await started.wait()
    await kernel.close()
    run.cancel()

    async with await Kernel.open(file) as kernel:
        model = ScriptedModel(script=script[1:])
        agent = create_agent(model, [slow], checkpointer=kernel)
        # Reading the thread registers the agent; the interrupted run then resumes on its own.
        await agent.aget_state({"configurable": {"thread_id": "t"}})
        for run in await kernel.session.tasks(kind="lc.run", live=True):
            await kernel.session.wait_task(run["id"])
        state = await agent.aget_state({"configurable": {"thread_id": "t"}})
    assert _shape(state.values["messages"]) == [("human", "go", []), ("ai", "", ["slow"]), ("tool", "done: x", []), ("ai", "Finished.", [])]
    assert len(model.calls) == 1, "the model is not called again for the step that was already committed"


class Turns(AgentMiddleware):
    """Tells the model which turn it is on, and records the last answer in a field of its own."""

    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:  # noqa: ANN401
        return {"messages": [HumanMessage(f"turn {len([m for m in state['messages'] if m.type == 'ai'])}")]}

    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:  # noqa: ANN401
        return {"messages": [HumanMessage(f"saw {state['messages'][-1].text or 'tool calls'}")]}


async def test_hooks_match_pregel() -> None:
    script = [ai(call("get_weather", "c1", city="Oslo")), ai(text="Cold.")]
    async with await Kernel.open() as kernel:
        durable = create_agent(ScriptedModel(script=script), [get_weather], middleware=[Turns()], checkpointer=kernel)
        ours = await durable.ainvoke({"messages": [HumanMessage("go")]})
    pregel = create_pregel_agent(ScriptedModel(script=script), [get_weather], middleware=[Turns()])
    theirs = await pregel.ainvoke({"messages": [HumanMessage("go")]})
    assert _shape(ours["messages"]) == _shape(theirs["messages"])
    assert [m.content for m in ours["messages"] if m.type == "human"] == ["go", "turn 0", "saw tool calls", "turn 1", "saw Cold."]


class EndEarly(AgentMiddleware):
    """Ends the run after the first model turn, before its tool calls run."""

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:  # noqa: ANN401
        return {"messages": [HumanMessage("stopping")], "jump_to": "end"}


class KeepGoing(AgentMiddleware):
    """Sends the agent back to the model once when it tries to stop, as a stop hook does."""

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:  # noqa: ANN401
        if any(m.content == "keep going" for m in state["messages"]):
            return None
        return {"messages": [HumanMessage("keep going")], "jump_to": "model"}


@pytest.mark.parametrize("middleware", [EndEarly, KeepGoing])
async def test_jumps_move_the_loop_as_in_pregel(middleware: type[AgentMiddleware]) -> None:
    script = [ai(call("get_weather", "c1", city="Oslo")), ai(text="Cold."), ai(text="Still cold.")]
    async with await Kernel.open() as kernel:
        durable = create_agent(ScriptedModel(script=script), [get_weather], middleware=[middleware()], checkpointer=kernel)
        ours = await durable.ainvoke({"messages": [HumanMessage("go")]})
    pregel = create_pregel_agent(ScriptedModel(script=script), [get_weather], middleware=[middleware()])
    theirs = await pregel.ainvoke({"messages": [HumanMessage("go")]})
    assert _shape(ours["messages"]) == _shape(theirs["messages"])


@tool
def ask(question: str) -> str:
    """Ask the user a question."""
    return f"user said: {interrupt({'question': question})}"


class Confirm(AgentMiddleware):
    """Asks the user before every model call, twice, and records the answers."""

    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:  # noqa: ANN401
        first = interrupt("first?")
        second = interrupt("second?")
        return {"messages": [HumanMessage(f"{first} then {second}")]}


async def _converse(agent: Any, answers: list[str], config: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:  # noqa: ANN401
    """Run to the end, answering each interrupt in turn; returns the questions asked and the final state."""
    result = await agent.ainvoke({"messages": [HumanMessage("go")]}, config)
    asked = []
    for answer in answers:
        (pending,) = result["__interrupt__"]
        asked.append(pending.value)
        result = await agent.ainvoke(Command(resume=answer), config)
    return asked, result


@pytest.mark.parametrize(("tools", "middleware", "answers"), [([ask], [], ["blue"]), ([], [Confirm], ["yes", "no"])], ids=["tool", "hook"])
async def test_interrupts_ask_and_resume_as_in_pregel(tools: list[Any], middleware: list[type[AgentMiddleware]], answers: list[str]) -> None:
    script = [ai(call("ask", "c1", question="Favorite color?")), ai(text="Noted.")] if tools else [ai(text="Noted.")]
    config = {"configurable": {"thread_id": "asking"}}
    async with await Kernel.open() as kernel:
        model = ScriptedModel(script=script)
        durable = create_agent(model, tools, middleware=[m() for m in middleware], checkpointer=kernel)
        ours = await _converse(durable, answers, config)
    pregel = create_pregel_agent(ScriptedModel(script=script), tools, middleware=[m() for m in middleware], checkpointer=InMemorySaver())
    theirs = await _converse(pregel, answers, config)
    assert ours[0] == theirs[0]
    assert _shape(ours[1]["messages"]) == _shape(theirs[1]["messages"])
    assert len(model.calls) == len(script), "steps before the interrupt are not repeated"


async def test_leaving_a_stream_early_aborts_its_run() -> None:
    started = asyncio.Event()

    @tool
    async def wait_forever(reason: str) -> str:
        """Never returns."""
        started.set()
        await asyncio.Event().wait()
        return reason

    config = {"configurable": {"thread_id": "left"}}
    async with await Kernel.open() as kernel:
        agent = create_agent(
            ScriptedModel(script=[ai(call("wait_forever", "c1", reason="x")), ai(text="Back.")]), [wait_forever], checkpointer=kernel
        )

        async def consume() -> None:
            async for _ in agent.astream({"messages": [HumanMessage("go")]}, config):
                pass

        consumer = asyncio.create_task(consume())
        await started.wait()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert await kernel.session.tasks(kind="lc.run", live=True) == []
        await agent.aupdate_state(config, {"messages": [HumanMessage("interrupted")]})
        result = await agent.ainvoke({"messages": [HumanMessage("again")]}, config)
    assert _shape(result["messages"]) == [
        ("human", "go", []),
        ("ai", "", ["wait_forever"]),
        ("human", "interrupted", []),
        ("human", "again", []),
        ("ai", "Back.", []),
    ]


async def test_human_in_the_loop_interrupts_and_resumes() -> None:
    script = [ai(call("get_weather", "c1", city="Paris")), ai(text="Done.")]
    hitl = HumanInTheLoopMiddleware(interrupt_on={"get_weather": True})
    config = {"configurable": {"thread_id": "hitl"}}
    async with await Kernel.open() as kernel:
        agent = create_agent(ScriptedModel(script=script), [get_weather], middleware=[hitl], checkpointer=kernel)
        first = await agent.ainvoke({"messages": [HumanMessage("weather?")]}, config)
        interrupt = first["__interrupt__"][0]
        assert interrupt.value["action_requests"][0]["name"] == "get_weather"
        result = await agent.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert _shape(result["messages"])[-2:] == [("tool", "It is sunny in Paris.", []), ("ai", "Done.", [])]
    assert "__interrupt__" not in result


async def test_hitl_resume_matches_pregel() -> None:
    script = [ai(call("get_weather", "c1", city="Paris")), ai(text="Done.")]
    config = {"configurable": {"thread_id": "same"}}
    decisions = Command(resume={"decisions": [{"type": "reject", "message": "not now"}]})
    async with await Kernel.open() as kernel:
        hitl = HumanInTheLoopMiddleware(interrupt_on={"get_weather": True})
        durable = create_agent(ScriptedModel(script=script), [get_weather], middleware=[hitl], checkpointer=kernel)
        await durable.ainvoke({"messages": [HumanMessage("weather?")]}, config)
        ours = await durable.ainvoke(decisions, config)
    hitl = HumanInTheLoopMiddleware(interrupt_on={"get_weather": True})
    pregel = create_pregel_agent(ScriptedModel(script=script), [get_weather], middleware=[hitl], checkpointer=InMemorySaver())
    await pregel.ainvoke({"messages": [HumanMessage("weather?")]}, config)
    theirs = await pregel.ainvoke(decisions, config)
    assert _shape(ours["messages"]) == _shape(theirs["messages"])


async def test_an_agent_called_from_a_tool_runs_as_a_subagent() -> None:
    researcher = create_agent(ScriptedModel(script=[ai(text="Rome is in Italy.")]), [], name="researcher")

    @tool
    async def research(question: str) -> str:
        """Ask the researcher."""
        result = await researcher.ainvoke({"messages": [HumanMessage(question)]})
        return result["messages"][-1].content

    script = [ai(call("research", "c1", question="Where is Rome?")), ai(text="It is in Italy.")]
    async with await Kernel.open() as kernel:
        agent = create_agent(ScriptedModel(script=script), [research], checkpointer=kernel)
        chunks = [chunk async for chunk in agent.astream({"messages": [HumanMessage("Rome?")]}, stream_mode="updates", subgraphs=True)]
        conversations = await kernel.session.conversations()
    namespaces = {ns for ns, _ in chunks}
    assert () in namespaces
    assert any(len(ns) == 1 and ns[0].startswith("tools:") for ns in namespaces)
    assert chunks[-1][1]["model"]["messages"][0].content == "It is in Italy."
    assert len(conversations) == 2
    assert conversations[1]["owner"]["conversationId"] == conversations[0]["id"]


async def test_a_subagents_approval_stops_and_resumes_its_parent() -> None:
    @tool
    def delete_file(path: str) -> str:
        """Delete a file."""
        return f"deleted {path}"

    hitl = HumanInTheLoopMiddleware(interrupt_on={"delete_file": True})
    cleaner = create_agent(
        ScriptedModel(script=[ai(call("delete_file", "d1", path="/work/x")), ai(text="Cleaned.")]), [delete_file], middleware=[hitl], name="cleaner"
    )

    @tool
    async def clean(path: str) -> str:
        """Ask the cleaner to remove a path."""
        result = await cleaner.ainvoke({"messages": [HumanMessage(f"remove {path}")]})
        return result["messages"][-1].content

    script = [ai(call("clean", "c1", path="/work/x")), ai(text="All clean.")]
    config = {"configurable": {"thread_id": "parent"}}
    async with await Kernel.open() as kernel:
        agent = create_agent(ScriptedModel(script=script), [clean], checkpointer=kernel)
        first = await agent.ainvoke({"messages": [HumanMessage("clean up")]}, config)
        (interrupt,) = first["__interrupt__"]
        assert interrupt.value["action_requests"][0]["args"] == {"path": "/work/x"}
        result = await agent.ainvoke(Command(resume={interrupt.id: {"decisions": [{"type": "approve"}]}}), config)
    assert _shape(result["messages"])[-2:] == [("tool", "Cleaned.", []), ("ai", "All clean.", [])]
