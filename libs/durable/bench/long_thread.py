"""Long threads: the durable runtime against LangGraph's Pregel engine with its SQLite checkpointer.

Both run the same `create_agent` loop over a SQLite file: every turn is a
user message, a tool call, a ~2 KB tool result, and an answer, from a
deterministic fake model, so the numbers measure the runtime and its storage,
not a model. Reported per runtime:

- turn latency at a few points of the thread, and the total;
- resume: open the file in a fresh runtime and read the thread's state;
- the file size, write-ahead log included.

The `-delta` variants use Deep Agents' `DeepAgentState`, whose messages
`DeltaChannel` makes Pregel store writes plus a snapshot every 50 steps; the
plain variants use `create_agent`'s default `add_messages` channel, which
makes Pregel store the whole list in every checkpoint. The durable runtime
stores messages as entries either way; the schema only changes the reducer.
`--skip-plain` leaves out the plain variants, whose Pregel file grows quadratically.

    uv run --group bench python bench/long_thread.py [turns] [--skip-plain]
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents.graph import DeepAgentState
from langchain.agents import create_agent as create_pregel_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepagents_durable import Kernel, create_agent as create_durable_agent

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

RESULT = "x" * 2048
ANSWER = "y" * 512


@tool
def lookup(query: str) -> str:
    """Look something up."""
    return f"{query}: {RESULT}"


class TurnModel(BaseChatModel):
    """Calls `lookup` after a user message and answers after a tool result."""

    @property
    def _llm_type(self) -> str:
        return "turn"

    def _generate(self, messages: list[BaseMessage], **_: Any) -> ChatResult:
        if messages[-1].type == "human":
            message = AIMessage(
                content="", tool_calls=[{"name": "lookup", "args": {"query": messages[-1].text}, "id": f"call-{len(messages)}", "type": "tool_call"}]
            )
        else:
            message = AIMessage(content=ANSWER)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, *_: Any, **__: Any) -> TurnModel:
        return self


@dataclass
class Result:
    name: str
    turns: list[float]
    resume: float
    size: int
    messages: int

    def row(self, marks: list[int]) -> str:
        cells = [f"{self.turns[mark - 1] * 1000:7.2f}" for mark in marks]
        total = sum(self.turns)
        return f"{self.name:<12} {' '.join(cells)}  {total:8.2f}s  {self.resume * 1000:8.2f}ms  {self.size / 1e6:8.2f}MB  {self.messages:>6}"


def size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.parent.glob(f"{path.name}*"))


async def turns(invoke: Callable[[int], Awaitable[None]], count: int) -> list[float]:
    latencies = []
    for turn in range(count):
        start = time.perf_counter()
        await invoke(turn)
        latencies.append(time.perf_counter() - start)
    return latencies


async def durable(path: Path, count: int, *, schema: type | None = None, name: str = "durable") -> Result:
    config = {"configurable": {"thread_id": "bench"}}
    async with await Kernel.open(path) as kernel:
        agent = create_durable_agent(TurnModel(), [lookup], checkpointer=kernel, state_schema=schema)
        latencies = await turns(lambda turn: agent.ainvoke({"messages": [HumanMessage(f"question {turn}")]}, config), count)
    start = time.perf_counter()
    async with await Kernel.open(path) as kernel:
        agent = create_durable_agent(TurnModel(), [lookup], checkpointer=kernel, state_schema=schema)
        state = await agent.aget_state(config)
    resume = time.perf_counter() - start
    return Result(name, latencies, resume, size(path), len(state.values["messages"]))


async def pregel(path: Path, count: int, *, schema: type | None = None, name: str = "pregel") -> Result:
    config = {"configurable": {"thread_id": "bench"}}
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        agent = create_pregel_agent(TurnModel(), [lookup], checkpointer=saver, state_schema=schema)
        latencies = await turns(lambda turn: agent.ainvoke({"messages": [HumanMessage(f"question {turn}")]}, config), count)
    start = time.perf_counter()
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        agent = create_pregel_agent(TurnModel(), [lookup], checkpointer=saver, state_schema=schema)
        state = await agent.aget_state(config)
    resume = time.perf_counter() - start
    return Result(name, latencies, resume, size(path), len(state.values["messages"]))


def marks(count: int) -> Iterator[int]:
    yield from sorted({1, max(1, count // 4), max(1, count // 2), count})


async def main(count: int, *, plain: bool) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        results = [
            await durable(root / "durable-delta.sqlite", count, schema=DeepAgentState, name="durable-delta"),
            await pregel(root / "delta.sqlite", count, schema=DeepAgentState, name="pregel-delta"),
        ]
        if plain:
            results.append(await durable(root / "durable.sqlite", count))
            results.append(await pregel(root / "pregel.sqlite", count))
    points = list(marks(count))
    header = " ".join(f"{'turn ' + str(mark):>7}" for mark in points)
    print(f"{count} turns, {results[0].messages} messages; turn latencies in ms")
    print(f"{'runtime':<12} {header}  {'total':>9}  {'resume':>10}  {'file':>10}  {'msgs':>6}")
    for result in results:
        print(result.row(points))
    for result in results:
        tail = result.turns[-max(1, count // 10) :]
        print(f"{result.name}: median of the last 10% of turns {statistics.median(tail) * 1000:.2f}ms")


if __name__ == "__main__":
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    asyncio.run(main(int(arguments[0]) if arguments else 300, plain="--skip-plain" not in sys.argv))
