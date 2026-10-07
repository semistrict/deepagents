"""Where a cold resume of a long thread spends its time.

uv run --group bench python bench/resume.py [turns]
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

from deepagents.graph import DeepAgentState
from langchain_core.messages import HumanMessage
from long_thread import TurnModel, lookup

from deepagents_durable import Kernel, create_agent, serde, thread
from deepagents_durable.schema import Schema


async def main(turns: int) -> None:
    config = {"configurable": {"thread_id": "resume"}}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "resume.sqlite"
        async with await Kernel.open(path) as kernel:
            agent = create_agent(TurnModel(), [lookup], checkpointer=kernel, state_schema=DeepAgentState)
            for turn in range(turns):
                await agent.ainvoke({"messages": [HumanMessage(f"q{turn}")]}, config)
        async with await Kernel.open(path) as kernel:
            start = time.perf_counter()
            entries = await kernel.session.context(2)
            fetched = time.perf_counter()
            messages = [serde.load_message(stored["entry"]["data"]) for stored in entries if stored["entry"]["kind"] == thread.MESSAGE]
            decoded = time.perf_counter()
            state = thread.ThreadState(Schema([DeepAgentState]))
            state.fold(entries)
            folded = time.perf_counter()
    print(f"{len(entries)} entries, {len(messages)} messages")
    print(f"fetch entries (Rust + conversion): {(fetched - start) * 1000:7.2f} ms")
    print(f"decode messages (pydantic):        {(decoded - fetched) * 1000:7.2f} ms")
    print(f"fold (decode + index):             {(folded - decoded) * 1000:7.2f} ms")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000))
