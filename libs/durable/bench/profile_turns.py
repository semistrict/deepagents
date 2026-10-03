"""Profile the durable runtime's late turns on a long thread.

uv run --group bench python bench/profile_turns.py [turns] [profiled]
"""

from __future__ import annotations

import asyncio
import cProfile
import pstats
import sys
import tempfile
import time
from pathlib import Path

from deepagents.graph import DeepAgentState
from langchain_core.messages import HumanMessage
from long_thread import TurnModel, lookup

from deepagents_durable import Kernel, create_agent


async def main(turns: int, profiled: int) -> None:
    config = {"configurable": {"thread_id": "profile"}}
    with tempfile.TemporaryDirectory() as directory:
        async with await Kernel.open(Path(directory) / "profile.sqlite") as kernel:
            agent = create_agent(TurnModel(), [lookup], checkpointer=kernel, state_schema=DeepAgentState)
            for turn in range(turns - profiled):
                await agent.ainvoke({"messages": [HumanMessage(f"q{turn}")]}, config)
            profile = cProfile.Profile()
            start = time.perf_counter()
            profile.enable()
            for turn in range(profiled):
                await agent.ainvoke({"messages": [HumanMessage(f"q{turn}")]}, config)
            profile.disable()
            elapsed = time.perf_counter() - start
    print(f"{profiled} turns after {turns - profiled}: {elapsed / profiled * 1000:.2f} ms per turn")
    pstats.Stats(profile).sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    arguments = [int(argument) for argument in sys.argv[1:]]
    asyncio.run(main(*(arguments or [800, 100])))
