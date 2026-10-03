"""Latency of one Python-to-kernel round trip, a read, and a commit.

uv run --group bench python bench/roundtrip.py
"""

from __future__ import annotations

import asyncio
import time

from deepagents_durable import Kernel


async def timed(label: str, count: int, call) -> None:  # noqa: ANN001
    start = time.perf_counter()
    for _ in range(count):
        await call()
    print(f"{label:<20} {(time.perf_counter() - start) / count * 1e6:8.1f} us")


async def main() -> None:
    async with await Kernel.open() as kernel:
        root = await kernel.commit(lambda tx: tx.create_root())
        await timed("seq()", 5000, kernel.session.seq)
        await timed("conversation()", 5000, lambda: kernel.session.conversation(root))

        async def commit() -> None:
            tx = kernel.session.begin()
            tx.append_entry(root, "x", {"data": 1})
            await tx.commit()

        await timed("begin + commit", 5000, commit)


if __name__ == "__main__":
    asyncio.run(main())
