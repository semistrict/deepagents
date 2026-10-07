"""Human input asked for with `interrupt()`, answered by running the step again.

LangChain middleware and tools ask a human for input by calling
`langgraph.types.interrupt(value)`. The durable runtime runs every step of a
run (a hook, the model call, a tool call) as a unit it may repeat, so a step
that asks stops the run: the question is saved on the thread with everything
earlier steps did, and the step's own partial work is dropped. The answer
arrives with the thread's next input, and the step runs again from its
start, its `interrupt()` calls returning the recorded answers in order until
one asks something new.

`interrupt()` finds the recorded answers through the ambient config, under
the two keys it reads; `Answers` is what those keys hold.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD, CONFIG_KEY_SEND, RESUME

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.errors import GraphInterrupt
    from langgraph.types import Interrupt


class Answers:
    """One step's answers so far, handed out in the order the step asks."""

    def __init__(self, answers: list[Any]) -> None:
        """Answer the step's first `len(answers)` questions with `answers`."""
        self.resume = list(answers)
        self._asked = itertools.count()

    def interrupt_counter(self) -> int:
        """The index of the question being asked."""
        return next(self._asked)

    @staticmethod
    def get_null_resume(consume: bool = False) -> None:  # noqa: FBT001, FBT002  # `interrupt()` passes it positionally
        """No answer arrives except through `resume`."""
        del consume


def configurable(answers: Answers, send: Callable[[list[tuple[str, Any]]], None]) -> dict[str, Any]:
    """The config entries `interrupt()` and `push_message()` read.

    `send` receives state writes made mid-step; the runtime already holds
    every answer, so `interrupt()`'s record of them is dropped.
    """

    def write(writes: list[tuple[str, Any]]) -> None:
        send([(name, value) for name, value in writes if name != RESUME])

    return {CONFIG_KEY_SCRATCHPAD: answers, CONFIG_KEY_SEND: write}


def asked(raised: GraphInterrupt) -> list[Interrupt]:
    """The questions a step's `GraphInterrupt` carries."""
    return list(raised.args[0]) if raised.args else []
