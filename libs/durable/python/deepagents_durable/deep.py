"""Deep Agents on the durable runtime: `create_deep_agent`, and a factory for apps that build their own."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import deepagents

from deepagents_durable.agent import DurableAgent, create_agent
from deepagents_durable.backends import DurableStateBackend

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.store.base import BaseStore

    from deepagents_durable.kernel import Kernel
    from deepagents_durable.threads import Threads


def agent_factory(checkpointer: Kernel | Threads | None = None, *, store: BaseStore | None = None) -> Callable[..., DurableAgent]:
    """A `create_agent` replacement for `agent_factory=` parameters, building agents whose threads live in `checkpointer`.

    Apps that assemble agents through `deepagents.create_deep_agent` pass
    this as its `agent_factory`. The LangGraph `checkpointer` those callers
    pass along is ignored. `store` replaces any store they pass.
    """

    def create(model: Any, **kwargs: Any) -> DurableAgent:  # noqa: ANN401  # create_agent's arguments
        kwargs.pop("checkpointer", None)
        if store is not None:
            kwargs["store"] = store
        return create_agent(model, checkpointer=checkpointer, **kwargs)

    return create


def create_deep_agent(
    *args: Any, checkpointer: Kernel | Threads | None = None, **kwargs: Any
) -> DurableAgent:  # forwards create_deep_agent's arguments
    """Build a Deep Agent whose loop, and every declarative subagent's, runs on a durable kernel.

    Takes `deepagents.create_deep_agent`'s arguments. `checkpointer` is where
    the threads live, a kernel or a `Threads` directory. Without a `backend`,
    files live in the thread, as with `StateBackend`.
    """
    kwargs.setdefault("backend", DurableStateBackend())
    # The SDK's return type names the graph its default factory compiles.
    return deepagents.create_deep_agent(*args, agent_factory=agent_factory(checkpointer), **kwargs)  # ty: ignore[invalid-return-type]
