"""The app's client for an agent running in its own process on the durable runtime.

!!! warning "Experimental"

    See `client.launch.durable`.

The app drives it as a local agent: `astream`, `aget_state`, and
`aupdate_state` go straight to the `DurableAgent`. What the app otherwise
asks of the server, here the process answers itself: the approval-mode
Store record each turn writes goes to the store the agent's middleware
reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from deepagents_durable import DurableAgent
    from langchain_core.runnables import RunnableConfig
    from langgraph.store.base import BaseStore


class DurableClient:
    """A `DurableAgent` and its store, behind the agent surface the app uses."""

    def __init__(self, agent: DurableAgent, store: BaseStore) -> None:
        """Serve `agent`, whose middleware reads `store`."""
        self.agent = agent
        self.store = store

    def astream(
        self,
        input: Any,  # noqa: A002, ANN401  # an input dict or a Command
        config: RunnableConfig | None = None,
        **kwargs: Any,  # stream options
    ) -> AsyncIterator[Any]:
        """Stream a run; leaving early aborts it.

        Returns:
            The run's chunks, shaped as the requested stream modes shape them.
        """
        return self.agent.astream(input, config, **kwargs)

    async def ainvoke(
        self,
        input: Any,  # noqa: A002, ANN401  # an input dict or a Command
        config: RunnableConfig | None = None,
        **kwargs: Any,  # invoke options
    ) -> dict[str, Any]:
        """Run until the agent answers or interrupts.

        Returns:
            The thread's final state.
        """
        return await self.agent.ainvoke(input, config, **kwargs)

    async def aget_state(self, config: RunnableConfig) -> Any:  # noqa: ANN401  # a StateSnapshot
        """Read a thread's current state.

        Returns:
            The thread's `StateSnapshot`.
        """
        return await self.agent.aget_state(config)

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | None,
        *,
        as_node: str | None = None,
    ) -> RunnableConfig:
        """Change a thread between runs.

        Returns:
            The config naming the thread's new state.
        """
        return await self.agent.aupdate_state(config, values, as_node)

    async def aput_store_item(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> None:
        """Write a store item the agent's middleware reads."""
        await self.store.aput(namespace, key, value, index=False)

    def with_config(self, config: Mapping[str, Any]) -> DurableClient:
        """Return a client whose agent binds `config`."""
        return DurableClient(self.agent.with_config(dict(config)), self.store)  # ty: ignore[invalid-argument-type]
