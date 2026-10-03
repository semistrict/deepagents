"""What code running inside one step of a run can reach.

Hooks, model calls, and tools run with a `Scope`: the thread's state, the
`Runtime` LangChain middleware expects, the updates sent mid-step (a backend
writing a file before the tool returns), and the answers to the step's
earlier `interrupt()` calls. The scope is also exposed as the ambient
runnable config, so `get_runtime()`, `get_store()`, `get_stream_writer()`,
and `interrupt()` keep working inside middleware and tools.
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import AsyncCallbackManager
from langchain_core.runnables.config import merge_configs, set_config_context
from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME

from deepagents_durable import answers as answering

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

    from deepagents_durable.thread import ThreadState

CURRENT: contextvars.ContextVar[Scope | None] = contextvars.ContextVar("deepagents_durable_scope", default=None)


@dataclass
class Scope:
    """One step's view of its thread, and the updates it sends before returning."""

    state: ThreadState
    runtime: Runtime
    node: str
    step: int
    ns: str
    thread_id: str | None
    agent_name: str | None
    callbacks: list[Any] = field(default_factory=list)
    answers: list[Any] = field(default_factory=list)
    """Answers to the step's `interrupt()` calls, from earlier runs of it."""
    sent: list[tuple[str, Any]] = field(default_factory=list)

    def send(self, updates: list[tuple[str, Any]]) -> None:
        """Queue updates the step applies with its result."""
        self.sent.extend(updates)

    def read(self, name: str) -> Any:  # noqa: ANN401  # any field value
        """A field as this step sees it: the thread's value plus the step's own sent updates."""
        own = [value for sent, value in self.sent if sent == name]
        current = self.state.fields.get(name)
        return self.state.schema.merge(name, current, own) if own else current

    def config(self, base: RunnableConfig) -> RunnableConfig:
        """The ambient config for this step: the run's config plus where the step is."""
        configurable: dict[str, Any] = {
            CONFIG_KEY_RUNTIME: self.runtime,
            "checkpoint_ns": self.ns,
            **answering.configurable(answering.Answers(self.answers), self.send),
        }
        metadata: dict[str, Any] = {
            "langgraph_step": self.step,
            "langgraph_node": self.node,
            "langgraph_checkpoint_ns": self.ns,
            "checkpoint_ns": self.ns,
        }
        if self.thread_id is not None:
            configurable["thread_id"] = metadata["thread_id"] = self.thread_id
        if self.agent_name is not None:
            metadata["lc_agent_name"] = self.agent_name
        callbacks = AsyncCallbackManager(handlers=list(self.callbacks), inheritable_handlers=list(self.callbacks))
        return merge_configs(base, {CONF: configurable, "metadata": metadata, "callbacks": callbacks})

    async def run(self, base: RunnableConfig, call: Callable[[RunnableConfig], Coroutine[Any, Any, Any]]) -> Any:  # noqa: ANN401  # the call's result
        """Run `call(config)` with this scope and its config as the ambient context."""
        config = self.config(base)
        token = CURRENT.set(self)
        try:
            with set_config_context(config) as context:
                return await asyncio.create_task(call(config), context=context)
        finally:
            CURRENT.reset(token)
