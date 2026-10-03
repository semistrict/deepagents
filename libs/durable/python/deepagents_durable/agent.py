"""The public agent: `create_agent` on a durable kernel instead of a graph.

`DurableAgent` keeps the calling conventions LangChain agents have:
`invoke`/`ainvoke` return the final state, `stream`/`astream` yield the same
chunk shapes for the `values`, `updates`, `messages`, and `custom` modes,
`get_state` reads a thread back, and `update_state` changes it between runs.
Leaving an `astream` early aborts its run. A thread is a conversation in a
kernel's SQLite file: one kernel's, or with a `Threads` checkpointer, a file
of the thread's own, listed in the threads' index. An agent invoked from inside a tool call runs as a subagent: its
own conversation, owned by the tool call, streamed under the tool's namespace.
A subagent awaiting approval stops its tool call, and so its parent's run;
the parent's resume input reaches it when the tool call runs again.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import ensure_config, merge_configs
from langgraph._internal._constants import CONF
from langgraph.constants import END
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, Interrupt, StateSnapshot
from pydantic import BaseModel

from deepagents_durable import serde, thread
from deepagents_durable.events import END as END_EVENT
from deepagents_durable.kernel import Kernel
from deepagents_durable.runtime import CURRENT_TOOL, AgentRuntime, AwaitingInputError, RunContext, ToolScope
from deepagents_durable.spec import build
from deepagents_durable.threads import Threads

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import SystemMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.store.base import BaseStore

    from deepagents_durable.spec import AgentSpec

CHILDREN = "lc.children"
"""Conversation document family mapping a tool call ID to the subagent conversation it runs."""


class RunFailedError(RuntimeError):
    """A run ended without an answer."""


class _Loop:
    """A background event loop for the synchronous API."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="deepagents-durable", daemon=True).start()

    def run(self, coroutine: Any) -> Any:  # noqa: ANN401  # the coroutine's result
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()


_background: _Loop | None = None
_defaults: dict[asyncio.AbstractEventLoop, Kernel] = {}


def _sync(coroutine: Any, scope: ToolScope | None) -> Any:  # noqa: ANN401  # the coroutine's result
    """Run a coroutine for the synchronous API: on the kernel's loop inside a tool, else in the background."""
    global _background  # noqa: PLW0603  # one background loop per process
    if scope is not None:
        return asyncio.run_coroutine_threadsafe(coroutine, scope.runtime.loop).result()
    if _background is None:
        _background = _Loop()
    return _background.run(coroutine)


async def _default_kernel() -> Kernel:
    loop = asyncio.get_running_loop()
    if loop not in _defaults:
        _defaults[loop] = await Kernel.open()
    return _defaults[loop]


def runtime_for(kernel: Kernel) -> AgentRuntime:
    """The agent runtime serving a kernel, created on first use from its event loop."""
    if kernel.agents is None:
        kernel.agents = AgentRuntime(kernel)
    return kernel.agents


def _storable(config: RunnableConfig) -> RunnableConfig:
    """The ambient config without the keys the runtime sets per step."""
    configurable = {
        key: value for key, value in config.get(CONF, {}).items() if not key.startswith("__pregel_") and key not in {"checkpoint_ns", "checkpoint_id"}
    }
    metadata = {key: value for key, value in config.get("metadata", {}).items() if not key.startswith("langgraph_") and key != "checkpoint_ns"}
    return {**config, CONF: configurable, "metadata": metadata}


def _coerce(schema: type | None, context: Any) -> Any:  # noqa: ANN401  # any context
    """A run's context as its agent's `context_schema` declares it, as LangChain agents receive it."""
    if not isinstance(context, dict) or schema is None:
        return context
    if dataclasses.is_dataclass(schema):
        names = {item.name for item in dataclasses.fields(schema)}
        return schema(**{key: value for key, value in context.items() if key in names})
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_validate(context)
    return context


def _payload(input_: Any) -> dict[str, Any]:  # noqa: ANN401  # an input dict or a Command
    if isinstance(input_, Command):
        return {"command": {"update": serde.dump(input_.update) if input_.update is not None else None, "resume": serde.dump(input_.resume)}}
    return {"input": serde.dump(input_)}


@dataclass(frozen=True)
class _Target:
    """Where one invocation runs."""

    runtime: AgentRuntime
    key: str
    """The agent's registry key: a subagent's is scoped under its parent's."""
    conversation: int
    thread_id: str | None
    events: tuple[int, tuple[str, ...]]
    scope: ToolScope | None


class DurableAgent(Runnable[Any, dict[str, Any]]):
    """A LangChain agent whose loop runs as durable tasks."""

    def __init__(self, spec: AgentSpec, checkpointer: Kernel | Threads | None = None, config: RunnableConfig | None = None) -> None:
        """Bind an agent definition to where its threads live; without a checkpointer it uses the ambient or a default kernel."""
        self.spec = spec
        self.checkpointer = checkpointer
        self.config: RunnableConfig = merge_configs(spec.config, config or {})
        self.name = spec.name
        self.key = spec.name or "agent"

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> DurableAgent:  # RunnableConfig fields
        """A copy with `config` merged into the bound config; a given `recursion_limit` always applies."""
        given: dict[str, Any] = {**(config or {}), **kwargs}
        merged = merge_configs(self.config, given)  # ty: ignore[invalid-argument-type]
        if "recursion_limit" in given:
            # `merge_configs` drops a limit equal to LangChain's default.
            merged["recursion_limit"] = given["recursion_limit"]
        return DurableAgent(self.spec, self.checkpointer, merged)

    async def _target(self, config: RunnableConfig, scope: ToolScope | None) -> _Target:
        if scope is not None:
            key = f"{scope.agent}/{self.key}"
            scope.runtime.register(key, self.spec)
            conversation = await self._child(scope.runtime, scope)
            return _Target(scope.runtime, key, conversation, None, (scope.thread, scope.ns), scope)
        raw = config.get(CONF, {}).get("thread_id")
        thread_id = None if raw is None else str(raw)
        runtime = runtime_for(await self._kernel(thread_id))
        runtime.register(self.key, self.spec)
        conversation = await self._thread(runtime, thread_id)
        return _Target(runtime, self.key, conversation, thread_id, (conversation, ()), None)

    async def _kernel(self, thread_id: str | None) -> Kernel:
        if isinstance(self.checkpointer, Threads):
            if thread_id is None:
                msg = "an agent whose threads each have a file needs a `thread_id` in its config"
                raise ValueError(msg)
            return await self.checkpointer.kernel(thread_id)
        return self.checkpointer or await _default_kernel()

    async def _indexed(self, target: _Target, metadata: dict[str, Any] | None) -> None:
        """Refresh a top-level thread's row in its `Threads` index; `metadata=None` keeps the row's."""
        if not isinstance(self.checkpointer, Threads) or target.scope is not None or target.thread_id is None:
            return
        state = await target.runtime.state(target.conversation, self.spec)
        first = next((message.text for message in state.messages if message.type == "human"), None)
        await self.checkpointer.record(target.thread_id, metadata=metadata, message_count=len(state.messages), first_message=first)

    @staticmethod
    async def _thread(runtime: AgentRuntime, thread_id: str | None) -> int:
        if thread_id is not None and thread_id in runtime.threads:
            return runtime.threads[thread_id]

        async def resolve(tx: Any) -> int:  # noqa: ANN401  # a kernel transaction
            found = await thread.find(tx, thread_id) if thread_id is not None else None
            return found if found is not None else thread.create(tx, thread_id)

        conversation = await runtime.kernel.commit(resolve)
        if thread_id is not None:
            runtime.threads[thread_id] = conversation
        return conversation

    @staticmethod
    async def _child(runtime: AgentRuntime, scope: ToolScope) -> int:
        """The subagent conversation of a tool call, created on its first run."""

        async def resolve(tx: Any) -> int:  # noqa: ANN401  # a kernel transaction
            found = await tx.doc(CHILDREN, thread.scope(scope.conversation), scope.call_id)
            if found is not None:
                return found["conversationId"]
            child = tx.create_conversation(owner={"conversationId": scope.conversation, "taskId": scope.task})
            tx.put_doc(CHILDREN, thread.scope(scope.conversation), {"conversationId": child}, key=scope.call_id)
            return child

        return await runtime.kernel.commit(resolve)

    def _merged(self, config: RunnableConfig | None, scope: ToolScope | None) -> RunnableConfig:
        """The run's config: a subagent's starts from its tool call's ambient config."""
        ambient = _storable(ensure_config()) if scope is not None else {}
        return merge_configs(ambient, self.config, config or {})

    async def _submit(self, target: _Target, input_: Any, config: RunnableConfig, context: Any) -> tuple[int, int]:  # noqa: ANN401
        scope = target.scope
        if scope is not None and scope.resume is not None and (await target.runtime.state(target.conversation, self.spec)).pending is not None:
            input_ = Command(resume=scope.resume)
        inherited = scope.context.context if scope is not None else None
        run_context = RunContext(config=config, context=_coerce(self.spec.context_schema, context) if context is not None else inherited)
        return await target.runtime.submit(
            target.key, target.conversation, _payload(input_), run_context, thread_id=target.thread_id, events=target.events
        )

    async def _settled(self, target: _Target, submission: int) -> dict[str, Any]:
        record = await target.runtime.kernel.session.wait_submission(submission)
        if record["status"] == "done":
            return record["result"]
        detail = record.get("detail") or record.get("reason", "the run ended without an answer")
        if isinstance(detail, str) and detail.startswith("GraphRecursionError("):
            raise GraphRecursionError(detail)
        raise RunFailedError(detail)

    async def _interrupts(self, target: _Target) -> tuple[Interrupt, ...]:
        pending = (await target.runtime.state(target.conversation, self.spec)).pending
        if pending is None:
            return ()
        return tuple(Interrupt(value=serde.load(item["value"]), id=item["id"]) for item in pending["interrupts"])

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, *, context: Any = None, **_: Any) -> dict[str, Any]:  # noqa: A002, ANN401
        """Run until the agent answers or interrupts; returns the final state."""
        scope = CURRENT_TOOL.get()
        merged = self._merged(config, scope)
        target = await self._target(merged, scope)
        submission = (await self._submit(target, input, merged, context))[0]
        try:
            result = await self._settled(target, submission)
        finally:
            await self._indexed(target, merged.get("metadata", {}))
        state = await target.runtime.state(target.conversation, self.spec)
        output = state.output()
        if result.get("status") == "interrupted":
            if scope is not None:
                raise AwaitingInputError(state.pending["interrupts"])  # ty: ignore[not-subscriptable]  # set when a run stops
            output["__interrupt__"] = list(await self._interrupts(target))
        return output

    async def astream(  # one loop translating events
        self,
        input: Any,  # noqa: A002, ANN401
        config: RunnableConfig | None = None,
        *,
        stream_mode: str | Sequence[str] | None = None,
        subgraphs: bool | None = False,
        context: Any = None,  # noqa: ANN401
        **_: Any,
    ) -> AsyncIterator[Any]:
        """Stream a run's events in LangGraph's chunk shapes; `stream_mode` defaults to `"values"`."""
        stream_mode = stream_mode or "values"
        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)
        scope = CURRENT_TOOL.get()
        merged = self._merged(config, scope)
        target = await self._target(merged, scope)
        subscription = target.runtime.bus.subscribe(target.events[0])
        run: int | None = None
        try:
            submission, run = await self._submit(target, input, merged, context)
            base = target.events[1]
            while True:
                event = await subscription.next()
                if event.mode == END_EVENT and event.run == run:
                    run = None
                    break
                if event.mode not in modes or event.ns[: len(base)] != base:
                    continue
                ns = event.ns[len(base) :]
                if ns and not subgraphs:
                    continue
                single = isinstance(stream_mode, str)
                if subgraphs:
                    yield (ns, event.data) if single else (ns, event.mode, event.data)
                else:
                    yield event.data if single else (event.mode, event.data)
            await self._settled(target, submission)
        finally:
            subscription.close()
            if run is not None:
                # The caller stopped listening before the run ended: stop the run too.
                await asyncio.shield(target.runtime.abort(run))
            await asyncio.shield(self._indexed(target, merged.get("metadata", {})))

    async def aget_state(self, config: RunnableConfig, *, subgraphs: bool = False) -> StateSnapshot:  # noqa: ARG002  # LangGraph's signature
        """A thread's current state; empty for a thread that does not exist yet."""
        merged = merge_configs(self.config, config)
        thread_id = merged.get(CONF, {}).get("thread_id")
        if isinstance(self.checkpointer, Threads) and thread_id is not None and not self.checkpointer.exists(str(thread_id)):
            empty_config = {CONF: {"thread_id": str(thread_id), "checkpoint_ns": ""}}
            return StateSnapshot(values={}, next=(), config=empty_config, metadata=None, created_at=None, parent_config=None, tasks=(), interrupts=())  # ty: ignore[invalid-argument-type]
        target = await self._target(merged, None)
        session = target.runtime.kernel.session
        state = await target.runtime.state(target.conversation, self.spec)
        interrupts = await self._interrupts(target)
        seq = await session.seq()
        snapshot_config = {CONF: {"thread_id": target.thread_id, "checkpoint_ns": "", "checkpoint_id": str(seq)}}
        return StateSnapshot(
            values=state.values(),
            next=(state.pending["node"],) if state.pending is not None else (),
            config=snapshot_config,  # ty: ignore[invalid-argument-type]
            metadata={"source": "loop", "step": seq},
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=interrupts,
        )

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | None,
        as_node: str | None = None,
        **_: Any,  # graph-only options such as `task_id`
    ) -> RunnableConfig:
        """Change a thread between runs, merging `values` as a step's updates would.

        `as_node="__end__"` also settles the thread: what a stopped run
        awaits is dropped. Other node names are accepted and ignored, since
        there are no nodes. A run in progress is waited for first.
        """
        target = await self._target(merge_configs(self.config, config), None)
        updates = list(values.items()) if values else []
        await target.runtime.update(target.conversation, self.spec, updates, settle=as_node == END)
        await self._indexed(target, None)
        seq = await target.runtime.kernel.session.seq()
        return {CONF: {"thread_id": target.thread_id, "checkpoint_ns": "", "checkpoint_id": str(seq)}}

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> dict[str, Any]:  # noqa: A002, ANN401
        """Synchronous `ainvoke`."""
        scope = CURRENT_TOOL.get()
        return _sync(self._with_scope(scope, self.ainvoke(input, config, **kwargs)), scope)

    def stream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[Any]:  # noqa: A002, ANN401
        """Synchronous `astream`; collects the stream, then yields it."""

        async def collect() -> list[Any]:
            return [chunk async for chunk in self.astream(input, config, **kwargs)]

        yield from _sync(collect(), None)

    def get_state(self, config: RunnableConfig, *, subgraphs: bool = False) -> StateSnapshot:
        """Synchronous `aget_state`."""
        return _sync(self.aget_state(config, subgraphs=subgraphs), None)

    def update_state(self, config: RunnableConfig, values: dict[str, Any] | None, as_node: str | None = None, **kwargs: Any) -> RunnableConfig:
        """Synchronous `aupdate_state`."""
        return _sync(self.aupdate_state(config, values, as_node, **kwargs), None)

    @staticmethod
    async def _with_scope(scope: ToolScope | None, coroutine: Any) -> Any:  # noqa: ANN401  # the coroutine's result
        token = CURRENT_TOOL.set(scope)
        try:
            return await coroutine
        finally:
            CURRENT_TOOL.reset(token)


def create_agent(  # mirrors langchain's create_agent
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]] = (),
    response_format: Any = None,  # noqa: ANN401
    state_schema: type | None = None,
    context_schema: type | None = None,
    checkpointer: Kernel | Threads | None = None,
    store: BaseStore | None = None,
    name: str | None = None,
    **_: Any,  # graph-only options such as `debug` and `cache`
) -> DurableAgent:
    """Build an agent like `langchain.agents.create_agent`, running on a durable kernel.

    `checkpointer` is where the threads live: a kernel whose SQLite file
    holds them all, or a `Threads` directory giving each its own file.
    Agents built without one use the kernel of the agent that invokes them
    as a subagent, or a per-event-loop in-memory kernel.
    """
    spec = build(
        model,
        tools,
        system_prompt=system_prompt,
        middleware=middleware,
        response_format=response_format,
        state_schema=state_schema,
        context_schema=context_schema,
        store=store,
        name=name,
    )
    return DurableAgent(spec, checkpointer)
