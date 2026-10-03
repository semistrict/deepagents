"""The agent loop as durable tasks.

A run is one `lc.run` task per submitted input, carrying that input. The loop
is three sequences of steps:

- `agent`: the `before_agent` hooks, once per run;
- `turn`: the `before_model` hooks, the model call through every
  `wrap_model_call` middleware, and the `after_model` hooks;
- `finish`: the `after_agent` hooks.

The run's phases drive them, each ending in one atomic commit of its state
changes and the next phase. `start` places the input and runs `agent` and the
first `turn`; when the model calls tools, the same commit creates one
`lc.tool` child task per call and the run waits for them. `collect` applies
the tool results in call order and runs the next `turn`; `finish` runs its
sequence. A hook's `jump_to` moves the loop: `"model"` to a turn, `"tools"` to
the pending tool calls, `"end"` to `finish`.

The input and tool results are durable before the phase that uses them, so a
crash repeats at most the phase in flight. A step that calls `interrupt()`
stops the run instead, keeping what the steps before it did; the next input
carries the answer, and the run continues by running that step again (see
`answers`). A tool call that asks, or whose subagent asks, stops its round
the same way.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.factory import _build_commands, _fetch_last_ai_and_tool_messages
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest
from langchain_core.runnables.config import merge_configs
from langgraph.errors import GraphInterrupt, GraphRecursionError
from langgraph.runtime import ExecutionInfo, Runtime
from langgraph.types import Command, Interrupt

from deepagents_durable import serde, thread
from deepagents_durable.answers import asked
from deepagents_durable.events import END as END_EVENT, Event, EventBus
from deepagents_durable.scope import Scope
from deepagents_durable.spec import execute_model
from deepagents_durable.tasks import Invocation, TaskDefinition
from deepagents_durable.thread import PENDING, Change, ThreadState
from deepagents_durable.tokens import TokenTap

if TYPE_CHECKING:
    from langchain_core.messages import AnyMessage, BaseMessage, ToolCall
    from langchain_core.runnables import RunnableConfig

    from deepagents_durable import _core
    from deepagents_durable.kernel import Kernel
    from deepagents_durable.spec import AgentSpec

RUN = "lc.run"
TOOL = "lc.tool"
WARM_THREADS = 64
"""Threads whose state stays in memory between runs."""
DONE = "done"
"""Not a phase: the run settles in the same commit."""

Updates = list[tuple[str, Any]]


class AwaitingInputError(Exception):
    """A subagent stopped for human input; the tool call that runs it stops with it."""

    def __init__(self, interrupts: list[dict[str, Any]]) -> None:
        """Carry the subagent's stored interrupts."""
        super().__init__("a subagent is awaiting input")
        self.interrupts = interrupts


class ThreadBusyError(RuntimeError):
    """A thread already has a run in progress."""


@dataclass
class RunContext:
    """What a run uses that cannot be stored: the caller's config objects and context."""

    config: RunnableConfig = field(default_factory=dict)
    context: Any = None


@dataclass
class ToolScope:
    """Set while a tool call runs, so an agent the tool invokes runs as its subagent."""

    runtime: AgentRuntime
    task: int
    conversation: int
    call_id: str
    thread: int
    ns: tuple[str, ...]
    resume: Any
    context: RunContext
    agent: str
    """The registry key of the agent whose tool this is."""


CURRENT_TOOL: contextvars.ContextVar[ToolScope | None] = contextvars.ContextVar("deepagents_durable_tool", default=None)


@dataclass
class _Live:
    """A run's state while this process drives it."""

    state: ThreadState
    key: tuple[int, int]
    seen: set[str] = field(default_factory=set)
    """IDs of messages already streamed in messages mode."""


@dataclass
class _Step:
    """One phase's work: its change, and the node updates to stream once it commits."""

    change: Change = field(default_factory=Change)
    nodes: list[tuple[str, Updates]] = field(default_factory=list)

    def record(self, state: ThreadState, node: str, updates: Updates) -> None:
        """Apply a node's updates and remember them for streaming."""
        self.change.extend(state.apply(updates))
        self.nodes.append((node, updates))


@dataclass(frozen=True)
class _Stopped:
    """A step asked for input: the questions, and where the run continues once answered."""

    interrupts: list[Interrupt]
    sequence: str
    at: int
    node: str
    answers: list[Any]


def _merge(updates: Updates) -> dict[str, Any]:
    """Updates as the dict stream consumers see."""
    merged: dict[str, Any] = {}
    for name, value in updates:
        if name in merged and isinstance(merged[name], list) and isinstance(value, list):
            merged[name] = [*merged[name], *value]
        else:
            merged[name] = value
    return merged


def _updates(result: Any) -> Updates:  # noqa: ANN401  # any hook, model, or tool result
    """A model's or tool's result as state updates."""
    if result is None:
        return []
    if isinstance(result, list):
        return [update for item in result for update in _updates(item)]
    if isinstance(result, Command):
        if result.goto or result.graph is not None or result.resume is not None:
            msg = f"Command(goto/graph/resume) is a graph instruction the durable runtime does not follow: {result!r}"
            raise NotImplementedError(msg)
        update = result.update or {}
        return list(update.items()) if isinstance(update, dict) else list(update)
    if isinstance(result, dict):
        return list(result.items())
    msg = f"expected a dict, a Command, or None; got {type(result).__name__}"
    raise TypeError(msg)


def _hook_result(result: Any) -> tuple[Updates, str | None]:  # noqa: ANN401  # any hook result
    """A hook's result as state updates, and where it moves the loop."""
    if isinstance(result, dict) and "jump_to" in result:
        updates = dict(result)
        return _updates(updates), updates.pop("jump_to")
    return _updates(result), None


def _langchain(messages: list[BaseMessage]) -> list[AnyMessage]:
    """A thread's messages as LangChain's helpers type them: each is one of its concrete message classes."""
    return cast("list[AnyMessage]", messages)


def _pending_calls(spec: AgentSpec, messages: list[BaseMessage]) -> list[ToolCall]:
    """Tool calls of the last AI message that have no result yet."""
    last, results = _fetch_last_ai_and_tool_messages(_langchain(messages))
    if last is None:
        return []
    answered = {message.tool_call_id for message in results}
    return [call for call in last.tool_calls if call["id"] not in answered and call["name"] not in spec.output_tools]


def _after_turn(spec: AgentSpec, values: dict[str, Any]) -> str:
    """The phase after a model turn: tools, another turn, or the end."""
    last, _ = _fetch_last_ai_and_tool_messages(values["messages"])
    if last is None or not last.tool_calls:
        return "finish"
    if _pending_calls(spec, values["messages"]):
        return "tools"
    return "finish" if "structured_response" in values else "turn"


def _after_tools(spec: AgentSpec, values: dict[str, Any]) -> str:
    """The phase after a tool round: another turn, or the end for direct returns and structured output."""
    last, results = _fetch_last_ai_and_tool_messages(values["messages"])
    if last is None:
        return "turn"
    client = [tool for call in last.tool_calls if (tool := spec.tool(call["name"])) is not None]
    if client and all(tool.return_direct for tool in client):
        return "finish"
    return "finish" if any(message.name in spec.output_tools for message in results) else "turn"


def _route(spec: AgentSpec, sequence: str, jump: str | None, values: dict[str, Any]) -> str:
    """Where the loop goes after a sequence, or after a hook's jump out of it."""
    if jump == "end":
        following = DONE if sequence == "finish" else "finish"
    elif jump == "model":
        following = "turn"
    elif jump == "tools":
        following = "tools" if _pending_calls(spec, values["messages"]) else _after_turn(spec, values)
    elif jump is not None:
        msg = f"jump_to must be 'model', 'tools', or 'end'; got {jump!r}"
        raise ValueError(msg)
    elif sequence == "agent":
        following = "turn"
    elif sequence == "turn":
        following = _after_turn(spec, values)
    else:
        following = DONE
    return DONE if following == "finish" and not spec.after_agent else following


def _stages(spec: AgentSpec, sequence: str) -> list[tuple[str, AgentMiddleware | None]]:
    """A sequence's steps, as `(hook, middleware)`; the model call is `("model", None)`."""
    if sequence == "agent":
        return [("before_agent", item) for item in spec.before_agent]
    if sequence == "turn":
        return [
            *(("before_model", item) for item in spec.before_model),
            ("model", None),
            *(("after_model", item) for item in reversed(spec.after_model)),
        ]
    return [("after_agent", item) for item in reversed(spec.after_agent)]


def _hook(middleware: AgentMiddleware, name: str) -> tuple[Any, bool]:
    """A middleware's hook, async when it overrides the async variant."""
    if getattr(type(middleware), f"a{name}") is not getattr(AgentMiddleware, f"a{name}"):
        return getattr(middleware, f"a{name}"), True
    return getattr(middleware, name), False


def _resume_value(resume: Any, interrupt_id: str) -> Any:  # noqa: ANN401  # any resume payload
    """The part of a resume payload meant for one interrupt: keyed by its ID, or the whole payload."""
    if isinstance(resume, dict) and interrupt_id in resume:
        return resume[interrupt_id]
    return resume


def _stored(interrupt: Interrupt) -> dict[str, Any]:
    return {"id": interrupt.id, "value": serde.dump(interrupt.value)}


def _loaded(stored: list[dict[str, Any]]) -> tuple[Interrupt, ...]:
    return tuple(Interrupt(value=serde.load(item["value"]), id=item["id"]) for item in stored)


class AgentRuntime:
    """Runs agents' loops as durable tasks on one kernel.

    The kernel's file has one owner, this process, so the runtime keeps what
    it knows in memory: warm thread states, which threads are busy, and which
    conversation each thread ID maps to.
    """

    def __init__(self, kernel: Kernel) -> None:
        """Serve runs and tool calls of every agent registered on `kernel`."""
        self.kernel = kernel
        self.loop = asyncio.get_running_loop()
        self.bus = EventBus()
        self._agents: dict[str, AgentSpec] = {}
        self._ready: dict[str, asyncio.Event] = {}
        self._contexts: dict[int, RunContext] = {}
        self._live: dict[int, _Live] = {}
        self._warm: OrderedDict[tuple[int, int], asyncio.Future[ThreadState]] = OrderedDict()
        """Each warm thread's state, as a future so concurrent first reads share one load."""
        self._busy: dict[int, int] = {}
        """The run in progress on each busy conversation."""
        self._recovered = False
        self.threads: dict[str, int] = {}
        """The conversation of each thread ID seen so far; the mapping never changes."""
        phases = {"start": self._start, "turn": self._turn, "collect": self._collect, "finish": self._finish}
        kernel.register(TaskDefinition(kind=RUN, phases=phases, abort=self._abort_run, on_fault=self._run_faulted))
        kernel.register(TaskDefinition(kind=TOOL, phases={"call": self._tool_call}, abort=self._abort_tool))

    def register(self, key: str, spec: AgentSpec) -> None:
        """Make an agent runnable under `key`; runs waiting for it proceed."""
        self._agents[key] = spec
        self._ready.setdefault(key, asyncio.Event()).set()

    async def agent(self, key: str) -> AgentSpec:
        """The agent registered under `key`, waiting until one is."""
        await self._ready.setdefault(key, asyncio.Event()).wait()
        return self._agents[key]

    async def _recover_busy(self) -> None:
        """Learn, once, which conversations have runs left over from an earlier process."""
        if not self._recovered:
            self._recovered = True
            for task in await self.kernel.session.tasks(kind=RUN, live=True):
                self._busy.setdefault(task["conversationId"], task["id"])

    async def _active(self, conversation: int) -> int | None:
        """The run in progress on a conversation, if any."""
        await self._recover_busy()
        run = self._busy.get(conversation)
        if run is None:
            return None
        # The run may have just settled and not yet cleaned up after itself.
        busy = await self.kernel.session.task(run)
        return run if busy is not None and busy["state"]["status"] != "terminal" else None

    async def submit(
        self,
        key: str,
        conversation: int,
        payload: dict[str, Any],
        context: RunContext,
        *,
        thread_id: str | None,
        events: tuple[int, tuple[str, ...]] | None = None,
    ) -> tuple[int, int]:
        """Admit a run on an idle conversation; returns `(submission, run task)`.

        Raises:
            ThreadBusyError: The conversation already has a run in progress.
        """
        busy = await self._active(conversation)
        if busy is not None:
            msg = f"thread {thread_id or conversation} is busy with run {busy}"
            raise ThreadBusyError(msg)
        root, ns = events or (conversation, ())
        task_input = {"agent": key, "threadId": thread_id, "thread": root, "ns": list(ns), "payload": payload}

        def admit(tx: _core.Tx) -> tuple[int, int]:
            submission = tx.create_submission(conversation, {"type": "input"})
            run = tx.create_task(conversation, RUN, {**task_input, "submission": submission}, {"phase": "start"})
            return submission, run

        submission, run = await self.kernel.commit(admit)
        self._busy[conversation] = run
        self._contexts[run] = context
        return submission, run

    async def abort(self, run: int) -> None:
        """Abort a run and its tool calls, and wait until it has settled."""

        def mark(tx: _core.Tx) -> None:
            tx.abort_task(run)

        task = await self.kernel.session.task(run)
        if task is not None and task["state"]["status"] != "terminal":
            await self.kernel.commit(mark)
            await self.kernel.session.wait_task(run)

    async def update(self, conversation: int, spec: AgentSpec, updates: Updates, *, settle: bool = False) -> None:
        """Change a thread between runs, as if a step had made `updates`.

        `settle` also drops what a stopped run awaits, so the thread starts
        fresh at its next input. A run in progress is waited for first.
        """
        busy = await self._active(conversation)
        if busy is not None:
            await self.kernel.session.wait_task(busy)
        state = await self.state(conversation, spec)
        change = state.apply(updates)
        retire = settle and state.pending is not None
        if retire:
            state.pending = None

        def write(tx: _core.Tx) -> None:
            state.persist(tx, conversation, change)
            if retire:
                tx.retire_doc(PENDING, thread.scope(conversation))

        try:
            await self.kernel.commit(write)
        except BaseException:
            self._warm.pop((conversation, id(spec.schema)), None)
            raise

    async def state(self, conversation: int, spec: AgentSpec) -> ThreadState:
        """A thread's current state, kept warm between runs since only this process writes it."""
        key = (conversation, id(spec.schema))
        loading = self._warm.get(key)
        if loading is None:
            loading = asyncio.ensure_future(thread.load(self.kernel.session, conversation, spec.schema))
            self._warm[key] = loading
            while len(self._warm) > WARM_THREADS:
                self._warm.popitem(last=False)
        self._warm.move_to_end(key)
        try:
            return await asyncio.shield(loading)
        except BaseException:
            if self._warm.get(key) is loading and loading.done():
                del self._warm[key]
            raise

    # Shared plumbing.

    def _context(self, run: int) -> RunContext:
        return self._contexts.get(run) or RunContext()

    def _config(self, run: int, spec: AgentSpec) -> RunnableConfig:
        return merge_configs(spec.config, self._context(run).config)

    async def _live_state(self, invocation: Invocation, spec: AgentSpec) -> _Live:
        run = invocation.task.get("owner") or invocation.id
        live = self._live.get(run)
        if live is None:
            live = _Live(await self.state(invocation.conversation, spec), (invocation.conversation, id(spec.schema)))
            self._live[run] = live
        return live

    def _forget(self, run: int) -> None:
        """Drop a run's state, and the thread's warm copy, which it may have changed without committing."""
        live = self._live.pop(run, None)
        if live is not None:
            self._warm.pop(live.key, None)

    def _scope(self, invocation: Invocation, spec: AgentSpec, live: _Live, node: str, task_id: str, answers: list[Any] | None = None) -> Scope:
        run = invocation.task.get("owner") or invocation.id
        ns = "|".join([*invocation.input["ns"], f"{node}:{task_id}"])
        info = ExecutionInfo(checkpoint_id=str(invocation.id), checkpoint_ns=ns, task_id=task_id, thread_id=invocation.input["threadId"])
        runtime = Runtime(
            context=self._context(run).context, store=spec.store, stream_writer=partial(self._publish, invocation, "custom"), execution_info=info
        )

        def token(chunk: Any, metadata: dict[str, Any]) -> None:  # noqa: ANN401  # a message chunk
            if chunk.id is not None:
                live.seen.add(chunk.id)
            self._publish(invocation, "messages", (chunk, metadata))

        step = self._step_number(invocation)
        return Scope(
            live.state, runtime, node, step, ns, invocation.input["threadId"], spec.name, callbacks=[TokenTap(token)], answers=list(answers or [])
        )

    @staticmethod
    def _step_number(invocation: Invocation) -> int:
        """The step a phase computes: one past its checkpoint's; a tool call's is its round's."""
        if invocation.task["kind"] == TOOL:
            return invocation.input["step"]
        state = invocation.task["state"]
        return (state["checkpoint"].get("step", 0) if "checkpoint" in state else 0) + 1

    def _publish(self, invocation: Invocation, mode: str, data: Any) -> None:  # noqa: ANN401  # any stream payload
        event = Event(thread=invocation.input["thread"], ns=tuple(invocation.input["ns"]), mode=mode, data=data, run=invocation.id)
        self.bus.publish(event)

    def _published(self, invocation: Invocation, live: _Live, step: _Step, metadata: dict[str, Any]) -> None:
        """Stream a committed phase: its node updates, its new messages, and the values."""
        for node, updates in step.nodes:
            if node != "tools":  # each tool call streamed its own update when it finished
                self._publish(invocation, "updates", {node: _merge(updates)})
        for operation, message in step.change.messages:
            if operation in {"add", "replace"} and message.id not in live.seen:
                live.seen.add(message.id)
                self._publish(invocation, "messages", (message, metadata))
        self._publish(invocation, "values", live.state.output())

    def _metadata(self, invocation: Invocation, spec: AgentSpec, live: _Live) -> dict[str, Any]:
        scope = self._scope(invocation, spec, live, "model", str(invocation.id))
        return dict(scope.config(self._config(invocation.id, spec)).get("metadata", {}))

    def _ended(self, invocation: Invocation) -> None:
        self._busy.pop(invocation.conversation, None)
        self._live.pop(invocation.id, None)
        self._contexts.pop(invocation.id, None)
        self._publish(invocation, END_EVENT, None)

    @staticmethod
    def _settled(invocation: Invocation, **settlement: Any) -> dict[str, Any]:  # settlement fields
        """The run's submission record, settled."""
        return {"id": invocation.input["submission"], "conversationId": invocation.conversation, "type": "input", **settlement}

    def _spawn(
        self, invocation: Invocation, spec: AgentSpec, live: _Live, stage: _core.Step, *, done: dict[str, Any], answers: dict[str, Any] | None = None
    ) -> None:
        """Create a task for each pending tool call and wait for them, in the step's commit.

        `answers` carries, per call ID, what a call that asked for input before
        runs again with: its recorded `answers`, or its subagent's `resume`.
        """
        calls = _pending_calls(spec, live.state.messages)
        number = invocation.checkpoint.get("step", 0) + 1
        shared = {key: invocation.input[key] for key in ("agent", "threadId", "thread", "ns", "submission")} | {"step": number}
        tasks = [
            stage.tx.create_task(
                invocation.conversation, TOOL, {**shared, "call": call, **(answers or {}).get(call["id"], {})}, {"phase": "call"}, owner=invocation.id
            )
            for call in calls
            if call["id"] not in done
        ]
        following = {"phase": "collect", "step": number, "calls": [call["id"] for call in calls], "tasks": tasks, "done": done}
        if tasks:
            stage.wait(tasks, following)
        else:
            stage.advance(following)

    async def _commit(
        self, invocation: Invocation, spec: AgentSpec, live: _Live, step: _Step, following: str, *, retire_pending: bool = False
    ) -> None:
        """Store a phase's change and move the run to `following`: tools, a phase, or `DONE`.

        Memory changes first, as callers may read it the moment the commit lands;
        a failed commit drops the thread's warm state.
        """
        if retire_pending:
            live.state.pending = None
        try:
            async with invocation.step() as stage:
                live.state.persist(stage.tx, invocation.conversation, step.change, by_task=invocation.id)
                if retire_pending:
                    stage.tx.retire_doc(PENDING, thread.scope(invocation.conversation))
                if following == DONE:
                    stage.tx.put_submission(self._settled(invocation, status="done", result={"status": "success"}))
                    stage.finish({"status": "success"})
                elif following == "tools":
                    self._spawn(invocation, spec, live, stage, done={})
                else:
                    stage.advance({"phase": following, "step": invocation.checkpoint.get("step", 0) + 1})
        except BaseException:
            self._forget(invocation.id)
            raise
        self._published(invocation, live, step, self._metadata(invocation, spec, live))
        if following == DONE:
            self._ended(invocation)

    async def _stop(self, invocation: Invocation, live: _Live, step: _Step, pending: dict[str, Any]) -> None:
        """Stop the run awaiting input, saving what it waits for and where it continues."""
        live.state.pending = pending
        try:
            async with invocation.step() as stage:
                live.state.persist(stage.tx, invocation.conversation, step.change, by_task=invocation.id)
                stage.tx.put_doc(PENDING, thread.scope(invocation.conversation), pending)
                stage.tx.put_submission(self._settled(invocation, status="done", result={"status": "interrupted"}))
                stage.finish({"status": "interrupted"})
        except BaseException:
            self._forget(invocation.id)
            raise
        self._published(invocation, live, step, {})
        interrupts = _loaded(pending["interrupts"])
        self._publish(invocation, "updates", {"__interrupt__": interrupts})
        self._publish(invocation, "values", {"__interrupt__": interrupts})
        self._ended(invocation)

    # Steps and sequences.

    async def _hook_step(
        self, invocation: Invocation, spec: AgentSpec, live: _Live, step: _Step, hook: str, middleware: AgentMiddleware, answers: list[Any]
    ) -> str | None:
        """Run one middleware hook, record its updates, and return where it jumps."""
        function, is_async = _hook(middleware, hook)
        wants_config = "config" in inspect.signature(function).parameters
        node = f"{middleware.name}.{hook}"
        scope = self._scope(invocation, spec, live, node, str(uuid.uuid4()), answers)

        async def call(config: RunnableConfig) -> Any:  # noqa: ANN401  # any hook result
            result = function(live.state.values(), scope.runtime, **({"config": config} if wants_config else {}))
            return await result if is_async else result

        updates, jump = _hook_result(await scope.run(self._config(invocation.id, spec), call))
        step.record(live.state, node, [*scope.sent, *updates])
        return jump

    async def _model(self, invocation: Invocation, spec: AgentSpec, live: _Live, step: _Step, answers: list[Any]) -> None:
        scope = self._scope(invocation, spec, live, "model", str(uuid.uuid4()), answers)

        async def call(_: RunnableConfig) -> list[Command]:
            values = live.state.values()
            request = ModelRequest(
                model=spec.model,
                tools=spec.default_tools,
                system_message=spec.system_message,
                response_format=spec.response_format,
                messages=_langchain(live.state.model_context()),
                tool_choice=None,
                state=cast("AgentState[Any]", values),
                runtime=scope.runtime,
            )
            handler = partial(execute_model, spec)
            if spec.wrap_model_call is None:
                return _build_commands(await handler(request))
            composed = await spec.wrap_model_call(request, handler)
            return _build_commands(composed.model_response, composed.commands)

        commands = await scope.run(self._config(invocation.id, spec), call)
        step.record(live.state, "model", [*scope.sent, *_updates(commands)])

    async def _sequence(
        self, invocation: Invocation, spec: AgentSpec, live: _Live, step: _Step, sequence: str, at: int, answers: list[Any]
    ) -> str | _Stopped:
        """Run a sequence's steps from `at`; returns where the loop goes next, or why it stopped."""
        if sequence == "turn" and at == 0:
            self._check_steps(invocation)
        stages = _stages(spec, sequence)
        jump = None
        for index in range(at, len(stages)):
            hook, middleware = stages[index]
            given = answers if index == at else []
            try:
                if middleware is None:
                    await self._model(invocation, spec, live, step, given)
                else:
                    jump = await self._hook_step(invocation, spec, live, step, hook, middleware, given)
            except GraphInterrupt as raised:
                node = "model" if middleware is None else f"{middleware.name}.{hook}"
                return _Stopped(asked(raised), sequence, index, node, given)
            if jump is not None:
                break
        return _route(spec, sequence, jump, live.state.values())

    async def _drive(
        self,
        invocation: Invocation,
        spec: AgentSpec,
        live: _Live,
        step: _Step,
        sequence: str,
        *,
        at: int = 0,
        answers: list[Any] | None = None,
        retire_pending: bool = False,
    ) -> None:
        """Run sequences from `sequence` until the run must commit, then commit or stop.

        The `agent` sequence flows straight into the first turn; any other
        destination is the next phase.
        """
        given = answers or []
        while True:
            following = await self._sequence(invocation, spec, live, step, sequence, at, given)
            if isinstance(following, _Stopped):
                pending = {
                    "kind": "interrupt",
                    "sequence": following.sequence,
                    "at": following.at,
                    "node": following.node,
                    "answers": [serde.dump(answer) for answer in following.answers],
                    "interrupts": [_stored(item) for item in following.interrupts],
                }
                await self._stop(invocation, live, step, pending)
                return
            if sequence == "agent" and following == "turn":
                sequence, at, given = "turn", 0, []
                continue
            await self._commit(invocation, spec, live, step, following, retire_pending=retire_pending)
            return

    def _check_steps(self, invocation: Invocation) -> None:
        limit = self._context(invocation.id).config.get("recursion_limit", 9_999)
        if invocation.checkpoint.get("step", 0) > limit:
            msg = f"Recursion limit of {limit} reached without hitting a stop condition."
            raise GraphRecursionError(msg)

    # Run phases.

    async def _start(self, invocation: Invocation) -> None:
        spec = await self.agent(invocation.input["agent"])
        live = await self._live_state(invocation, spec)
        payload = invocation.input["payload"]
        command = payload.get("command") or {}
        step = _Step()
        step.record(live.state, "__start__", list(spec.schema.inputs(serde.load(payload.get("input")) or {}).items()))
        if command.get("update") is not None:
            step.record(live.state, "__start__", list(serde.load(command["update"]).items()))
        step.nodes.clear()
        # Input is not streamed back as output, but the values with it are, as the run's first state.
        live.seen.update(message.id for operation, message in step.change.messages if operation in {"add", "replace"})
        self._publish(invocation, "values", live.state.output())
        pending, resume = live.state.pending, serde.load(command.get("resume"))
        if pending is not None and resume is not None:
            await self._resume(invocation, spec, live, step, pending, resume)
            return
        await self._drive(invocation, spec, live, step, "agent", retire_pending=pending is not None)

    async def _resume(self, invocation: Invocation, spec: AgentSpec, live: _Live, step: _Step, pending: dict[str, Any], resume: Any) -> None:  # noqa: ANN401
        """Continue a stopped thread with the input it was waiting for."""
        if pending["kind"] == "interrupt":
            answer = _resume_value(resume, pending["interrupts"][0]["id"])
            answers = [*(serde.load(item) for item in pending["answers"]), answer]
            await self._drive(invocation, spec, live, step, pending["sequence"], at=pending["at"], answers=answers, retire_pending=True)
            return
        answers: dict[str, Any] = {}
        for call, waiting in pending["waiting"].items():
            if waiting["subagent"]:
                answers[call] = {"resume": serde.dump(resume)}
            else:
                answers[call] = {"answers": [*waiting["answers"], serde.dump(_resume_value(resume, waiting["interrupts"][0]))]}
        live.state.pending = None
        try:
            async with invocation.step() as stage:
                live.state.persist(stage.tx, invocation.conversation, step.change, by_task=invocation.id)
                stage.tx.retire_doc(PENDING, thread.scope(invocation.conversation))
                self._spawn(invocation, spec, live, stage, done=pending["done"], answers=answers)
        except BaseException:
            self._forget(invocation.id)
            raise

    async def _turn(self, invocation: Invocation) -> None:
        spec = await self.agent(invocation.input["agent"])
        await self._drive(invocation, spec, await self._live_state(invocation, spec), _Step(), "turn")

    async def _finish(self, invocation: Invocation) -> None:
        spec = await self.agent(invocation.input["agent"])
        await self._drive(invocation, spec, await self._live_state(invocation, spec), _Step(), "finish")

    # Tool rounds.

    async def _collect(self, invocation: Invocation) -> None:
        spec = await self.agent(invocation.input["agent"])
        checkpoint = invocation.checkpoint
        done: dict[str, Any] = dict(checkpoint["done"])
        waiting: dict[str, Any] = {}
        interrupts: list[dict[str, Any]] = []
        for task_id in checkpoint["tasks"]:
            record = await invocation.session.task(task_id)
            outcome: dict[str, Any] = (
                record["state"]["outcome"] if record else {"status": "faulted", "error": {"message": f"tool task {task_id} is missing"}}
            )
            if outcome["status"] != "completed":
                raise RuntimeError(outcome.get("error", {}).get("message", f"tool task {task_id} ended {outcome['status']}"))
            result = outcome["result"]
            if "interrupts" in result:
                interrupts.extend(result["interrupts"])
                ids = [item["id"] for item in result["interrupts"]]
                waiting[result["call"]] = {"subagent": result.get("subagent", False), "answers": result.get("answers", []), "interrupts": ids}
            else:
                done[result["call"]] = result["updates"]
        live = await self._live_state(invocation, spec)
        if interrupts:
            await self._stop(
                invocation, live, _Step(), {"kind": "tools", "node": "tools", "done": done, "waiting": waiting, "interrupts": interrupts}
            )
            return
        step = _Step()
        step.record(live.state, "tools", [(name, serde.load(value)) for call in checkpoint["calls"] for name, value in done.get(call, [])])
        following = _after_tools(spec, live.state.values())
        if following == "turn":
            await self._drive(invocation, spec, live, step, "turn")
        else:
            await self._commit(invocation, spec, live, step, DONE if not spec.after_agent else following)

    async def _tool_call(self, invocation: Invocation) -> None:
        spec = await self.agent(invocation.input["agent"])
        call = invocation.input["call"]
        run = invocation.task["owner"]
        live = await self._live_state(invocation, spec)
        task_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{invocation.conversation}:tools:{call['id']}"))
        answers = [serde.load(answer) for answer in invocation.input.get("answers", [])]
        scope = self._scope(invocation, spec, live, "tools", task_id, answers)
        tool_node = spec.tool_node
        if tool_node is None:
            msg = "the agent has no tools"
            raise RuntimeError(msg)
        payload = {"__type": "tool_call_with_context", "tool_call": call, "state": live.state.values()}
        token = CURRENT_TOOL.set(
            ToolScope(
                runtime=self,
                task=invocation.id,
                conversation=invocation.conversation,
                call_id=call["id"],
                thread=invocation.input["thread"],
                ns=(*invocation.input["ns"], f"tools:{task_id}"),
                resume=serde.load(invocation.input.get("resume")),
                context=self._context(run),
                agent=invocation.input["agent"],
            )
        )
        try:
            output = await scope.run(self._config(run, spec), lambda config: tool_node.ainvoke(payload, config))
        except AwaitingInputError as waiting:
            async with invocation.step() as stage:
                stage.finish({"call": call["id"], "interrupts": waiting.interrupts, "subagent": True})
            return
        except GraphInterrupt as raised:
            async with invocation.step() as stage:
                stage.finish(
                    {"call": call["id"], "interrupts": [_stored(item) for item in asked(raised)], "answers": invocation.input.get("answers", [])}
                )
            return
        finally:
            CURRENT_TOOL.reset(token)
        updates = [*scope.sent, *_updates(output)]
        async with invocation.step() as stage:
            stage.finish({"call": call["id"], "updates": [[name, serde.dump(value)] for name, value in updates]})
        self._publish(invocation, "updates", {"tools": _merge(updates)})

    # Aborts and faults.

    async def _abort_run(self, invocation: Invocation) -> None:
        self._forget(invocation.id)
        async with invocation.step() as stage:
            stage.tx.put_submission(self._settled(invocation, status="unanswered", reason="aborted"))
            stage.aborted()
        self._ended(invocation)

    async def _abort_tool(self, invocation: Invocation) -> None:
        async with invocation.step() as stage:
            stage.aborted()

    async def _run_faulted(self, tx: _core.Tx, task: dict[str, Any], message: str) -> None:
        submission = task["input"]["submission"]
        tx.put_submission(
            {
                "id": submission,
                "conversationId": task["conversationId"],
                "type": "input",
                "status": "unanswered",
                "reason": "faulted",
                "detail": message,
            }
        )
        self._forget(task["id"])
        self._contexts.pop(task["id"], None)
        self._busy.pop(task["conversationId"], None)
        self.bus.publish(Event(thread=task["input"]["thread"], ns=tuple(task["input"]["ns"]), mode=END_EVENT, data=None, run=task["id"]))
