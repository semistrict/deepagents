"""Single-source bodies for paired sync/async middleware hooks.

`AgentMiddleware` exposes every hook twice (`wrap_model_call` /
`awrap_model_call`, `before_agent` / `abefore_agent`, ...), which forces
middleware whose logic is identical in both worlds to maintain two
near-verbatim copies of each body, diverging only at a handful of I/O calls.

This module removes that duplication. A hook body is written once as a
*flow*: a generator that yields effects and receives their results back at
the `yield`. Two drivers execute a flow -- `run_flow` resolves each effect
synchronously, `arun_flow` asynchronously:

- yield `Call(request)` -- invoke the hook's wrapped handler (the model or
    tool call being intercepted) with `request`.
- yield `Io(sync, async_)` -- run middleware-owned I/O; the driver picks the
    implementation matching its world. When both arms take the same
    arguments, build the effect with `io(sync_fn, async_fn, *args)`.
- yield `Gather(*flows)` -- run sub-flows to completion: sequentially under
    `run_flow`, concurrently (via `asyncio.gather`) under `arun_flow`.

An exception raised while executing an effect is thrown back into the flow
at the `yield`, so `try/except` around a `yield` behaves exactly like
`try/except` around a direct call. The flow's `return` value becomes the
hook's return value. Flows compose with `yield from`.

Example:
    ```python
    def model_call_flow(self, request: ModelRequest) -> Flow[ModelResponse]:
        data = yield flows(backend).read(path)
        try:
            return (yield Call(request))
        except SomeTransientError:
            return (yield Call(fallback_request))
    ```

Middleware classes should not drive flows by hand: subclassing
`deepagents.middleware._flow_base.FlowMiddleware` generates the sync/async
hook pairs from flow methods like the one above. `run_flow`/`arun_flow` are
for non-hook boundaries (tool implementations, module-level helpers).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar, cast

R = TypeVar("R")

SyncHandler: TypeAlias = Callable[[Any], Any]
AsyncHandler: TypeAlias = Callable[[Any], Awaitable[Any]]


@dataclass(frozen=True)
class Call:
    """Effect: invoke the hook's wrapped handler with `request`.

    Yielding this from a flow performs the intercepted model or tool call.
    The handler's return value is sent back into the flow; a flow may yield
    `Call` any number of times (retries, fallbacks) or not at all
    (short-circuit).
    """

    request: Any


@dataclass(frozen=True)
class Io:
    """Effect: run middleware-owned I/O with one implementation per world.

    `sync` is a no-argument callable executed by `run_flow`; `async_` is a
    no-argument callable returning an awaitable, executed by `arun_flow`.
    Both must produce the same logical result so the flow stays
    world-agnostic.
    """

    sync: Callable[[], Any]
    async_: Callable[[], Awaitable[Any]]


class Gather:
    """Effect: run sub-flows to completion, sequentially in sync and concurrently in async.

    `run_flow` drives each sub-flow to completion in argument order;
    `arun_flow` drives them concurrently via `asyncio.gather`. The result
    sent back into the parent flow is the list of sub-flow return values, in
    argument order. Sub-flows may yield any effect; the parent's handler is
    passed through.
    """

    __slots__ = ("flows",)

    def __init__(self, *flows: Flow[Any]) -> None:
        """Capture the sub-flows to run."""
        self.flows = flows


Effect: TypeAlias = Call | Io | Gather

Flow: TypeAlias = Generator[Effect, Any, R]


def io(sync_fn: Callable[..., Any], async_fn: Callable[..., Awaitable[Any]], /, *args: Any, **kwargs: Any) -> Io:
    """Build an `Io` effect from a sync/async callable pair sharing one argument list.

    Arguments are bound eagerly, so loop variables are captured by value.
    """
    return Io(lambda: sync_fn(*args, **kwargs), lambda: async_fn(*args, **kwargs))


class NoHandlerError(RuntimeError):
    """A flow yielded `Call` but the driver was not given a handler.

    Hooks without a wrapped handler (`before_agent`, `after_model`, ...)
    drive their flows handler-less; such flows may only yield `Io`.
    """

    def __init__(self) -> None:
        """Initialize with a fixed message."""
        super().__init__("Flow yielded Call() but was driven without a handler; only wrap_* hooks can Call.")


def run_flow(flow: Flow[R], handler: SyncHandler | None = None) -> R:
    """Drive `flow` to completion, resolving every effect synchronously."""
    try:
        effect = next(flow)
        while True:
            try:
                result = _execute(effect, handler)
            except Exception as exc:  # noqa: BLE001 -- delivered to the flow's `yield`; re-raised there if uncaught
                effect = flow.throw(exc)
            else:
                effect = flow.send(result)
    except StopIteration as stop:
        return cast("R", stop.value)


async def arun_flow(flow: Flow[R], handler: AsyncHandler | None = None) -> R:
    """Drive `flow` to completion, resolving every effect asynchronously."""
    try:
        effect = next(flow)
        while True:
            try:
                result = await _aexecute(effect, handler)
            except Exception as exc:  # noqa: BLE001 -- delivered to the flow's `yield`; re-raised there if uncaught
                effect = flow.throw(exc)
            else:
                effect = flow.send(result)
    except StopIteration as stop:
        return cast("R", stop.value)


def _execute(effect: Effect, handler: SyncHandler | None) -> Any:  # noqa: ANN401 -- effect results are inherently untyped
    if isinstance(effect, Call):
        if handler is None:
            raise NoHandlerError
        return handler(effect.request)
    if isinstance(effect, Gather):
        return [run_flow(flow, handler) for flow in effect.flows]
    return effect.sync()


async def _aexecute(effect: Effect, handler: AsyncHandler | None) -> Any:  # noqa: ANN401 -- effect results are inherently untyped
    if isinstance(effect, Call):
        if handler is None:
            raise NoHandlerError
        return await handler(effect.request)
    if isinstance(effect, Gather):
        return await asyncio.gather(*(arun_flow(flow, handler) for flow in effect.flows))
    return await effect.async_()
