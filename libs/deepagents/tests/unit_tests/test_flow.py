"""Unit tests for the sync/async flow drivers in `deepagents._flow`."""

import asyncio

import pytest

from deepagents._flow import Call, Flow, Gather, Io, NoHandlerError, arun_flow, io, run_flow
from deepagents.backends.protocol import BackendProtocol, WriteResult


class SomeError(Exception):
    """Sentinel error for retry tests."""


def echo_flow(request: str) -> Flow[str]:
    """Yield one Call and return its result, uppercased."""
    response = yield Call(request)
    return response.upper()


def test_run_flow_call_dispatches_to_handler() -> None:
    result = run_flow(echo_flow("req"), lambda request: f"handled:{request}")
    assert result == "HANDLED:REQ"


async def test_arun_flow_call_dispatches_to_handler() -> None:
    async def handler(request: str) -> str:
        return f"handled:{request}"

    result = await arun_flow(echo_flow("req"), handler)
    assert result == "HANDLED:REQ"


def test_run_flow_return_without_effects() -> None:
    def flow() -> Flow[int]:
        return 42
        yield  # pragma: no cover - makes this a generator

    assert run_flow(flow()) == 42


def test_run_flow_io_uses_sync_arm() -> None:
    def flow() -> Flow[str]:
        return (yield Io(lambda: "sync", _fail_async))

    assert run_flow(flow()) == "sync"


async def test_arun_flow_io_uses_async_arm() -> None:
    async def async_arm() -> str:
        return "async"

    def flow() -> Flow[str]:
        return (yield Io(_fail_sync, async_arm))

    assert await arun_flow(flow()) == "async"


def test_handler_exception_is_thrown_into_flow() -> None:
    def flow(request: str) -> Flow[str]:
        try:
            return (yield Call(request))
        except SomeError:
            return (yield Call("retry"))

    calls: list[str] = []

    def handler(request: str) -> str:
        calls.append(request)
        if request == "first":
            raise SomeError
        return "ok"

    assert run_flow(flow("first"), handler) == "ok"
    assert calls == ["first", "retry"]


async def test_handler_exception_is_thrown_into_flow_async() -> None:
    def flow(request: str) -> Flow[str]:
        try:
            return (yield Call(request))
        except SomeError:
            return (yield Call("retry"))

    async def handler(request: str) -> str:
        if request == "first":
            raise SomeError
        return "ok"

    assert await arun_flow(flow("first"), handler) == "ok"


def test_uncaught_exception_propagates() -> None:
    def flow() -> Flow[str]:
        return (yield Call("req"))

    def handler(request: str) -> str:
        raise SomeError(request)

    with pytest.raises(SomeError):
        run_flow(flow(), handler)


def test_exception_raised_by_flow_propagates() -> None:
    def flow() -> Flow[str]:
        msg = "from flow"
        raise ValueError(msg)
        yield  # pragma: no cover - makes this a generator

    with pytest.raises(ValueError, match="from flow"):
        run_flow(flow())


def test_call_without_handler_raises() -> None:
    def flow() -> Flow[str]:
        return (yield Call("req"))

    with pytest.raises(NoHandlerError):
        run_flow(flow())


async def test_call_without_handler_raises_async() -> None:
    def flow() -> Flow[str]:
        return (yield Call("req"))

    with pytest.raises(NoHandlerError):
        await arun_flow(flow())


def test_flows_compose_with_yield_from() -> None:
    def inner() -> Flow[str]:
        return (yield Io(lambda: "inner-io", _fail_async))

    def outer() -> Flow[str]:
        inner_result = yield from inner()
        response = yield Call(inner_result)
        return f"outer:{response}"

    assert run_flow(outer(), lambda request: f"handled:{request}") == "outer:handled:inner-io"


def test_multiple_calls_in_one_flow() -> None:
    def flow() -> Flow[list[str]]:
        first = yield Call("one")
        second = yield Call("two")
        return [first, second]

    assert run_flow(flow(), lambda request: request * 2) == ["oneone", "twotwo"]


def test_io_binds_loop_variables_eagerly() -> None:
    def flow() -> Flow[list[str]]:
        results = []
        for name in ["a", "b", "c"]:
            results.append((yield io(str.upper, _fail_async_upper, name)))  # noqa: PERF401 -- yield is illegal inside a comprehension
        return results

    assert run_flow(flow()) == ["A", "B", "C"]


def test_gather_runs_subflows_sequentially_in_order() -> None:
    order: list[str] = []

    def sub(name: str) -> Flow[str]:
        order.append(f"start:{name}")
        result = yield Io(name.upper, _fail_async)
        order.append(f"end:{name}")
        return result

    def flow() -> Flow[list[str]]:
        return (yield Gather(sub("a"), sub("b")))

    assert run_flow(flow()) == ["A", "B"]
    assert order == ["start:a", "end:a", "start:b", "end:b"]


async def test_gather_runs_subflows_concurrently() -> None:
    first_started = asyncio.Event()

    def waiter() -> Flow[str]:
        # Blocks until the second sub-flow runs -- only completes if Gather
        # actually drives sub-flows concurrently.
        async def wait() -> str:
            await asyncio.wait_for(first_started.wait(), timeout=5)
            return "waited"

        return (yield Io(_fail_sync, wait))

    def setter() -> Flow[str]:
        async def set_event() -> str:
            first_started.set()
            return "set"

        return (yield Io(_fail_sync, set_event))

    def flow() -> Flow[list[str]]:
        return (yield Gather(waiter(), setter()))

    assert await arun_flow(flow()) == ["waited", "set"]


def test_backend_flows_write_pairs_sync_and_async() -> None:
    backend = _RecordingBackend()

    def flow() -> Flow[WriteResult]:
        return (yield backend.flows.write("/f.txt", "content"))

    result = run_flow(flow())
    assert result.error is None
    assert backend.calls == ["write:/f.txt"]


async def test_backend_flows_write_uses_async_twin() -> None:
    backend = _RecordingBackend()

    def flow() -> Flow[WriteResult]:
        return (yield backend.flows.write("/f.txt", "content"))

    result = await arun_flow(flow())
    assert result.error is None
    assert backend.calls == ["awrite:/f.txt"]


class _RecordingBackend(BackendProtocol):
    """Backend recording which of write/awrite was invoked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def write(self, file_path: str, content: str) -> WriteResult:
        del content
        self.calls.append(f"write:{file_path}")
        return WriteResult(path=file_path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        del content
        self.calls.append(f"awrite:{file_path}")
        return WriteResult(path=file_path)


def _fail_sync() -> str:
    msg = "sync arm must not run under arun_flow"
    raise AssertionError(msg)


async def _fail_async() -> str:
    msg = "async arm must not run under run_flow"
    raise AssertionError(msg)


async def _fail_async_upper(name: str) -> str:
    del name
    msg = "async arm must not run under run_flow"
    raise AssertionError(msg)
