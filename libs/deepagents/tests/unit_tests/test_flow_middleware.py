"""Unit tests for `deepagents.middleware._flow_base.FlowMiddleware`."""

import inspect
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from deepagents._flow import Call, Flow, Io
from deepagents.middleware._flow_base import FlowMiddleware


class _ModelCallOnly(FlowMiddleware):
    """Middleware defining only `model_call_flow`."""

    def model_call_flow(self, request: str) -> Flow[str]:
        """Uppercase the request before the call."""
        return (yield Call(request.upper()))


class _NodeOnly(FlowMiddleware):
    """Middleware defining only `before_agent_flow`, with an extended signature."""

    def before_agent_flow(self, state: dict[str, Any], runtime: object, config: dict[str, Any]) -> Flow[dict[str, Any]]:
        """Return a state update built from an Io effect."""
        del runtime, config
        value = yield Io(lambda: "sync-io", _async_io)
        return {**state, "loaded": value}


class _Empty(FlowMiddleware):
    """Middleware defining no flows at all."""


class _ExplicitHook(FlowMiddleware):
    """Middleware whose explicit hook must win over the generated one."""

    def model_call_flow(self, request: str) -> Flow[str]:
        """Never driven via wrap_model_call: the explicit hook bypasses it."""
        return (yield Call(request))

    def wrap_model_call(self, request: str, handler: Callable[[str], str]) -> str:  # type: ignore[override]
        """Explicit hook that ignores the flow."""
        del request
        return handler("explicit")


class _Subclassed(_ModelCallOnly):
    """Overrides only the flow; inherits the generated drivers."""

    def model_call_flow(self, request: str) -> Flow[str]:
        """Reverse instead of uppercase."""
        return (yield Call(request[::-1]))


async def _async_io() -> str:
    return "async-io"


def test_wrap_hooks_are_generated_onto_the_subclass() -> None:
    assert _ModelCallOnly.__dict__["wrap_model_call"] is not AgentMiddleware.wrap_model_call
    assert _ModelCallOnly.__dict__["awrap_model_call"] is not AgentMiddleware.awrap_model_call
    # Only the flow's hooks are generated -- no node hooks, no tool hooks.
    assert "before_agent" not in _ModelCallOnly.__dict__
    assert "wrap_tool_call" not in _ModelCallOnly.__dict__


def test_no_flows_means_no_generated_hooks() -> None:
    assert _Empty.wrap_model_call is AgentMiddleware.wrap_model_call
    assert _Empty.before_agent is AgentMiddleware.before_agent


def test_generated_wrap_model_call_runs_sync() -> None:
    result = _ModelCallOnly().wrap_model_call("req", lambda request: f"handled:{request}")
    assert result == "handled:REQ"


async def test_generated_awrap_model_call_runs_async() -> None:
    async def handler(request: str) -> str:
        return f"handled:{request}"

    result = await _ModelCallOnly().awrap_model_call("req", handler)
    assert result == "handled:REQ"


def test_generated_node_hooks_run_both_worlds() -> None:
    mw = _NodeOnly()
    assert mw.before_agent({"x": 1}, None, {}) == {"x": 1, "loaded": "sync-io"}


async def test_generated_node_hook_async_uses_async_arm() -> None:
    mw = _NodeOnly()
    assert await mw.abefore_agent({"x": 1}, None, {}) == {"x": 1, "loaded": "async-io"}


def test_node_hook_adopts_flow_signature() -> None:
    params = list(inspect.signature(_NodeOnly.before_agent).parameters)
    assert params == ["self", "state", "runtime", "config"]


def test_explicit_hook_wins_over_generated_one() -> None:
    result = _ExplicitHook().wrap_model_call("req", lambda request: f"handled:{request}")
    assert result == "handled:explicit"
    # The async twin was still generated from the flow.
    assert "awrap_model_call" in _ExplicitHook.__dict__


def test_subclass_overriding_flow_reuses_inherited_drivers() -> None:
    result = _Subclassed().wrap_model_call("abc", lambda request: f"handled:{request}")
    assert result == "handled:cba"


def test_can_jump_to_is_copied_to_node_hooks() -> None:
    def after_agent_flow(self: FlowMiddleware, state: dict[str, Any], runtime: object) -> Flow[None]:
        del self, state, runtime
        return None
        yield  # pragma: no cover - makes this a generator

    after_agent_flow.__can_jump_to__ = ["model"]  # type: ignore[attr-defined]
    cls = type("_Jumpy", (FlowMiddleware,), {"after_agent_flow": after_agent_flow})
    assert cls.after_agent.__can_jump_to__ == ["model"]
    assert cls.aafter_agent.__can_jump_to__ == ["model"]
