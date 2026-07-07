"""Middleware for filtering excluded tools from model requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents._flow import Call
from deepagents.middleware._flow_base import FlowMiddleware

if TYPE_CHECKING:
    from langchain.agents.middleware.types import (
        ModelRequest,
        ModelResponse,
    )
    from langchain_core.tools import BaseTool

    from deepagents._flow import Flow


def _tool_name(tool: BaseTool | dict[str, str]) -> str | None:
    """Extract tool name from a `BaseTool` or dict tool."""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


class _ToolExclusionMiddleware(FlowMiddleware[Any, Any, Any]):
    """Middleware that filters excluded tools from the model request.

    Should be placed late in the middleware stack (after all
    tool-injecting middleware) so it can strip middleware-injected tools
    (filesystem, subagent, etc.) that the harness profile marks as excluded.

    Args:
        excluded: Tool names to remove before the model sees them.
    """

    def __init__(self, *, excluded: frozenset[str]) -> None:
        self._excluded = excluded

    def model_call_flow(self, request: ModelRequest[Any]) -> Flow[ModelResponse[Any]]:
        """Filter excluded tools before they reach the model."""
        if self._excluded:
            filtered = [t for t in request.tools if _tool_name(t) not in self._excluded]
            request = request.override(tools=filtered)
        return (yield Call(request))
