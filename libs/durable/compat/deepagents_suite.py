"""Pytest plugin that runs deepagents' own test suite on the durable runtime.

Every `create_agent` deepagents makes, for the main agent and its declarative
subagents, becomes a durable agent. LangGraph checkpointers passed by tests
are ignored: threads live in a fresh in-memory kernel per test.

    make deepagents-suite
"""

from __future__ import annotations

from typing import Any

import deepagents.graph
import deepagents.middleware.subagents
import pytest

from deepagents_durable import agent as durable
from deepagents_durable.kernel import Kernel


def _factory(model: Any, tools: Any = None, *, checkpointer: Any = None, **kwargs: Any) -> durable.DurableAgent:  # noqa: ANN401
    kernel = checkpointer if isinstance(checkpointer, Kernel) else None
    return durable.create_agent(model, tools, checkpointer=kernel, **kwargs)


@pytest.fixture(autouse=True)
def _durable_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deepagents.graph, "create_agent", _factory)
    monkeypatch.setattr(deepagents.middleware.subagents, "create_agent", _factory)
    monkeypatch.setattr(durable, "_defaults", {})
