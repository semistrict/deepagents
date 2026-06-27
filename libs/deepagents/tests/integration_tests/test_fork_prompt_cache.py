from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any, Protocol

import pytest
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_deep_agent

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.skipif(
    os.environ.get("DEEPAGENTS_RUN_LIVE_PROMPT_CACHE_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="Set DEEPAGENTS_RUN_LIVE_PROMPT_CACHE_TESTS=1 and OPENAI_API_KEY to run live prompt-cache tests.",
)

MODEL_ID = os.environ.get("DEEPAGENTS_OPENAI_PROMPT_CACHE_MODEL", "gpt-5.4")


def _live_model() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_ID, temperature=0)


def _cacheable_prefix() -> str:
    chunk = (
        "Stable prompt-cache verification sentence. "
        "The following prefix is intentionally repeated so provider prompt caching "
        "has enough tokens to report a cache read. "
    )
    return chunk * 260


SYSTEM_PROMPT = _cacheable_prefix() + "\n\nAlways use the task tool exactly once. Select the cache_probe subagent and do not answer directly."
USER_PROMPT = "Delegate to the cache_probe subagent with the instruction: confirm the cache probe."


class _Invokable(Protocol):
    def invoke(self, values: dict[str, Any], config: dict[str, Any]) -> object: ...


def _nested_int(value: object, *path: str) -> int:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return 0
        current = current.get(key)
    return current if isinstance(current, int) else 0


def _cached_token_count(message: AIMessage) -> int:
    """Return cached-input token count from known provider metadata shapes."""
    total = 0
    response_metadata = message.response_metadata
    usage_metadata = message.usage_metadata

    total += _nested_int(response_metadata, "token_usage", "prompt_tokens_details", "cached_tokens")
    total += _nested_int(response_metadata, "usage", "prompt_tokens_details", "cached_tokens")
    total += _nested_int(response_metadata, "prompt_tokens_details", "cached_tokens")
    total += _nested_int(usage_metadata, "input_token_details", "cache_read")
    return total


class _CaptureModelResponses(AgentMiddleware[Any, Any, Any]):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[AIMessage] = []

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        response = handler(request)
        self.messages.extend(message for message in response.result if isinstance(message, AIMessage))
        return response


def _invoke_cache_probe(agent: _Invokable) -> None:
    agent.invoke(
        {"messages": [{"role": "user", "content": USER_PROMPT}]},
        config={"configurable": {"thread_id": f"live-prompt-cache-{uuid.uuid4()}"}},
    )


def test_forked_subagent_reports_openai_prompt_cache_reuse() -> None:
    capture = _CaptureModelResponses()
    agent = create_deep_agent(
        model=_live_model(),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
        subagents=[
            {
                "name": "cache_probe",
                "description": "Reports a short answer for live prompt-cache verification.",
                "system_prompt": "Reply with exactly: cache probe complete",
                "tools": [],
                "middleware": [capture],
                "fork": True,
            }
        ],
    )

    _invoke_cache_probe(agent)
    capture.messages.clear()
    _invoke_cache_probe(agent)

    assert capture.messages, "Forked subagent model response was not captured; the parent may not have delegated."
    cached = max(_cached_token_count(message) for message in capture.messages)
    assert cached > 0
