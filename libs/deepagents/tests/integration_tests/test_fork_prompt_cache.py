from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any, Protocol

import pytest
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_deep_agent

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.skipif(
    os.environ.get("DEEPAGENTS_RUN_LIVE_PROMPT_CACHE_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="Set DEEPAGENTS_RUN_LIVE_PROMPT_CACHE_TESTS=1 and OPENAI_API_KEY to run live prompt-cache tests.",
)

MODEL_ID = os.environ.get("DEEPAGENTS_OPENAI_PROMPT_CACHE_MODEL", "gpt-5.4-mini")
SUBAGENT_COUNT = 10
PAIR_COUNT = 18


def _live_model() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_ID, temperature=0)


def _cacheable_prefix() -> str:
    chunk = (
        "Stable prior conversation detail for prompt-cache verification. "
        "This sentence is intentionally repeated inside historical turns so the forked "
        "subagent has a long inherited message prefix to reuse. "
    )
    return chunk * 8


SYSTEM_PROMPT = "Use the task tool calls requested by the scripted parent model."
USER_PROMPT = "Delegate to every cache probe subagent so each one can recover its assigned paired number from prior context."
MIN_EXPECTED_CACHED_TOKENS = 4_096
MIN_EXPECTED_CACHE_RATIO = 0.80
PAIR_BY_KEY = {
    "38472": "918263",
    "75910": "284650",
    "12638": "775491",
    "90841": "336702",
    "47205": "640119",
    "63194": "502873",
    "81527": "197346",
    "29386": "854201",
    "54072": "419638",
    "68713": "730524",
    "34958": "265917",
    "71620": "948305",
    "20549": "573816",
    "89314": "681270",
    "45826": "307945",
    "97035": "126584",
    "62108": "792463",
    "13479": "845026",
}
PAIR_KEYS = list(PAIR_BY_KEY)


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
    response_metadata = message.response_metadata
    usage_metadata = message.usage_metadata

    return max(
        _nested_int(response_metadata, "token_usage", "prompt_tokens_details", "cached_tokens"),
        _nested_int(response_metadata, "usage", "prompt_tokens_details", "cached_tokens"),
        _nested_int(response_metadata, "prompt_tokens_details", "cached_tokens"),
        _nested_int(usage_metadata, "input_token_details", "cache_read"),
    )


def _input_token_count(message: AIMessage) -> int:
    response_metadata = message.response_metadata
    usage_metadata = message.usage_metadata
    return (
        _nested_int(response_metadata, "token_usage", "prompt_tokens")
        or _nested_int(response_metadata, "usage", "prompt_tokens")
        or _nested_int(usage_metadata, "input_tokens")
    )


def _long_prior_conversation() -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for idx, key in enumerate(PAIR_KEYS):
        value = PAIR_BY_KEY[key]
        messages.append(
            HumanMessage(
                content=(
                    f"Historical user turn {idx}: {_cacheable_prefix()} "
                    f"Record lookup pair KEY-{key} VALUE-{value}. "
                    "Remember this as shared context before any subagent is forked."
                )
            )
        )
        messages.append(
            AIMessage(
                content=(
                    f"Historical assistant turn {idx}: {_cacheable_prefix()} "
                    f"I have stored lookup pair KEY-{key} VALUE-{value}. "
                    "This response is part of the parent conversation prefix."
                )
            )
        )
    messages.append(HumanMessage(content=USER_PROMPT))
    return messages


class _CaptureModelResponses(AgentMiddleware[Any, Any, Any]):
    def __init__(self, *, name: str) -> None:
        super().__init__()
        self.subagent_name = name
        self.messages: list[AIMessage] = []

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        response = handler(request)
        self.messages.extend(message for message in response.result if isinstance(message, AIMessage))
        return response


class _CaptureBySubagent(AgentMiddleware[Any, Any, Any]):
    def __init__(self) -> None:
        super().__init__()
        self.messages_by_subagent: dict[str, list[AIMessage]] = {}

    def middleware_for(self, name: str) -> _CaptureModelResponses:
        middleware = _CaptureModelResponses(name=name)
        self.messages_by_subagent[name] = middleware.messages
        return middleware

    def clear(self) -> None:
        for messages in self.messages_by_subagent.values():
            messages.clear()


def _subagent_name(idx: int) -> str:
    return f"cache_probe_{idx}"


def _subagent_key(idx: int) -> str:
    return PAIR_KEYS[idx]


def _parallel_fork_tool_calls() -> list[dict[str, Any]]:
    return [
        {
            "name": "task",
            "args": {
                "description": (f"Find the paired VALUE for KEY-{_subagent_key(idx)} in the prior conversation. Respond with only the VALUE digits."),
                "subagent_type": _subagent_name(idx),
            },
            "id": f"call-cache-probe-{idx}",
            "type": "tool_call",
        }
        for idx in range(SUBAGENT_COUNT)
    ]


class _ScriptParentTaskCalls(AgentMiddleware[Any, Any, Any]):
    def __init__(self) -> None:
        super().__init__()
        self._remaining = [
            AIMessage(content="", tool_calls=_parallel_fork_tool_calls()),
            AIMessage(content="All probes complete."),
            AIMessage(content="", tool_calls=_parallel_fork_tool_calls()),
            AIMessage(content="All probes complete."),
        ]

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        if self._remaining:
            return ModelResponse(result=[self._remaining.pop(0)])
        return handler(request)


def _invoke_cache_probe(agent: _Invokable) -> None:
    agent.invoke(
        {"messages": _long_prior_conversation()},
        config={"configurable": {"thread_id": f"live-prompt-cache-{uuid.uuid4()}"}},
    )


def test_forked_subagent_reports_openai_prompt_cache_reuse() -> None:
    capture = _CaptureBySubagent()
    agent = create_deep_agent(
        model=_live_model(),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
        middleware=[_ScriptParentTaskCalls()],
        subagents=[
            {
                "name": _subagent_name(idx),
                "description": f"Retrieves the paired value for live prompt-cache verification probe {idx}.",
                "system_prompt": (
                    f"You are cache probe {idx}. Your assigned lookup key is KEY-{_subagent_key(idx)}. "
                    "Use the inherited prior conversation to find the matching VALUE. "
                    "Return only the VALUE digits and no other text."
                ),
                "tools": [],
                "middleware": [capture.middleware_for(_subagent_name(idx))],
                "fork": True,
            }
            for idx in range(SUBAGENT_COUNT)
        ],
    )

    _invoke_cache_probe(agent)
    capture.clear()
    _invoke_cache_probe(agent)

    failures: list[str] = []
    for name, messages in capture.messages_by_subagent.items():
        if not messages:
            failures.append(f"{name}: no forked model response captured")
            continue
        idx = int(name.rsplit("_", maxsplit=1)[1])
        expected_value = PAIR_BY_KEY[_subagent_key(idx)]
        fork_response = max(messages, key=_cached_token_count)
        response_text = fork_response.text
        cached = _cached_token_count(fork_response)
        input_tokens = _input_token_count(fork_response)
        ratio = cached / input_tokens if input_tokens else 0
        if expected_value not in response_text:
            failures.append(f"{name}: expected VALUE-{expected_value}, got {response_text!r}")
        if cached < MIN_EXPECTED_CACHED_TOKENS or ratio < MIN_EXPECTED_CACHE_RATIO:
            failures.append(f"{name}: input_tokens={input_tokens}, cached_tokens={cached}, ratio={ratio:.2%}")

    assert not failures, "Expected substantial fork prompt-cache reuse for every subagent:\n" + "\n".join(failures)
