"""ChatOpenAI subclass for the ChatGPT backend API.

Strips reasoning content blocks from messages before sending, because the
ChatGPT backend with ``store=false`` cannot look up reasoning item IDs from
previous responses.

Also injects the built-in ``web_search`` tool into every request so the
model can search the web alongside regular function tools.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

_WEB_SEARCH_TOOL = {"type": "web_search"}


def _strip_reasoning(messages: list[BaseMessage]) -> None:
    """Remove reasoning content blocks from AI messages in-place."""
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            cleaned = [
                c for c in content
                if not (isinstance(c, dict) and c.get("type") == "reasoning")
            ]
            if len(cleaned) != len(content):
                msg.content = cleaned  # type: ignore[assignment]


def _inject_web_search(kwargs: dict[str, Any]) -> None:
    """Ensure the built-in web_search tool is in the tools list."""
    tools = kwargs.get("tools")
    if tools is None:
        kwargs["tools"] = [_WEB_SEARCH_TOOL]
        return
    if not any(t.get("type") == "web_search" for t in tools if isinstance(t, dict)):
        tools.append(_WEB_SEARCH_TOOL)


class ChatGPTOpenAI(ChatOpenAI):
    """ChatOpenAI variant for the ChatGPT backend API."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _strip_reasoning(messages)
        _inject_web_search(kwargs)
        return super()._generate(messages, stop=stop, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        _strip_reasoning(messages)
        _inject_web_search(kwargs)
        yield from super()._stream(messages, stop=stop, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _strip_reasoning(messages)
        _inject_web_search(kwargs)
        return await super()._agenerate(messages, stop=stop, **kwargs)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _strip_reasoning(messages)
        _inject_web_search(kwargs)
        async for chunk in super()._astream(messages, stop=stop, **kwargs):
            yield chunk
