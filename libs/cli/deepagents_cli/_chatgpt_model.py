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

from langchain_core.messages import BaseMessage, SystemMessage
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


def _extract_system_to_instructions(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], str | None]:
    """Move SystemMessages out of the list and return them as instructions text.

    The OpenAI Responses API rejects ``{"role": "system", ...}`` items in the
    input array. The system prompt must be passed via the ``instructions``
    parameter instead. This helper extracts all SystemMessages, concatenates
    their text, and returns the filtered list together with the combined
    instructions string (or ``None`` when there were no system messages).
    """
    system_parts: list[str] = []
    filtered: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            content = msg.content
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.extend(
                    block if isinstance(block, str) else str(block)
                    for block in content
                )
        else:
            filtered.append(msg)
    if not system_parts:
        return messages, None
    return filtered, "\n\n".join(system_parts)


class ChatGPTOpenAI(ChatOpenAI):
    """ChatOpenAI variant for the ChatGPT backend API."""

    def _prepare(
        self, messages: list[BaseMessage], kwargs: dict[str, Any]
    ) -> list[BaseMessage]:
        """Common pre-processing for every call path."""
        _strip_reasoning(messages)
        _inject_web_search(kwargs)
        # The Responses API rejects system-role items in the input array.
        # Move them to the ``instructions`` parameter instead.
        messages, instructions = _extract_system_to_instructions(messages)
        if instructions is not None:
            kwargs["instructions"] = instructions
        return messages

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        messages = self._prepare(messages, kwargs)
        return super()._generate(messages, stop=stop, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        messages = self._prepare(messages, kwargs)
        yield from super()._stream(messages, stop=stop, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        messages = self._prepare(messages, kwargs)
        return await super()._agenerate(messages, stop=stop, **kwargs)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        messages = self._prepare(messages, kwargs)
        async for chunk in super()._astream(messages, stop=stop, **kwargs):
            yield chunk
