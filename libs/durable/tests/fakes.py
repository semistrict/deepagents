"""A scripted chat model for tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool


class ScriptedModel(BaseChatModel):
    """Answers with the next scripted message, repeating the last one when the script runs out.

    Streams a text answer word by word, so token streaming can be observed.
    """

    script: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        index = min(len(self.calls), len(self.script)) - 1
        return self.script[index].model_copy()

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **_: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **_: Any
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next(messages)
        words = message.text.split(" ") if isinstance(message.content, str) and message.content else [""]
        for index, word in enumerate(words):
            last = index == len(words) - 1
            text = word if last else f"{word} "
            chunk = AIMessageChunk(content=text, id=message.id, tool_call_chunks=self._tool_chunks(message) if last else [])
            if run_manager is not None:
                run_manager.on_llm_new_token(text, chunk=ChatGenerationChunk(message=chunk))
            yield ChatGenerationChunk(message=chunk)

    @staticmethod
    def _tool_chunks(message: AIMessage) -> list[dict[str, Any]]:
        return [
            {"name": call["name"], "args": __import__("json").dumps(call["args"]), "id": call["id"], "index": i, "type": "tool_call_chunk"}
            for i, call in enumerate(message.tool_calls)
        ]

    def bind_tools(self, tools: Sequence[dict[str, Any] | type | BaseTool | Any], **_: Any) -> Runnable[Any, BaseMessage]:
        return self


def call(name: str, call_id: str, **args: Any) -> dict[str, Any]:
    """A tool call."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ai(*calls: dict[str, Any], text: str = "") -> AIMessage:
    """An AI message with tool calls, or a final answer."""
    return AIMessage(content=text, tool_calls=list(calls))
