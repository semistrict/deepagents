"""Token streaming for `stream_mode="messages"`.

Attaching a handler that is a `_StreamingCallbackHandler` makes chat models
stream; each chunk is forwarded with its model run's metadata, as LangGraph's
`StreamMessagesHandler` does. Tokens are not durable: the committed message is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langgraph.constants import TAG_NOSTREAM

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from uuid import UUID

    from langchain_core.messages import BaseMessage, BaseMessageChunk


class TokenTap(BaseCallbackHandler, _StreamingCallbackHandler):
    """Forwards streamed chat model chunks with their run's metadata."""

    run_inline = True

    def __init__(self, emit: Callable[[BaseMessageChunk, dict[str, Any]], None]) -> None:
        """Send every chunk to `emit(chunk, metadata)`."""
        self._emit = emit
        self._runs: dict[UUID, dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002  # callback signature
        messages: list[list[BaseMessage]],  # noqa: ARG002  # callback signature
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        """Remember a streaming model run's metadata unless it opted out."""
        if TAG_NOSTREAM not in (tags or []):
            self._runs[run_id] = {**(metadata or {}), "tags": tags or []}

    def on_llm_new_token(self, token: str | list[str | dict[str, Any]], *, chunk: Any = None, run_id: UUID, **_: Any) -> None:  # noqa: ANN401, ARG002
        """Forward one chunk."""
        metadata = self._runs.get(run_id)
        if metadata is not None and isinstance(chunk, ChatGenerationChunk):
            self._emit(chunk.message, metadata)

    def on_llm_end(self, response: Any, *, run_id: UUID, **_: Any) -> None:  # noqa: ANN401, ARG002
        """Forget a finished run."""
        self._runs.pop(run_id, None)

    def tap_output_aiter(self, run_id: UUID, output: AsyncIterator[Any]) -> AsyncIterator[Any]:  # noqa: ARG002
        """Pass output through unchanged."""
        return output

    def tap_output_iter(self, run_id: UUID, output: Iterator[Any]) -> Iterator[Any]:  # noqa: ARG002
        """Pass output through unchanged."""
        return output
