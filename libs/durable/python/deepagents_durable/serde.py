"""JSON encoding of messages and state values for durable records.

Messages use LangChain's own dict form. Other values are plain JSON, with
tagged objects for messages nested inside values, Pydantic models,
dataclasses, and bytes so they come back as the same types.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib
from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

_MESSAGE = "__lc_message__"
_PYDANTIC = "__pydantic__"
_DATACLASS = "__dataclass__"
_BYTES = "__bytes__"


def dump_message(message: BaseMessage) -> dict[str, Any]:
    """Encode a message as LangChain's `{"type", "data"}` dict."""
    return message_to_dict(message)


_MESSAGE_TYPES: dict[str, type[BaseMessage]] = {
    "human": HumanMessage,
    "ai": AIMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
    "chat": ChatMessage,
    "function": FunctionMessage,
    "AIMessageChunk": AIMessageChunk,
    "HumanMessageChunk": HumanMessageChunk,
}


def load_message(data: dict[str, Any]) -> BaseMessage:
    """Decode a message encoded by `dump_message`.

    The fields were validated when the message was built, before it was
    stored, so known types are reconstructed without validating again.
    """
    kind = _MESSAGE_TYPES.get(data["type"])
    if kind is None:
        return messages_from_dict([data])[0]
    return kind.model_construct(**data["data"])


def _qualified(kind: type) -> str:
    return f"{kind.__module__}:{kind.__qualname__}"


def _resolve(name: str) -> type:
    module, _, qualname = name.partition(":")
    target: Any = importlib.import_module(module)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def dump(value: Any) -> Any:  # noqa: ANN401, PLR0911  # one branch per supported shape
    """Encode a state value as JSON."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseMessage):
        return {_MESSAGE: dump_message(value)}
    if isinstance(value, BaseModel):
        return {_PYDANTIC: _qualified(type(value)), "value": value.model_dump(mode="json")}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {field.name: dump(getattr(value, field.name)) for field in dataclasses.fields(value)}
        return {_DATACLASS: _qualified(type(value)), "value": fields}
    if isinstance(value, bytes):
        return {_BYTES: base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [dump(item) for item in value]
    msg = f"cannot store a value of type {type(value).__name__}"
    raise TypeError(msg)


def _load_pydantic(value: dict[str, Any]) -> Any:  # noqa: ANN401  # any model
    return _resolve(value[_PYDANTIC]).model_validate(value["value"])  # ty: ignore[unresolved-attribute]


def _load_dataclass(value: dict[str, Any]) -> Any:  # noqa: ANN401  # any dataclass
    return _resolve(value[_DATACLASS])(**{key: load(item) for key, item in value["value"].items()})


_TAGGED: dict[str, Callable[[dict[str, Any]], Any]] = {
    _MESSAGE: lambda value: load_message(value[_MESSAGE]),
    _PYDANTIC: _load_pydantic,
    _DATACLASS: _load_dataclass,
    _BYTES: lambda value: base64.b64decode(value[_BYTES]),
}
"""Decoders of the tagged objects `dump` writes, by tag."""


def load(value: Any) -> Any:  # noqa: ANN401  # mirrors dump
    """Decode a value encoded by `dump`."""
    if isinstance(value, list):
        return [load(item) for item in value]
    if not isinstance(value, dict):
        return value
    for tag, decode in _TAGGED.items():
        if tag in value:
            return decode(value)
    return {key: load(item) for key, item in value.items()}
