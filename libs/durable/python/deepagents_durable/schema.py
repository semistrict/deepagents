"""Agent state fields: how updates to each one merge, and whether it is stored or shown.

Middleware declares its state as `TypedDict` schemas. A field annotated with a
reducer, such as `Annotated[dict, merge]` or a `DeltaChannel(reducer)`, merges
updates with that function; any other field takes the last value written.
Fields marked `UntrackedValue` or `EphemeralValue` live only in memory, and
`OmitFromSchema` hides a field from invocation input or output. The
`messages` field is not merged here: the thread stores it as entries.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NotRequired, Required, get_args, get_origin, get_type_hints

from langchain.agents.middleware.types import OmitFromSchema
from langgraph.channels.base import BaseChannel
from langgraph.channels.delta import DeltaChannel
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.types import Overwrite

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

MESSAGES = "messages"


@dataclass(frozen=True)
class Field:
    """One state field's rules."""

    name: str
    merge: Callable[[Any, list[Any]], Any] | None
    """Combines the current value with one step's writes; `None` keeps the last write."""
    stored: bool
    """Whether the value is persisted; untracked and ephemeral fields are not."""
    input: bool
    output: bool


def _batch(reducer: Callable[[Any, Any], Any]) -> Callable[[Any, list[Any]], Any]:
    """A binary reducer applied to each write in turn."""

    def merge(current: Any, writes: list[Any]) -> Any:  # noqa: ANN401  # any field value
        for write in writes:
            current = write if current is None else reducer(current, write)
        return current

    return merge


def _unwrap(annotation: Any) -> Any:  # noqa: ANN401  # any type annotation
    if get_origin(annotation) in {Required, NotRequired}:
        return get_args(annotation)[0]
    return annotation


def _field(name: str, annotation: Any) -> Field:  # noqa: ANN401  # any type annotation
    metadata = getattr(_unwrap(annotation), "__metadata__", ())
    merge: Callable[[Any, list[Any]], Any] | None = None
    stored = True
    for item in metadata:
        if isinstance(item, DeltaChannel):
            merge = item.reducer
        elif isinstance(item, (UntrackedValue, EphemeralValue)) or item is UntrackedValue or item is EphemeralValue:
            stored = False
        elif isinstance(item, BaseChannel) or (inspect.isclass(item) and issubclass(item, BaseChannel)):
            continue
        elif callable(item) and not isinstance(item, OmitFromSchema) and len(inspect.signature(item).parameters) == 2:  # noqa: PLR2004  # (current, update)
            merge = _batch(item)
    omit_input = any(isinstance(item, OmitFromSchema) and item.input for item in metadata)
    omit_output = any(isinstance(item, OmitFromSchema) and item.output for item in metadata)
    return Field(name=name, merge=merge, stored=stored, input=not omit_input, output=not omit_output)


@functools.lru_cache(maxsize=256)
def _hints(schema: type) -> dict[str, Any]:
    return get_type_hints(schema, include_extras=True)


class Schema:
    """The fields of an agent's state, merged from its schemas; a later schema wins a field."""

    def __init__(self, schemas: Iterable[type]) -> None:
        """Merge `schemas` in order."""
        annotations: dict[str, Any] = {}
        for schema in schemas:
            annotations.update({name: hint for name, hint in _hints(schema).items() if name != "__slots__"})
        self.fields = {name: _field(name, annotation) for name, annotation in annotations.items()}
        self.private = frozenset(name for name, field in self.fields.items() if not field.input and not field.output)

    def merge(self, name: str, current: Any, writes: list[Any]) -> Any:  # noqa: ANN401  # any field value
        """A field's value after one step's writes. An `Overwrite` bypasses the merge."""
        field = self.fields.get(name)
        overwrites = [index for index, write in enumerate(writes) if isinstance(write, Overwrite)]
        if overwrites:
            last = overwrites[-1]
            current, writes = writes[last].value, writes[last + 1 :]
            if not writes:
                return current
        if field is None or field.merge is None:
            return writes[-1]
        return field.merge(current, writes)

    def stored(self, name: str) -> bool:
        """Whether a field's value is persisted."""
        field = self.fields.get(name)
        return field is None or field.stored

    def inputs(self, values: dict[str, Any]) -> dict[str, Any]:
        """The part of an invocation's input the schema accepts."""
        return {name: value for name, value in values.items() if name in self.fields and self.fields[name].input}

    def outputs(self, values: dict[str, Any]) -> dict[str, Any]:
        """The part of the state an invocation returns."""
        return {name: value for name, value in values.items() if name not in self.fields or self.fields[name].output}
