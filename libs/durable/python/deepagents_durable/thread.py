"""An agent thread: a conversation whose transcript holds the messages.

- Each message is an immutable `lc.message` entry, appended once.
- Replacing or removing a message appends an `lc.edit` entry naming the
  entry it edits, as pi-durable's context edits do; the original stays.
- `REMOVE_ALL_MESSAGES` appends an `lc.reset` entry that starts a new context.
- Every other stored field is one rewindable conversation document.

A step therefore writes only what changed, and a thread grows linearly with
its messages.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolMessage, convert_to_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Overwrite

from deepagents_durable import serde
from deepagents_durable.schema import MESSAGES, Schema

if TYPE_CHECKING:
    from deepagents_durable import _core

THREAD_IDS = "lc.thread-id"
FIELD = "lc.field"
PENDING = "lc.pending"
"""Conversation document: what a stopped run awaits, and where it continues."""
MESSAGE = "lc.message"
EDIT = "lc.edit"
RESET = "lc.reset"

SESSION = {"kind": "session"}


def scope(conversation: int) -> dict[str, Any]:
    """The document scope of a conversation."""
    return {"kind": "conversation", "conversationId": conversation}


async def find(tx: _core.Tx, thread_id: str) -> int | None:
    """The conversation of a thread ID, as the transaction sees it."""
    found = await tx.doc(THREAD_IDS, SESSION, thread_id)
    return None if found is None else found["conversationId"]


def create(tx: _core.Tx, thread_id: str | None, *, owner: dict[str, int] | None = None) -> int:
    """Create a thread's conversation, registering its thread ID when it has one."""
    conversation = tx.create_conversation(owner=owner)
    if thread_id is not None:
        tx.put_doc(THREAD_IDS, SESSION, {"conversationId": conversation}, key=thread_id)
    return conversation


def _messages(value: Any) -> list[BaseMessage]:  # noqa: ANN401  # any messages update
    """A messages update as typed messages with IDs."""
    items = value if isinstance(value, list) else [value]
    messages = [item if isinstance(item, BaseMessage) else convert_to_messages([item])[0] for item in items]
    for message in messages:
        if message.id is None:
            message.id = str(uuid.uuid4())
    return messages


def _id(message: BaseMessage) -> str:
    """A message's ID: every message a thread holds has one, stamped on its way in."""
    if message.id is None:
        msg = f"a thread message has no ID: {message!r}"
        raise ValueError(msg)
    return message.id


@dataclass
class Change:
    """What one step changed, to store and to stream."""

    updates: list[tuple[str, Any]] = field(default_factory=list)
    """The step's updates as applied, messages typed and with IDs."""
    messages: list[tuple[str, Any]] = field(default_factory=list)
    """Transcript operations: `("add" | "replace", message)`, `("remove", id)`, or `("reset", None)`."""
    fields: set[str] = field(default_factory=set)

    def __bool__(self) -> bool:
        """Whether anything changed."""
        return bool(self.messages or self.fields)

    def extend(self, other: Change) -> None:
        """Fold a later change into this one."""
        self.updates.extend(other.updates)
        self.messages.extend(other.messages)
        self.fields |= other.fields


class ThreadState:
    """A thread's messages and fields, as this process last committed or loaded them."""

    def __init__(self, schema: Schema) -> None:
        """An empty thread."""
        self.schema = schema
        self._messages: list[BaseMessage | None] = []
        self._position: dict[str, int] = {}
        self._entry: dict[str, int] = {}
        """The entry that stores each message, for edits."""
        self._view: list[BaseMessage] | None = []
        """The current messages, rebuilt only after a change."""
        self._owner: dict[str, str] = {}
        """The AI message of each tool call ID."""
        self._results: dict[str, int] = {}
        """How many tool results answer each tool call ID."""
        self._open: dict[str, str] = {}
        """Tool calls without a result, with their AI message."""
        self.fields: dict[str, Any] = {}
        self.pending: dict[str, Any] | None = None
        """What a stopped run of this thread awaits, if one did."""

    @property
    def messages(self) -> list[BaseMessage]:
        """The current messages, in order. Do not mutate the list."""
        if self._view is None:
            self._view = [message for message in self._messages if message is not None]
        return self._view

    def model_context(self) -> list[BaseMessage]:
        """The messages a model request sees, with dangling tool calls answered; see `model_context`."""
        messages = self.messages
        last = messages[-1].id if messages else None
        if all(owner == last for owner in self._open.values()):
            return list(messages)
        return model_context(messages)

    def _track(self, message: BaseMessage | None, delta: int) -> None:
        """Count a message's tool calls and results in (`delta=1`) or out (`delta=-1`)."""
        if isinstance(message, AIMessage):
            # A call without an ID can never be answered, so it is not tracked.
            for call_id in (call["id"] for call in message.tool_calls if call["id"] is not None):
                if delta > 0:
                    self._owner[call_id] = _id(message)
                    if not self._results.get(call_id):
                        self._open[call_id] = _id(message)
                else:
                    self._owner.pop(call_id, None)
                    self._open.pop(call_id, None)
        elif isinstance(message, ToolMessage):
            call = message.tool_call_id
            self._results[call] = self._results.get(call, 0) + delta
            if self._results[call] > 0:
                self._open.pop(call, None)
            elif call in self._owner:
                self._open[call] = self._owner[call]

    def values(self) -> dict[str, Any]:
        """The whole state, as hooks and tools see it."""
        return {MESSAGES: self.messages, **self.fields}

    def output(self) -> dict[str, Any]:
        """The state an invocation returns."""
        return self.schema.outputs(self.values())

    def _add(self, message: BaseMessage) -> str:
        self._view = None
        position = self._position.get(_id(message))
        if position is None:
            self._position[_id(message)] = len(self._messages)
            self._messages.append(message)
            self._track(message, 1)
            return "add"
        self._track(self._messages[position], -1)
        self._messages[position] = message
        self._track(message, 1)
        return "replace"

    def _remove(self, message_id: str) -> bool:
        position = self._position.pop(message_id, None)
        if position is None:
            return False
        self._view = None
        self._track(self._messages[position], -1)
        self._messages[position] = None
        return True

    def _reset(self) -> None:
        self._messages, self._position, self._entry, self._view = [], {}, {}, []
        self._owner, self._results, self._open = {}, {}, {}

    def _apply_messages(self, messages: list[BaseMessage], change: Change, *, overwrite: bool) -> None:
        if overwrite:
            self._reset()
            change.messages.append(("reset", None))
        for message in messages:
            if isinstance(message, RemoveMessage) and message.id == REMOVE_ALL_MESSAGES:
                self._reset()
                change.messages.append(("reset", None))
            elif isinstance(message, RemoveMessage):
                if self._remove(_id(message)):
                    change.messages.append(("remove", message.id))
            else:
                change.messages.append((self._add(message), message))

    def apply(self, updates: list[tuple[str, Any]]) -> Change:
        """Apply one step's updates in order."""
        change = Change()
        writes: dict[str, list[Any]] = {}
        for name, value in updates:
            if name == MESSAGES:
                messages = _messages(value.value if isinstance(value, Overwrite) else value)
                self._apply_messages(messages, change, overwrite=isinstance(value, Overwrite))
                change.updates.append((name, messages))
            else:
                writes.setdefault(name, []).append(value)
                change.updates.append((name, value))
        for name, values in writes.items():
            self.fields[name] = self.schema.merge(name, self.fields.get(name), values)
            change.fields.add(name)
        return change

    def persist(self, tx: _core.Tx, conversation: int, change: Change, *, by_task: int | None = None) -> None:
        """Store a change applied to this state."""
        for operation, payload in change.messages:
            if operation == "reset":
                tx.append_entry(conversation, RESET, None, head="self", by_task=by_task)
            elif operation == "remove":
                edit = {"edits": [{"target": self._entry.pop(payload), "action": "omit"}]}
                tx.append_entry(conversation, EDIT, edit, by_task=by_task)
            elif operation == "replace" and payload.id in self._entry:
                edit = {"edits": [{"target": self._entry[payload.id], "action": "replace", "messages": [serde.dump_message(payload)]}]}
                tx.append_entry(conversation, EDIT, edit, by_task=by_task)
            else:
                self._entry[payload.id] = tx.append_entry(conversation, MESSAGE, {"data": serde.dump_message(payload)}, by_task=by_task)
        for name in sorted(change.fields):
            if not self.schema.stored(name):
                continue
            value = {"value": serde.dump(self.fields[name])}
            tx.put_doc(FIELD, scope(conversation), value, key=name, history="rewindable", fork="asOf")

    def fold(self, entries: list[dict[str, Any]]) -> None:
        """Rebuild the messages from a conversation's active context entries."""
        self._reset()
        by_entry: dict[int, str] = {}
        for stored in entries:
            entry = stored["entry"]
            if entry["kind"] == MESSAGE:
                message = serde.load_message(entry["data"])
                self._add(message)
                self._entry[_id(message)] = entry["id"]
                by_entry[entry["id"]] = _id(message)
            elif entry["kind"] == EDIT:
                for edit in entry["edits"]:
                    message_id = by_entry.get(edit["target"])
                    if message_id is None:
                        continue
                    if edit["action"] == "omit":
                        self._remove(message_id)
                    else:
                        self._add(serde.load_message(edit["messages"][0]))


async def load(session: _core.Session, conversation: int, schema: Schema, at: int | None = None) -> ThreadState:
    """A thread's state now, or as of commit `at`."""
    state = ThreadState(schema)
    state.fold(await session.context(conversation, at))
    if at is None:
        stored = await session.doc(PENDING, scope(conversation))
        state.pending = None if stored is None else stored["value"]
    for record in await session.docs(scope(conversation), FIELD):
        name = record["key"]
        stored = await session.doc(FIELD, scope(conversation), name, at)
        if stored is not None:
            state.fields[name] = serde.load(stored["value"]["value"])
    return state


def model_context(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The messages a model request sees: every tool call answered, as pi-durable derives context.

    A call left without a result, by an interrupted run or a new message that
    arrived first, gets an error result right after its AI message.
    """
    answered = {message.tool_call_id for message in messages if isinstance(message, ToolMessage)}
    context: list[BaseMessage] = []
    for message in messages:
        context.append(message)
        if isinstance(message, AIMessage):
            context.extend(
                ToolMessage(
                    content=f"Tool call {call['name']} with id {call['id']} was cancelled - another message came in before it could be completed.",
                    name=call["name"],
                    tool_call_id=call["id"],
                )
                for call in message.tool_calls
                if call["id"] not in answered and message is not messages[-1]
            )
    return context
