"""Stream events from runs to `astream` callers.

A run publishes each event right after the commit it describes, so a caller
never sees anything that is not durable. Callers may live on other event
loops than the runs, so delivery is thread-safe.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

END = "__end__"
"""Mode of the event a run publishes after its final commit."""


@dataclass(frozen=True)
class Event:
    """One stream event of a thread or of a subagent below it."""

    thread: int
    """The conversation of the top-level thread the event belongs to."""
    ns: tuple[str, ...]
    """Namespace below that thread: `()` for the thread itself."""
    mode: str
    data: Any
    run: int | None = None
    """The run task that published it."""


class Subscription:
    """The events of one thread, in publication order."""

    def __init__(self, bus: EventBus, thread: int) -> None:
        """Subscribe on the running event loop."""
        self.thread = thread
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue[Event] = asyncio.Queue()
        self._bus = bus

    def deliver(self, event: Event) -> None:
        """Queue an event from any thread."""
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

    async def next(self) -> Event:
        """The next event."""
        return await self.queue.get()

    def close(self) -> None:
        """Stop receiving events."""
        self._bus.unsubscribe(self)


class EventBus:
    """Fans events out to the subscriptions of their thread."""

    def __init__(self) -> None:
        """Start with no subscribers."""
        self._subscriptions: list[Subscription] = []
        self._lock = threading.Lock()

    def subscribe(self, thread: int) -> Subscription:
        """Receive events of `thread` from now on."""
        subscription = Subscription(self, thread)
        with self._lock:
            self._subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        """Stop delivering to `subscription`."""
        with self._lock:
            self._subscriptions = [existing for existing in self._subscriptions if existing is not subscription]

    def publish(self, event: Event) -> None:
        """Deliver to every subscription of the event's thread."""
        with self._lock:
            targets = [subscription for subscription in self._subscriptions if subscription.thread == event.thread]
        for subscription in targets:
            subscription.deliver(event)
