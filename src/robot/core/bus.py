from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator, List

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_MAX = 32


class MessageBus:
    """
    In-process asyncio pub/sub.

    Publish by event type (each type carries a `topic` ClassVar).
    Subscribe by topic string. Multiple topics per subscriber are fine.

    If a subscriber's queue is full the *oldest* event is dropped so a slow
    consumer (e.g. the LLM brain loop) can never stall the motor safety loop.
    """

    def __init__(self, queue_max: int = _DEFAULT_QUEUE_MAX) -> None:
        self._queue_max = queue_max
        self._subscribers: dict[str, List[asyncio.Queue]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: Any) -> None:
        topic: str = type(event).topic
        queues = self._subscribers.get(topic, [])
        for q in queues:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                logger.warning("bus: dropped oldest event on topic %r (subscriber lagging)", topic)
            q.put_nowait(event)

    # ------------------------------------------------------------------
    # Subscribing
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def subscribe(self, *topics: str) -> AsyncIterator[asyncio.Queue]:
        """
        Async context manager. Yields a Queue that receives events for the
        given topics. Unregisters automatically on exit.

        Usage::

            async with bus.subscribe("motor.obstacle_detected") as q:
                while True:
                    event = await q.get()
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_max)
        for topic in topics:
            self._subscribers[topic].append(q)
        try:
            yield q
        finally:
            for topic in topics:
                try:
                    self._subscribers[topic].remove(q)
                except ValueError:
                    pass

    async def stream(self, *topics: str) -> AsyncGenerator[Any, None]:
        """
        Async generator that yields events indefinitely.

        Usage::

            async for event in bus.stream("hearing.transcript"):
                handle(event)
        """
        async with self.subscribe(*topics) as q:
            while True:
                yield await q.get()
