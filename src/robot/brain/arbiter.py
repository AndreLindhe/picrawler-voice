from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from ..behaviors.base import Behavior
from ..core.bus import MessageBus
from ..core.events import (
    BehaviorEnded,
    BehaviorStarted,
    ObstacleCleared,
    ObstacleDetected,
)
from ..core.state import StateManager

logger = logging.getLogger(__name__)

PRIORITY_PATROL = 0
PRIORITY_VOICE = 1
PRIORITY_NAVIGATE = 2   # reserved for Phase 2 TaskBehavior (voice-triggered tasks)


class _Request:
    __slots__ = ("behavior", "priority", "reason")

    def __init__(self, behavior: Behavior, priority: int, reason: str) -> None:
        self.behavior = behavior
        self.priority = priority
        self.reason = reason


class Arbiter:
    """
    Owns behavior preemption and the patrol-as-default loop.

    Priority order (higher wins):
        PRIORITY_PATROL = 0   — idle wandering
        PRIORITY_VOICE  = 1   — explicit voice command
        obstacle reflex       — handled out-of-band; suspends everything

    Usage::

        arbiter = Arbiter(bus, state, make_patrol=lambda: IdlePatrol(bus, state))
        asyncio.create_task(arbiter.run())

        # From the brain / hearing loop:
        arbiter.request(GoTo(bus, state, target="chair"), PRIORITY_VOICE, "user said go to chair")
    """

    def __init__(
        self,
        bus: MessageBus,
        state: StateManager,
        make_patrol: Callable[[], Behavior],
    ) -> None:
        self._bus = bus
        self._state = state
        self._make_patrol = make_patrol

        self._current_task: Optional[asyncio.Task] = None
        self._current_behavior: Optional[Behavior] = None
        self._current_priority: int = -1

        self._pending: Optional[_Request] = None

        # Set when the supervisor should re-evaluate (task done, new request, etc.)
        self._wakeup = asyncio.Event()

        # Cleared when an obstacle is present; set when path is clear.
        self._path_clear = asyncio.Event()
        self._path_clear.set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request(self, behavior: Behavior, priority: int, reason: str = "") -> None:
        """
        Ask the arbiter to run `behavior`.  If priority >= current, the
        current behavior is cancelled and `behavior` takes over.
        Safe to call from any coroutine in the same asyncio loop.
        """
        if priority < self._current_priority:
            logger.debug(
                "arbiter: ignoring %s (priority %d < current %d)",
                behavior.name,
                priority,
                self._current_priority,
            )
            return

        logger.info("arbiter: queuing %s (priority=%d, reason=%r)", behavior.name, priority, reason)
        self._pending = _Request(behavior, priority, reason)

        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

        self._wakeup.set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Long-lived supervisor task.  Start with asyncio.create_task(arbiter.run())."""
        asyncio.create_task(self._watch_obstacles(), name="arbiter.obstacle_watcher")

        try:
            await self._main_loop()
        finally:
            # On shutdown, cancel the current behavior and let its cleanup run.
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
                try:
                    await asyncio.shield(self._current_task)
                except (asyncio.CancelledError, Exception):
                    pass

    async def _main_loop(self) -> None:
        while True:
            # Step 0: reap finished task and publish event
            if self._current_task is not None and self._current_task.done():
                self._reap()

            # Step 1: block while obstacle is present
            await self._path_clear.wait()

            # Step 2: start a behavior if idle
            if self._current_task is None:
                req = self._pending or _Request(
                    self._make_patrol(), PRIORITY_PATROL, "default-patrol"
                )
                self._pending = None
                self._launch(req)

            # Step 3: wait for the next signal
            # Check *before* waiting in case the signal arrived while we worked.
            if not self._wakeup.is_set():
                await self._wakeup.wait()
            self._wakeup.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _launch(self, req: _Request) -> None:
        self._current_behavior = req.behavior
        self._current_priority = req.priority
        self._current_task = asyncio.create_task(
            req.behavior.run(), name=req.behavior.name
        )
        # Wake the supervisor when the task finishes (natural or cancelled).
        self._current_task.add_done_callback(lambda _: self._wakeup.set())
        self._bus.publish(BehaviorStarted(name=req.behavior.name, priority=req.priority))
        logger.info("arbiter: started %s", req.behavior.name)

    def _reap(self) -> None:
        assert self._current_task is not None
        t = self._current_task
        name = self._current_behavior.name if self._current_behavior else "unknown"

        if t.cancelled():
            reason = "cancelled"
        elif t.exception() is not None:
            reason = "error"
            logger.exception("arbiter: behavior %s raised", name, exc_info=t.exception())
        else:
            reason = "natural"

        self._bus.publish(BehaviorEnded(name=name, reason=reason))
        logger.info("arbiter: %s ended (%s)", name, reason)

        self._current_task = None
        self._current_behavior = None
        self._current_priority = -1

    async def _cancel_current_and_wait(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._current_task is not None and self._current_task.done():
            self._reap()

    async def _watch_obstacles(self) -> None:
        """Subscribes to obstacle events and suspends the arbiter when path is blocked."""
        async with self._bus.subscribe(
            ObstacleDetected.topic, ObstacleCleared.topic
        ) as q:
            while True:
                event = await q.get()
                if isinstance(event, ObstacleDetected):
                    logger.warning(
                        "arbiter: obstacle at %.1f cm — suspending behavior", event.distance_cm
                    )
                    self._path_clear.clear()
                    await self._cancel_current_and_wait()
                    # NOTE: on clear we resume with patrol, not the interrupted behavior.
                    # To restore an interrupted voice command instead, stash self._pending here.
                elif isinstance(event, ObstacleCleared):
                    logger.info("arbiter: path clear — resuming")
                    self._path_clear.set()
                    self._wakeup.set()
