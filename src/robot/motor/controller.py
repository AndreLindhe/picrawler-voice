from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton — set when CrawlerController is first constructed.
# Behaviours and emergency-stop paths call get_controller() to reach it.
_instance: Optional["CrawlerController"] = None


def get_controller() -> "CrawlerController":
    if _instance is None:
        raise RuntimeError(
            "CrawlerController has not been initialised — "
            "create CrawlerController() before calling get_controller()"
        )
    return _instance


class CrawlerController:
    """
    Async wrapper around Picrawler.

    All Picrawler calls block the calling thread (servo_move uses time.sleep).
    This class dispatches every call to a *single-worker* ThreadPoolExecutor so
    that concurrent callers are queued rather than racing over servo state.

    Usage::

        ctrl = CrawlerController()          # once at startup
        asyncio.create_task(ctrl.stand())   # safe from any coroutine
    """

    # Speed presets (0–100)
    SPEED_SLOW = 40
    SPEED_NORMAL = 60
    SPEED_FAST = 80

    def __init__(self) -> None:
        global _instance
        from picrawler import Picrawler  # type: ignore[import]

        logger.info("controller: initialising Picrawler (resets MCU, moves servos to home)")
        self._crawler = Picrawler()
        # Single worker = all servo calls are strictly serialised.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crawler")
        self._lock = asyncio.Lock()
        _instance = self

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    async def _run(self, fn, *args) -> None:
        """Dispatch a blocking Picrawler call to the single-worker executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, fn, *args)

    # ------------------------------------------------------------------
    # Posture
    # ------------------------------------------------------------------

    async def stand(self, speed: int = SPEED_SLOW) -> None:
        async with self._lock:
            logger.debug("controller: stand")
            await self._run(self._crawler.do_step, "stand", speed)

    async def sit(self, speed: int = SPEED_SLOW) -> None:
        """Lower the robot to its resting position. Use as 'stop'."""
        async with self._lock:
            logger.debug("controller: sit")
            await self._run(self._crawler.do_step, "sit", speed)

    # ------------------------------------------------------------------
    # Locomotion (one gait cycle per call)
    # ------------------------------------------------------------------

    async def forward(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "forward", steps, speed)

    async def backward(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "backward", steps, speed)

    async def turn_left(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "turn left", steps, speed)

    async def turn_right(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "turn right", steps, speed)

    async def turn_left_angle(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "turn left angle", steps, speed)

    async def turn_right_angle(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "turn right angle", steps, speed)

    # ------------------------------------------------------------------
    # Head / expression
    # ------------------------------------------------------------------

    async def look_left(self, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "look left", 1, speed)

    async def look_right(self, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "look right", 1, speed)

    async def look_up(self, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "look up", 1, speed)

    async def look_down(self, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "look down", 1, speed)

    async def wave(self, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "wave", 1, speed)

    # ------------------------------------------------------------------
    # Generic escape hatch for actions not wrapped above
    # ------------------------------------------------------------------

    async def do_action(self, name: str, steps: int = 1, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, name, steps, speed)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Sit down then shut down the executor cleanly."""
        await self.sit(speed=self.SPEED_SLOW)
        self._executor.shutdown(wait=True)
        logger.info("controller: shut down")
