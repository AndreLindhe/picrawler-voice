from __future__ import annotations

import asyncio
import logging
import time as _time_mod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

# Servos are grouped in legs of 3. Activating all 12 at once trips the Pi's
# overcurrent protection. We patch robot_hat.robot.time.sleep during Picrawler
# construction so that after every 3rd servo (one complete leg) we wait long
# enough for those servos to settle before the next leg starts.
_LEG_SIZE = 3
_INTER_LEG_PAUSE_S = 0.8

# Hard cap on servo speed. Values above this can draw enough current to trip
# the overcurrent/overvoltage protection, especially during multi-step gaits.
_MAX_SPEED = 62

# Brief pause inserted after every action so servos reach their target position
# before the next command starts. Without this, back-to-back commands (e.g.
# forward then sit) can find legs mid-travel and fold into bad angles.
_SETTLE_S = 0.18


def _make_picrawler():
    """Construct Picrawler with per-leg inrush limiting."""
    import robot_hat.robot as _rr
    from picrawler import Picrawler  # type: ignore[import]

    _count = [0]
    _orig_sleep = _rr.time.sleep

    def _leg_sleep(t: float) -> None:
        _count[0] += 1
        if _count[0] % _LEG_SIZE == 0:
            _orig_sleep(_INTER_LEG_PAUSE_S)
        else:
            _orig_sleep(t)

    _rr.time.sleep = _leg_sleep
    try:
        return Picrawler()
    finally:
        _rr.time.sleep = _orig_sleep

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

    # Speed presets (0–100, all clamped to _MAX_SPEED at dispatch time)
    SPEED_SLOW = 35
    SPEED_NORMAL = 55
    SPEED_FAST = 62

    def __init__(self) -> None:
        global _instance

        logger.info("controller: initialising Picrawler (one leg at a time to limit inrush current)")
        self._crawler = _make_picrawler()
        # Single worker = all servo calls are strictly serialised.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crawler")
        self._lock = asyncio.Lock()
        # True while locomotion is in progress; cleared by sit/stand so we know
        # to normalise leg position before lowering the robot.
        self._was_moving: bool = False
        _instance = self

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_speed(speed: int) -> int:
        return max(1, min(int(speed), _MAX_SPEED))

    async def _run(self, fn, *args) -> None:
        """Dispatch a blocking Picrawler call to the single-worker executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, fn, *args)
        # Let servos settle before the next command can be dispatched.
        await asyncio.sleep(_SETTLE_S)

    # ------------------------------------------------------------------
    # Posture
    # ------------------------------------------------------------------

    async def stand(self, speed: int = SPEED_SLOW) -> None:
        async with self._lock:
            logger.debug("controller: stand")
            self._was_moving = False
            await self._run(self._crawler.do_step, "stand", self._clamp_speed(speed))

    async def sit(self, speed: int = SPEED_SLOW) -> None:
        """Lower the robot to its resting position. Use as 'stop'."""
        async with self._lock:
            logger.debug("controller: sit (was_moving=%s)", self._was_moving)
            # Normalise leg position before lowering to avoid awkward fold angles
            # that can stall servos and trip the overvoltage protection.
            if self._was_moving:
                await self._run(self._crawler.do_step, "stand", self._clamp_speed(self.SPEED_SLOW))
                self._was_moving = False
            await self._run(self._crawler.do_step, "sit", self._clamp_speed(speed))

    # ------------------------------------------------------------------
    # Locomotion (one gait cycle per call)
    # ------------------------------------------------------------------

    async def forward(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            self._was_moving = True
            await self._run(self._crawler.do_action, "forward", steps, self._clamp_speed(speed))

    async def backward(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            self._was_moving = True
            await self._run(self._crawler.do_action, "backward", steps, self._clamp_speed(speed))

    async def turn_left(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            self._was_moving = True
            await self._run(self._crawler.do_action, "turn left", steps, self._clamp_speed(speed))

    async def turn_right(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            self._was_moving = True
            await self._run(self._crawler.do_action, "turn right", steps, self._clamp_speed(speed))

    async def turn_left_angle(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            self._was_moving = True
            await self._run(self._crawler.do_action, "turn left angle", steps, self._clamp_speed(speed))

    async def turn_right_angle(self, speed: int = SPEED_NORMAL, steps: int = 1) -> None:
        async with self._lock:
            self._was_moving = True
            await self._run(self._crawler.do_action, "turn right angle", steps, self._clamp_speed(speed))

    # ------------------------------------------------------------------
    # Head / expression
    # ------------------------------------------------------------------

    async def look_left(self, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "look left", 1, self._clamp_speed(speed))

    async def look_right(self, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "look right", 1, self._clamp_speed(speed))

    async def look_up(self, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "look up", 1, self._clamp_speed(speed))

    async def look_down(self, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "look down", 1, self._clamp_speed(speed))

    async def wave(self, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, "wave", 1, self._clamp_speed(speed))

    # ------------------------------------------------------------------
    # Generic escape hatch for actions not wrapped above
    # ------------------------------------------------------------------

    async def do_action(self, name: str, steps: int = 1, speed: int = SPEED_NORMAL) -> None:
        async with self._lock:
            await self._run(self._crawler.do_action, name, steps, self._clamp_speed(speed))

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Sit down then shut down the executor cleanly."""
        await self.sit(speed=self.SPEED_SLOW)
        self._executor.shutdown(wait=True)
        logger.info("controller: shut down")
