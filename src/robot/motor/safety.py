from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..core.bus import MessageBus
from ..core.events import ObstacleCleared, ObstacleDetected
from ..core.state import StateManager

logger = logging.getLogger(__name__)

# Ultrasonic pin names on the Robot HAT (from SunFounder's own examples).
_TRIG_PIN = "D2"
_ECHO_PIN = "D3"

# Distance thresholds.  Hysteresis prevents rapid toggling near the boundary.
OBSTACLE_THRESHOLD_CM = 20.0   # path blocked when distance drops to this
CLEAR_THRESHOLD_CM = 30.0      # path clear again only when distance rises to this

# Sensor tuning
_SONAR_SAMPLES = 3             # median over this many raw readings
_SONAR_SAMPLE_GAP_S = 0.02     # pause between individual samples
_SONAR_READ_TIMEOUT_S = 0.5    # asyncio timeout for one raw read
_LOOP_INTERVAL_S = 0.10        # how often the main loop fires (~10 Hz)


class SafetyLoop:
    """
    Reads the ultrasonic sensor at ~10 Hz and publishes ObstacleDetected /
    ObstacleCleared events on the bus when the path state changes.

    The loop uses hysteresis so a single noisy reading cannot re-trigger the
    arbiter.  Read failures (sensor timeout, -1 return) are counted; if we
    get too many in a row we treat the path as blocked (fail-safe).

    This class never directly commands the motors — it only publishes events.
    The arbiter reacts by cancelling the current behaviour and blocking until
    ObstacleCleared arrives.
    """

    _MAX_CONSECUTIVE_FAILURES = 5  # treat path as blocked after this many bad reads

    def __init__(self, bus: MessageBus, state: StateManager, sonar=None) -> None:
        self._bus = bus
        self._state = state
        self._blocked: bool = False
        self._failure_count: int = 0
        self._last_distance: Optional[float] = None

        if sonar is not None:
            self._sonar = sonar
        else:
            from robot_hat import Ultrasonic, Pin  # type: ignore[import]
            self._sonar = Ultrasonic(Pin(_TRIG_PIN), Pin(_ECHO_PIN))

    @property
    def last_distance(self) -> Optional[float]:
        """Most recent valid sonar reading. Updated at ~10 Hz by the safety loop."""
        return self._last_distance

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Long-lived task.  Start with asyncio.create_task(safety.run())."""
        logger.info("safety: sensor loop started (obstacle=%.0fcm, clear=%.0fcm)",
                    OBSTACLE_THRESHOLD_CM, CLEAR_THRESHOLD_CM)
        while True:
            distance = await self._read_filtered()
            await self._evaluate(distance)
            await asyncio.sleep(_LOOP_INTERVAL_S)

    # ------------------------------------------------------------------
    # Sensor reading
    # ------------------------------------------------------------------

    async def _read_once(self) -> Optional[float]:
        """Single raw read with asyncio timeout.  Returns None on failure."""
        try:
            value = await asyncio.wait_for(
                asyncio.to_thread(self._sonar.read, 1),
                timeout=_SONAR_READ_TIMEOUT_S,
            )
            if value is not None and value > 0:
                return float(value)
            return None
        except asyncio.TimeoutError:
            logger.debug("safety: sonar read timed out")
            return None
        except Exception:
            logger.debug("safety: sonar read exception", exc_info=True)
            return None

    async def _read_filtered(self) -> Optional[float]:
        """
        Take _SONAR_SAMPLES readings and return the median.
        Returns None if not enough valid samples were obtained.
        Increments/resets the failure counter accordingly.
        """
        readings: list[float] = []
        for _ in range(_SONAR_SAMPLES):
            v = await self._read_once()
            if v is not None:
                readings.append(v)
            await asyncio.sleep(_SONAR_SAMPLE_GAP_S)

        if not readings:
            self._failure_count += 1
            logger.warning("safety: no valid sonar readings (%d consecutive failures)",
                           self._failure_count)
            return None

        self._failure_count = 0
        readings.sort()
        dist = readings[len(readings) // 2]  # median
        self._last_distance = dist
        return dist

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    async def _evaluate(self, distance: Optional[float]) -> None:
        """Update obstacle state and publish events when it changes."""

        if distance is None:
            # Treat prolonged sensor failure as blocked (fail-safe).
            if self._failure_count >= self._MAX_CONSECUTIVE_FAILURES and not self._blocked:
                logger.warning("safety: declaring obstacle due to repeated sensor failure")
                await self._set_blocked(distance_cm=0.0)
            return

        if not self._blocked and distance <= OBSTACLE_THRESHOLD_CM:
            logger.warning("safety: obstacle detected at %.1f cm", distance)
            await self._set_blocked(distance_cm=distance)

        elif self._blocked and distance >= CLEAR_THRESHOLD_CM:
            logger.info("safety: path clear at %.1f cm", distance)
            await self._set_clear(distance_cm=distance)

    async def _set_blocked(self, distance_cm: float) -> None:
        self._blocked = True
        event = ObstacleDetected(distance_cm=distance_cm)
        await self._state.update(event)
        self._bus.publish(event)

    async def _set_clear(self, distance_cm: float) -> None:
        self._blocked = False
        self._failure_count = 0
        event = ObstacleCleared(distance_cm=distance_cm)
        await self._state.update(event)
        self._bus.publish(event)
