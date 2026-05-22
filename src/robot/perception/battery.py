from __future__ import annotations

"""
Battery monitor — reads the Robot HAT ADC and periodically logs voltage.

The Robot HAT uses a 3:1 voltage divider on pin A4, so:
    actual_voltage = adc_voltage * 3

Thresholds for a 2-cell 18650 pack (7.4V nominal, 8.4V full, 6.0V cutoff):
    >= 7.2V  — healthy
    >= 6.8V  — low    (warn once, then every 5 min)
    <  6.8V  — critical (warn + repeat every 2 min)
"""

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.bus import MessageBus
    from ..core.state import StateManager

from ..core.events import BatteryCritical, BatteryLow, SpeakRequest

logger = logging.getLogger(__name__)

_ADC_PIN = "A4"
_VOLTAGE_DIVIDER = 3.0

_VOLT_FULL = 8.4
_VOLT_LOW = 6.8        # warn
_VOLT_CRITICAL = 6.4   # urgent warn + repeat
_VOLT_DEAD = 6.0       # 0 %

_LOG_INTERVAL_S = 300        # log voltage every 5 min
_WARN_INTERVAL_LOW_S = 300   # repeat low warning every 5 min
_WARN_INTERVAL_CRIT_S = 120  # repeat critical warning every 2 min


def _percent(volts: float) -> int:
    pct = (volts - _VOLT_DEAD) / (_VOLT_FULL - _VOLT_DEAD) * 100
    return max(0, min(100, int(pct)))


class BatteryMonitor:
    """
    Long-lived async task that reads battery voltage from ADC A4 and:
      - Logs it at INFO level every 5 minutes
      - Publishes BatteryLow / BatteryCritical events when thresholds are crossed
      - Speaks a spoken warning so the user knows to charge
    """

    def __init__(self, bus: "MessageBus", state: "StateManager") -> None:
        self._bus = bus
        self._state = state
        self._adc = None
        self._last_log: float = 0.0
        self._last_warn: float = 0.0

    def _read_volts(self) -> float | None:
        try:
            if self._adc is None:
                from robot_hat import ADC
                self._adc = ADC(_ADC_PIN)
            v = self._adc.read_voltage()
            return float(v) * _VOLTAGE_DIVIDER
        except Exception:
            logger.debug("battery: ADC read failed", exc_info=True)
            return None

    async def run(self) -> None:
        import time
        logger.info("battery: monitor started (pin=%s, divider=%.0fx)", _ADC_PIN, _VOLTAGE_DIVIDER)

        # Log immediately at startup so battery state is visible in the journal.
        await self._check(time.monotonic(), force_log=True)

        while True:
            await asyncio.sleep(30)
            await self._check(time.monotonic())

    async def _check(self, now: float, force_log: bool = False) -> None:
        import time
        volts = await asyncio.to_thread(self._read_volts)
        if volts is None:
            return

        pct = _percent(volts)

        if force_log or (now - self._last_log >= _LOG_INTERVAL_S):
            logger.info("battery: %.2f V  %d%%", volts, pct)
            self._last_log = now

        if volts < _VOLT_CRITICAL:
            interval = _WARN_INTERVAL_CRIT_S
            if force_log or (now - self._last_warn >= interval):
                logger.warning("battery: CRITICAL %.2f V (%d%%) — charge immediately", volts, pct)
                self._bus.publish(BatteryCritical(voltage=volts, percent=pct))
                self._bus.publish(SpeakRequest(
                    text=f"Battery critical, {pct} percent. Please charge me now."
                ))
                self._last_warn = now

        elif volts < _VOLT_LOW:
            interval = _WARN_INTERVAL_LOW_S
            if force_log or (now - self._last_warn >= interval):
                logger.warning("battery: LOW %.2f V (%d%%) — consider charging soon", volts, pct)
                self._bus.publish(BatteryLow(voltage=volts, percent=pct))
                self._bus.publish(SpeakRequest(
                    text=f"Battery low, {pct} percent. Consider charging soon."
                ))
                self._last_warn = now
