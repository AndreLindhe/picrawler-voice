from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.bus import MessageBus
    from ..core.state import StateManager

logger = logging.getLogger(__name__)


class Behavior(ABC):
    """
    Contract for all robot behaviors.

    Subclasses implement `_run()` and optionally `_cleanup()`.

    Cancellation model (preemptive):
    - The arbiter cancels the current task via `task.cancel()`.
    - `_run()` receives `CancelledError`; it should propagate it.
    - `_cleanup()` always runs in a `finally` block, shielded from secondary
      cancellations so motors can be released cleanly.
    - If `_cleanup()` itself raises, a last-ditch motor stop is attempted.
    """

    # Priority constants — match the arbiter's values.
    PRIORITY_PATROL = 0
    PRIORITY_VOICE = 1

    def __init__(self, bus: "MessageBus", state: "StateManager") -> None:
        self.bus = bus
        self.state = state

    @property
    def name(self) -> str:
        return type(self).__name__

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def _run(self) -> None:
        """
        Main behavior loop.  Run until cancelled or naturally complete.
        Let CancelledError propagate — do NOT swallow it.
        """
        ...

    async def _cleanup(self) -> None:
        """
        Called after `_run()` exits for any reason (natural, cancelled, error).
        Stop motors, close file handles, etc.
        Default: no-op.
        """

    # ------------------------------------------------------------------
    # Runtime (called by the arbiter, not overridden)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        try:
            await self._run()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("%s: unhandled exception in _run()", self.name)
            raise
        finally:
            try:
                # Run cleanup synchronously inside the task so callers can
                # rely on it being done when they await the task.
                # If a second cancellation arrives mid-cleanup, shield only
                # the emergency-stop fallback.
                await self._cleanup()
            except asyncio.CancelledError:
                logger.warning(
                    "%s: cleanup cancelled — attempting emergency motor stop", self.name
                )
                try:
                    await asyncio.shield(self._emergency_stop())
                except Exception:
                    logger.exception("%s: emergency stop also failed", self.name)
                raise
            except Exception:
                logger.exception(
                    "%s: _cleanup() raised — attempting emergency motor stop", self.name
                )
                await self._emergency_stop()

    async def _emergency_stop(self) -> None:
        """Last-ditch: sit via the shared controller if available."""
        try:
            from ..motor.controller import get_controller
            await get_controller().sit(speed=40)
        except Exception:
            logger.exception("%s: emergency sit failed", self.name)
