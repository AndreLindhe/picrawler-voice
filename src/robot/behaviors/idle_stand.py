from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .base import Behavior
from ..core.bus import MessageBus
from ..core.state import StateManager
from ..motor.controller import CrawlerController, get_controller

logger = logging.getLogger(__name__)


class IdleStand(Behavior):
    """
    Default resting behavior: stand up and wait for a voice command.
    The robot stays still until preempted by a higher-priority request.
    """

    POSE_SPEED: int = 40

    def __init__(
        self,
        bus: MessageBus,
        state: StateManager,
        controller: Optional[CrawlerController] = None,
    ) -> None:
        super().__init__(bus, state)
        self._ctrl = controller

    def _get_ctrl(self) -> CrawlerController:
        return self._ctrl if self._ctrl is not None else get_controller()

    async def _run(self) -> None:
        await self._get_ctrl().stand(speed=self.POSE_SPEED)
        logger.info("idle_stand: standing by")
        await asyncio.sleep(float("inf"))

    async def _cleanup(self) -> None:
        pass
