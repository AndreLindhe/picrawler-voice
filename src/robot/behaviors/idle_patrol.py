from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

from .base import Behavior
from ..core.bus import MessageBus
from ..core.state import StateManager
from ..motor.controller import CrawlerController, get_controller

logger = logging.getLogger(__name__)


class IdlePatrol(Behavior):
    """
    Default autonomous behavior: explore by walking forward and turning.

    Walking pattern
    ---------------
    stand → [forward × 1..N] → random turn → repeat

    After each STEPS_BEFORE_TURN forward steps there is a TURN_PROBABILITY
    chance of turning.  Both thresholds are intentionally low so the robot
    visibly wanders rather than marching in a straight line forever.

    `initial_turn=True` (the default) makes the robot turn once before its
    first forward step.  This is important after an obstacle event: the safety
    loop clears at 30 cm, but the obstruction is usually still straight ahead.
    Turning first avoids immediately walking back into it.

    Cleanup
    -------
    `_cleanup()` calls sit(), so any preemption — obstacle, voice command,
    shutdown — leaves the robot in a stable resting position.

    Controller injection
    --------------------
    Pass `controller=` in tests to inject a mock without touching hardware.
    Production code passes nothing; the singleton is resolved at runtime.
    """

    WALK_SPEED: int = 70
    POSE_SPEED: int = 50

    # After this many forward steps, consider turning.
    STEPS_BEFORE_TURN: int = 3
    # Probability of actually turning when the threshold is reached.
    TURN_PROBABILITY: float = 0.45

    _TURN_ACTIONS: tuple[str, ...] = (
        "turn left",
        "turn right",
        "turn left angle",
        "turn right angle",
    )

    def __init__(
        self,
        bus: MessageBus,
        state: StateManager,
        controller: Optional[CrawlerController] = None,
        initial_turn: bool = True,
    ) -> None:
        super().__init__(bus, state)
        self._ctrl = controller
        self._initial_turn = initial_turn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ctrl(self) -> CrawlerController:
        return self._ctrl if self._ctrl is not None else get_controller()

    async def _turn(self, ctrl: CrawlerController) -> None:
        action = random.choice(self._TURN_ACTIONS)
        logger.debug("patrol: %s", action)
        await ctrl.do_action(action, steps=1, speed=self.WALK_SPEED)

    # ------------------------------------------------------------------
    # Behavior contract
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        ctrl = self._get_ctrl()

        await ctrl.stand(speed=self.POSE_SPEED)

        if self._initial_turn:
            await self._turn(ctrl)

        steps_forward = 0

        while True:
            if (
                steps_forward >= self.STEPS_BEFORE_TURN
                and random.random() < self.TURN_PROBABILITY
            ):
                await self._turn(ctrl)
                steps_forward = 0
            else:
                await ctrl.forward(speed=self.WALK_SPEED, steps=1)
                steps_forward += 1

    async def _cleanup(self) -> None:
        await self._get_ctrl().sit(speed=self.POSE_SPEED)
