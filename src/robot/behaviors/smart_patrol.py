from __future__ import annotations

"""
SmartPatrol — learning autonomous navigation behavior.

Replaces IdlePatrol as the default wander behavior.  When sonar reads close
it pauses, calls the LLM planner with full context and past memory, executes
the plan step-by-step (narrating aloud), then saves the episode to memory so
the robot improves over time.

Phase 1: sonar only.
Phase 2: scene.objects will be populated by Hailo YOLO — no changes needed here.
"""

import asyncio
import logging
import random
import time
from typing import Optional, TYPE_CHECKING

from .base import Behavior
from ..brain.nav_memory import NavMemory
from ..brain.nav_tools import NAV_TOOL_SCHEMAS, build_nav_executor
from ..brain.planner import Planner
from ..core.events import SpeakRequest
from ..motor.controller import CrawlerController, get_controller
from ..perception.scene import SceneSnapshot

if TYPE_CHECKING:
    from ..core.bus import MessageBus
    from ..core.state import StateManager
    from ..motor.safety import SafetyLoop

logger = logging.getLogger(__name__)

# Start planning when sonar drops to this distance (before the 20 cm hard stop).
_PLAN_AT_CM = 45.0
# Wander tuning (matches old IdlePatrol defaults).
_WALK_SPEED = 70
_POSE_SPEED = 50
_STEPS_BEFORE_TURN = 3
_TURN_PROBABILITY = 0.45
# After a failed plan (LLM unavailable), don't retry for this long.
# Robot runs the obstacle reflex instead so it never stands still indefinitely.
_PLAN_FAIL_COOLDOWN_S = 30.0

_NAV_SYSTEM_PROMPT = """\
You are the navigation brain of PiCrawler, a 4-legged spider robot.
Your job is to navigate around whatever is blocking the robot's path.

Rules:
- ALWAYS call speak() BEFORE each action to narrate your plan in one short sentence.
- Use read_sonar() after moves to check whether the path has cleared.
- A path is CLEAR when sonar reads above 40 cm.
- When the path is clear, call done(success=true, summary=...).
- If you cannot clear the path after several attempts, call done(success=false, summary=...).
- Be efficient — aim to clear in 3 to 6 steps.
- Spoken text must be SHORT — one sentence, plain English, no jargon.
- Learn from past experiences: if a strategy worked before, try it again.\
"""


class SmartPatrol(Behavior):
    """
    Default autonomous behavior with LLM-guided obstacle navigation.

    When sonar is clear this behaves like IdlePatrol (random walk).
    When sonar closes to _PLAN_AT_CM the robot pauses, reasons through the
    obstacle using the LLM planner, executes the resulting action sequence,
    narrates every step aloud, and saves the outcome to memory.
    """

    def __init__(
        self,
        bus: "MessageBus",
        state: "StateManager",
        planner: Planner,
        memory: NavMemory,
        safety: "SafetyLoop",
        controller: Optional[CrawlerController] = None,
    ) -> None:
        super().__init__(bus, state)
        self._planner = planner
        self._memory = memory
        self._safety = safety
        self._ctrl = controller
        self._plan_blocked_until: float = 0.0  # monotonic time — skip planner until then

    # ------------------------------------------------------------------
    # Behavior contract
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        ctrl = self._ctrl or get_controller()
        await ctrl.stand(speed=_POSE_SPEED)

        steps_since_turn = 0

        while True:
            dist = self._safety.last_distance

            if dist is not None and dist <= _PLAN_AT_CM:
                if time.monotonic() >= self._plan_blocked_until:
                    await self._plan_around(ctrl, dist)
                else:
                    # LLM recently unavailable — use simple obstacle reflex instead.
                    await self._obstacle_reflex(ctrl)
                steps_since_turn = 0
            else:
                # Free space — wander randomly like IdlePatrol.
                if (
                    steps_since_turn >= _STEPS_BEFORE_TURN
                    and random.random() < _TURN_PROBABILITY
                ):
                    await self._random_turn(ctrl)
                    steps_since_turn = 0
                else:
                    await ctrl.forward(speed=_WALK_SPEED, steps=1)
                    steps_since_turn += 1

    async def _cleanup(self) -> None:
        ctrl = self._ctrl or get_controller()
        await ctrl.sit(speed=_POSE_SPEED)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    async def _plan_around(self, ctrl: CrawlerController, dist: float) -> None:
        scene = SceneSnapshot(sonar_cm=dist)
        goal = "navigate around the obstacle ahead"

        def _speak(text: str) -> None:
            self.bus.publish(SpeakRequest(text=text))

        executor = build_nav_executor(
            ctrl=ctrl,
            get_sonar_fn=lambda: self._safety.last_distance,
            speak_fn=_speak,
        )

        logger.info("smart_patrol: obstacle at %.0f cm — calling planner", dist)

        try:
            result = await self._planner.run(
                goal=goal,
                scene=scene,
                system_prompt=_NAV_SYSTEM_PROMPT,
                tool_schemas=NAV_TOOL_SCHEMAS,
                executor=executor,
                max_steps=10,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("smart_patrol: planner raised — falling back to obstacle reflex for %.0fs", _PLAN_FAIL_COOLDOWN_S)
            self._plan_blocked_until = time.monotonic() + _PLAN_FAIL_COOLDOWN_S
            await self._obstacle_reflex(ctrl)
            return

        if result.summary == "LLM error":
            logger.warning("smart_patrol: LLM unavailable — obstacle reflex for %.0fs", _PLAN_FAIL_COOLDOWN_S)
            self._plan_blocked_until = time.monotonic() + _PLAN_FAIL_COOLDOWN_S
            await self._obstacle_reflex(ctrl)
            return

        logger.info(
            "smart_patrol: plan complete — success=%s steps=%d summary=%r",
            result.success,
            result.steps_taken,
            result.summary,
        )

        self._memory.save_episode(
            situation=scene.to_text(),
            sonar_range=scene.sonar_range(),
            goal=goal,
            actions=result.actions,
            success=result.success,
            summary=result.summary,
        )

    async def _obstacle_reflex(self, ctrl: CrawlerController) -> None:
        """Dumb fallback when LLM is unavailable: back up, turn, continue."""
        logger.info("smart_patrol: obstacle reflex — backing up and turning")
        await ctrl.backward(steps=2)
        direction = random.choice(["left", "right"])
        if random.random() < 0.5:
            fn = ctrl.turn_left_angle if direction == "left" else ctrl.turn_right_angle
        else:
            fn = ctrl.turn_left if direction == "left" else ctrl.turn_right
        await fn(steps=1)

    # ------------------------------------------------------------------
    # Wander helpers
    # ------------------------------------------------------------------

    async def _random_turn(self, ctrl: CrawlerController) -> None:
        direction = random.choice(["left", "right"])
        wide = random.random() < 0.35
        if direction == "left":
            fn = ctrl.turn_left_angle if wide else ctrl.turn_left
        else:
            fn = ctrl.turn_right_angle if wide else ctrl.turn_right
        await fn(speed=_WALK_SPEED, steps=1)
