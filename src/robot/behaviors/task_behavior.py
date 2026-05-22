from __future__ import annotations

"""
TaskBehavior — executes a single voice-requested task via the LLM planner.

Runs at PRIORITY_NAVIGATE (2) so it preempts SmartPatrol but can itself
be preempted by a higher-priority voice command in a future extension.

Phase 1: sonar + movement only.
Phase 2: add vision tools (look_around, find_object) to the executor and
         extend the system prompt — no changes needed here.
"""

import asyncio
import logging
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

_TASK_SYSTEM_PROMPT = """\
You are PiCrawler, a 4-legged spider robot executing a task requested by the user.
Available senses: front sonar distance (read_sonar).
Available actions: move forward/backward, turn left/right, speak.

Rules:
- Start by calling speak() with a one-sentence plan.
- Use read_sonar() regularly to avoid obstacles.
- Narrate your observations and progress aloud as you go.
- Keep all spoken text SHORT — one sentence, plain English.
- When the task is done (or you have done your best), call done().
- If the task is impossible with current sensors, say so and call done(success=false).\
"""


class TaskBehavior(Behavior):
    """
    One-shot behavior that runs the LLM planner for a voice-requested task.

    Created by the Orchestrator when the user says something like
    "find a red ball" or "explore the room". The arbiter runs it at
    PRIORITY_NAVIGATE so SmartPatrol is preempted for the duration.
    """

    def __init__(
        self,
        bus: "MessageBus",
        state: "StateManager",
        goal: str,
        planner: Planner,
        memory: NavMemory,
        safety: "SafetyLoop",
        controller: Optional[CrawlerController] = None,
    ) -> None:
        super().__init__(bus, state)
        self._goal = goal
        self._planner = planner
        self._memory = memory
        self._safety = safety
        self._ctrl = controller

    # ------------------------------------------------------------------
    # Behavior contract
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        ctrl = self._ctrl or get_controller()
        scene = SceneSnapshot(sonar_cm=self._safety.last_distance)

        def _speak(text: str) -> None:
            self.bus.publish(SpeakRequest(text=text))

        executor = build_nav_executor(
            ctrl=ctrl,
            get_sonar_fn=lambda: self._safety.last_distance,
            speak_fn=_speak,
        )

        logger.info("task_behavior: starting task %r", self._goal)

        try:
            result = await self._planner.run(
                goal=self._goal,
                scene=scene,
                system_prompt=_TASK_SYSTEM_PROMPT,
                tool_schemas=NAV_TOOL_SCHEMAS,
                executor=executor,
                max_steps=15,
            )
        except asyncio.CancelledError:
            raise

        logger.info(
            "task_behavior: done — success=%s steps=%d summary=%r",
            result.success,
            result.steps_taken,
            result.summary,
        )

        self._memory.save_episode(
            situation=scene.to_text(),
            sonar_range=scene.sonar_range(),
            goal=self._goal,
            actions=result.actions,
            success=result.success,
            summary=result.summary,
        )

    async def _cleanup(self) -> None:
        ctrl = self._ctrl or get_controller()
        await ctrl.sit(speed=50)
