from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Optional

from ..behaviors.base import Behavior
from ..behaviors.idle_patrol import IdlePatrol
from ..brain.tools import TOOL_SCHEMAS, build_actions
from ..core.events import FaceEnrollRequest, SpeakRequest, Transcript
from ..brain.arbiter import PRIORITY_PATROL, PRIORITY_VOICE, PRIORITY_NAVIGATE

if TYPE_CHECKING:
    from ..brain.arbiter import Arbiter
    from ..brain.llm_client import OllamaClient
    from ..brain.nav_memory import NavMemory
    from ..brain.planner import Planner
    from ..core.bus import MessageBus
    from ..core.state import StateManager
    from ..motor.controller import CrawlerController
    from ..motor.safety import SafetyLoop

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are PiCrawler, a friendly 4-legged spider robot with a voice interface.
You can both control the robot body AND hold a general conversation.

Rules:
- Keep all spoken replies to 1–3 short sentences — you are speaking aloud.
- For movement requests, always call the appropriate tool AND give a short spoken reply.
- For general questions (time, facts, jokes, weather, opinions, etc.) just answer conversationally — no tool call needed.
- The robot state JSON includes `current_time` and `current_date` — use them when asked.
- Be friendly, curious, and a little playful.
- If a question is outside your knowledge, say so briefly rather than guessing.

Movement: forward, backward, turn left/right, look in four directions, wave, stop.
Use start_patrol when asked to patrol, explore, wander, or roam.
Any movement command (including stop) cancels patrol and returns the robot to standby.

The robot has a camera that recognises people.
`people_visible` lists names of people currently in view.
- When asked "who is this?" or "do you know me?", describe who you can see.
- When asked to remember someone ("remember me, my name is X"), call enroll_face.\
"""


_COMMAND_KEYWORDS = frozenset({
    # movement
    "forward", "ahead", "advance", "proceed", "come", "approach",
    "backward", "back", "reverse", "retreat",
    "walk", "move", "go", "run",
    # turning
    "turn", "rotate", "spin", "face", "swing",
    "left", "right",
    # looking
    "look", "peek", "glance", "check", "watch", "see",
    "up", "down",
    # gestures
    "wave", "hello", "greet", "hi",
    # stopping
    "stop", "halt", "freeze", "stay", "sit",
    # patrol
    "patrol", "wander", "explore", "roam", "autonomous",
    # face
    "remember", "enroll", "learn", "name",
    # tasks (Phase 2)
    "find", "search", "locate", "fetch", "bring", "navigate",
})


def _tools_for(text: str) -> list:
    """Return tool schemas only when the transcript contains a command keyword.
    For pure questions/conversation the small local model reliably answers in
    text when no tools are offered — and reliably calls a random tool when they are."""
    words = set(text.lower().split())
    return TOOL_SCHEMAS if words & _COMMAND_KEYWORDS else []


class _VoiceAction(Behavior):
    """
    One-shot behavior that executes a list of controller actions in order.
    Created by the Orchestrator and handed to the Arbiter at PRIORITY_VOICE.
    """

    def __init__(
        self,
        bus: "MessageBus",
        state: "StateManager",
        actions: list[Callable],
        ctrl: "CrawlerController",
    ) -> None:
        super().__init__(bus, state)
        self._actions = actions
        self._ctrl = ctrl

    async def _run(self) -> None:
        for action in self._actions:
            await action()

    async def _cleanup(self) -> None:
        await self._ctrl.sit(speed=50)


class Orchestrator:
    """
    Subscribes to Transcript events, calls the LLM, dispatches a
    _VoiceAction to the Arbiter for any tool calls, and publishes a
    SpeakRequest for the spoken reply.

    One LLM call is in flight at a time; new transcripts received during
    processing are queued by the bus (bounded drop-oldest).
    """

    def __init__(
        self,
        bus: "MessageBus",
        state: "StateManager",
        arbiter: "Arbiter",
        llm: "OllamaClient",
        ctrl: "CrawlerController",
        *,
        planner: "Optional[Planner]" = None,
        memory: "Optional[NavMemory]" = None,
        safety: "Optional[SafetyLoop]" = None,
    ) -> None:
        self._bus = bus
        self._state = state
        self._arbiter = arbiter
        self._llm = llm
        self._ctrl = ctrl
        self._planner = planner
        self._memory = memory
        self._safety = safety

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Long-lived task.  Cancel to stop."""
        logger.info("orchestrator: ready")
        async with self._bus.subscribe(Transcript.topic) as q:
            while True:
                event: Transcript = await q.get()
                logger.info("orchestrator: received %r", event.text)
                try:
                    await self._handle(event)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("orchestrator: error handling transcript")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _handle(self, event: Transcript) -> None:
        state_json = await self._state.snapshot()

        messages = [
            {
                "role": "user",
                "content": (
                    f"[robot state: {state_json}]\n\nUser said: {event.text}"
                ),
            }
        ]

        try:
            msg = await self._llm.chat(messages, tools=_tools_for(event.text))
        except Exception:
            logger.exception("orchestrator: LLM call failed")
            self._bus.publish(SpeakRequest(text="Sorry, I couldn't reach my brain right now."))
            return

        # Execute any tool calls.
        tool_calls = msg.get("tool_calls") or []
        motor_calls = []
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            args: dict = fn.get("arguments") or {}
            if "parameters" in args and isinstance(args["parameters"], dict):
                args = args["parameters"]
            if name == "enroll_face":
                person = args.get("name", "").strip()
                if person:
                    self._bus.publish(FaceEnrollRequest(name=person))
            elif name == "start_patrol":
                patrol = IdlePatrol(self._bus, self._state, initial_turn=True)
                self._arbiter.request(patrol, PRIORITY_PATROL, reason="voice: patrol")
            elif name == "start_task":
                goal = args.get("goal", "").strip()
                if goal and self._planner and self._memory and self._safety:
                    from ..behaviors.task_behavior import TaskBehavior
                    task = TaskBehavior(
                        self._bus, self._state,
                        goal=goal,
                        planner=self._planner,
                        memory=self._memory,
                        safety=self._safety,
                        controller=self._ctrl,
                    )
                    self._arbiter.request(task, PRIORITY_NAVIGATE, reason=f"task: {goal[:40]}")
                else:
                    logger.warning("orchestrator: start_task called but planner/memory/safety not wired up")
                    self._bus.publish(SpeakRequest(text="Sorry, I can't run tasks right now."))
            else:
                motor_calls.append(call)

        if motor_calls:
            actions = build_actions(self._ctrl, motor_calls)
            if actions:
                behavior = _VoiceAction(self._bus, self._state, actions, self._ctrl)
                self._arbiter.request(behavior, PRIORITY_VOICE, reason=event.text[:40])

        # Speak the LLM's text reply (if any).
        reply = (msg.get("content") or "").strip()
        if reply:
            self._bus.publish(SpeakRequest(text=reply))
