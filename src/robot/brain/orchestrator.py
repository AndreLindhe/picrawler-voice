from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable

from ..behaviors.base import Behavior
from ..brain.tools import TOOL_SCHEMAS, build_actions
from ..core.events import SpeakRequest, Transcript
from ..brain.arbiter import PRIORITY_VOICE

if TYPE_CHECKING:
    from ..brain.arbiter import Arbiter
    from ..brain.llm_client import OllamaClient
    from ..core.bus import MessageBus
    from ..core.state import StateManager
    from ..motor.controller import CrawlerController

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the brain of a 4-legged spider robot called PiCrawler.
You receive voice commands and respond by calling tools to move the robot and \
by giving a short spoken reply.

Rules:
- Keep spoken replies to 1–2 short sentences.
- Always call at least one tool when the user asks for movement.
- If the command is unclear, ask for clarification instead of guessing.
- Be friendly and a little playful.

The robot can move forward/backward, turn left/right, look in four directions,
wave, and stop.  It explores autonomously when idle.\
"""


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
    ) -> None:
        self._bus = bus
        self._state = state
        self._arbiter = arbiter
        self._llm = llm
        self._ctrl = ctrl

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
            msg = await self._llm.chat(messages, tools=TOOL_SCHEMAS)
        except Exception:
            logger.exception("orchestrator: LLM call failed")
            self._bus.publish(SpeakRequest(text="Sorry, I couldn't reach my brain right now."))
            return

        # Execute any tool calls as a voice action.
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            actions = build_actions(self._ctrl, tool_calls)
            if actions:
                behavior = _VoiceAction(self._bus, self._state, actions, self._ctrl)
                self._arbiter.request(behavior, PRIORITY_VOICE, reason=event.text[:40])

        # Speak the LLM's text reply (if any).
        reply = (msg.get("content") or "").strip()
        if reply:
            self._bus.publish(SpeakRequest(text=reply))
