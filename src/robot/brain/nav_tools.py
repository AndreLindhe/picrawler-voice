from __future__ import annotations

"""
Tool schemas and executor builder for the navigation planner.

Kept separate from brain/tools.py (voice tools) so each tool set can evolve
independently.  The planner receives whichever set the caller passes in —
voice tasks in Phase 2 will define their own tool sets the same way.
"""

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..motor.controller import CrawlerController

# ------------------------------------------------------------------
# OpenAI-compatible schemas sent to the LLM
# ------------------------------------------------------------------

NAV_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Walk the robot forward or backward N gait steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward"],
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of gait cycles (1–6). Default 2.",
                        "default": 2,
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn",
            "description": (
                "Rotate the robot left or right. "
                "Use wide=true for a larger angle (~90°), "
                "false for a smaller nudge (~45°)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right"],
                    },
                    "wide": {
                        "type": "boolean",
                        "description": "True for ~90° turn, false for ~45°.",
                        "default": False,
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_sonar",
            "description": (
                "Read the current front sonar distance. "
                "Returns sonar_cm and a status: "
                "'clear' (>45 cm), 'approaching' (20–45 cm), 'obstacle' (<=20 cm)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": (
                "Speak a short sentence out loud. "
                "Always call this BEFORE taking an action to narrate your plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "One sentence, max ~15 words."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Signal that navigation is complete. "
                "Call with success=true when sonar reads >40 cm. "
                "Call with success=false if you cannot clear the obstacle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "summary": {
                        "type": "string",
                        "description": "One sentence describing what happened.",
                    },
                },
                "required": ["success", "summary"],
            },
        },
    },
]


# ------------------------------------------------------------------
# Executor factory
# ------------------------------------------------------------------

def build_nav_executor(
    ctrl: "CrawlerController",
    get_sonar_fn: Callable[[], float | None],
    speak_fn: Callable[[str], None],
) -> dict[str, Callable]:
    """
    Build a tool_name → async-callable mapping for the navigation planner.

    get_sonar_fn  — zero-arg callable that returns the current sonar distance
                    (or None on sensor failure).
    speak_fn      — zero-arg-ish callable(text) that publishes a SpeakRequest.
    `done` is intentionally absent — handled directly by the planner.
    """

    async def _read_sonar() -> dict:
        dist = get_sonar_fn()
        if dist is None:
            return {"sonar_cm": None, "status": "unknown"}
        if dist <= 20:
            status = "obstacle"
        elif dist <= 45:
            status = "approaching"
        else:
            status = "clear"
        return {"sonar_cm": round(dist, 1), "status": status}

    async def _move(direction: str = "forward", steps: int = 2) -> str:
        steps = max(1, min(int(steps), 6))
        if direction == "forward":
            await ctrl.forward(speed=65, steps=steps)
        else:
            await ctrl.backward(speed=65, steps=steps)
        return f"Moved {direction} {steps} steps."

    async def _turn(direction: str = "left", wide: bool = False) -> str:
        if direction == "left":
            fn = ctrl.turn_left_angle if wide else ctrl.turn_left
        else:
            fn = ctrl.turn_right_angle if wide else ctrl.turn_right
        await fn(speed=65, steps=2)
        return f"Turned {direction} ({'wide' if wide else 'normal'})."

    async def _speak(text: str = "") -> None:
        if text:
            speak_fn(text)

    return {
        "move": _move,
        "turn": _turn,
        "read_sonar": _read_sonar,
        "speak": _speak,
    }
