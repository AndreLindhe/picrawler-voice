from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine, TYPE_CHECKING

if TYPE_CHECKING:
    from ..motor.controller import CrawlerController

logger = logging.getLogger(__name__)

# OpenAI-compatible tool schemas sent to the LLM.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the robot forward or backward.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward"],
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of gait cycles (1–10, default 3).",
                        "default": 3,
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
            "description": "Turn the robot left or right in place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right"],
                    },
                    "wide": {
                        "type": "boolean",
                        "description": "True for a wider angle turn.",
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
            "name": "look",
            "description": "Aim the robot's head/camera in a direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right", "up", "down"],
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wave",
            "description": "Make the robot wave as a friendly gesture.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "Stop all movement and sit the robot down.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def build_actions(
    ctrl: "CrawlerController",
    tool_calls: list[dict[str, Any]],
) -> list[Callable[[], Coroutine]]:
    """
    Convert a list of Ollama tool_call dicts into a list of zero-argument
    async callables ready for VoiceAction to execute sequentially.

    Unknown tool names are logged and skipped.
    """
    actions: list[Callable[[], Coroutine]] = []

    for call in tool_calls:
        fn = call.get("function", {})
        name = fn.get("name", "")
        args: dict[str, Any] = fn.get("arguments", {}) or {}

        coro = _build_one(ctrl, name, args)
        if coro is not None:
            actions.append(coro)
        else:
            logger.warning("tools: unknown tool %r — skipped", name)

    return actions


# ------------------------------------------------------------------
# Private
# ------------------------------------------------------------------

def _build_one(
    ctrl: "CrawlerController",
    name: str,
    args: dict[str, Any],
) -> Callable[[], Coroutine] | None:
    speed = 70

    if name == "move":
        direction = args.get("direction", "forward")
        steps = int(args.get("steps", 3))
        steps = max(1, min(steps, 10))
        if direction == "forward":
            return lambda: ctrl.forward(speed=speed, steps=steps)
        else:
            return lambda: ctrl.backward(speed=speed, steps=steps)

    if name == "turn":
        direction = args.get("direction", "left")
        wide = bool(args.get("wide", False))
        if direction == "left":
            fn = ctrl.turn_left_angle if wide else ctrl.turn_left
        else:
            fn = ctrl.turn_right_angle if wide else ctrl.turn_right
        return lambda: fn(speed=speed, steps=2)

    if name == "look":
        direction = args.get("direction", "left")
        mapping = {
            "left": ctrl.look_left,
            "right": ctrl.look_right,
            "up": ctrl.look_up,
            "down": ctrl.look_down,
        }
        fn = mapping.get(direction)
        if fn:
            return lambda: fn(speed=speed)

    if name == "wave":
        return lambda: ctrl.wave(speed=speed)

    if name == "stop":
        return lambda: ctrl.sit(speed=50)

    return None
