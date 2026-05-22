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
            "description": (
                "Walk the robot forward or backward. "
                "Use for: go, walk, move, advance, come, approach, proceed, "
                "go back, back up, reverse, retreat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward"],
                        "description": "'forward' to walk ahead, 'backward' to walk back.",
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
            "description": (
                "Rotate the robot left or right on the spot. "
                "Use for: turn, spin, rotate, face left/right, swing left/right. "
                "Set wide=true for a larger angle turn."
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
                        "description": "True for a wider angle turn (e.g. 'spin around', 'turn all the way').",
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
            "description": (
                "Aim the robot's head and camera in a direction. "
                "Use for: look, peek, face, glance, check, see, watch. "
                "Examples: 'look left', 'look up', 'check what's behind', 'look down'."
            ),
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
            "description": (
                "Make the robot wave a leg as a friendly greeting. "
                "Use for: wave, say hello, greet, hi, hey there."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": (
                "Stop all movement immediately and sit the robot down. "
                "Use for: stop, halt, freeze, stay, don't move, sit down, stand still."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_patrol",
            "description": (
                "Start autonomous patrol mode: the robot wanders, turns, and explores on its own. "
                "Use for: patrol, wander, explore, roam, go for a walk, look around, autonomous mode."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_task",
            "description": (
                "Start a multi-step autonomous task that requires planning and movement. "
                "Use for: find X, search for X, look for X, locate X, fetch X, bring X, "
                "explore the room, navigate to X, check what's over there, describe your surroundings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "A clear one-sentence description of the task goal.",
                    },
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enroll_face",
            "description": (
                "Save the face of the person in front of the camera with a name. "
                "Use for: remember me, my name is X / remember this person, their name is X / "
                "learn my face / I am X / call me X."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The person's name to associate with their face.",
                    },
                },
                "required": ["name"],
            },
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
        # Some small models wrap args as {"function": "...", "parameters": {...}}
        if "parameters" in args and isinstance(args["parameters"], dict):
            args = args["parameters"]

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
