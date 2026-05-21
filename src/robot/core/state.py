from __future__ import annotations

import asyncio
import dataclasses
import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .events import (
    BehaviorEnded,
    BehaviorStarted,
    ObstacleCleared,
    ObstacleDetected,
    PerceptionUpdate,
    PersonDetected,
    StrangerDetected,
    Transcript,
    WakeWordDetected,
)


@dataclass
class _WorldState:
    # Motor / safety
    obstacle_present: bool = False
    obstacle_distance_cm: Optional[float] = None

    # Arbiter
    current_behavior: str = "none"
    behavior_priority: int = 0

    # Perception — list of {label, confidence, x, y, w, h}
    detections: List[Dict[str, Any]] = field(default_factory=list)

    # Perception — people currently visible in camera frame
    people_visible: List[str] = field(default_factory=list)

    # Voice
    last_transcript: str = ""
    last_wake_word_at: float = 0.0

    # Meta
    updated_at: float = field(default_factory=time.monotonic)


class StateManager:
    """
    Single source of truth for the robot's current world view.

    Events are what happened; state is what is true *now*.
    Call `update(event)` from any coroutine to advance state.
    Call `snapshot()` to get a JSON-serialisable dict suitable for feeding
    to the LLM as context.
    """

    def __init__(self) -> None:
        self._state = _WorldState()
        self._lock = asyncio.Lock()

    async def update(self, event: Any) -> None:
        async with self._lock:
            s = self._state
            match event:
                case ObstacleDetected(distance_cm=d):
                    s.obstacle_present = True
                    s.obstacle_distance_cm = d
                case ObstacleCleared(distance_cm=d):
                    s.obstacle_present = False
                    s.obstacle_distance_cm = d
                case PerceptionUpdate(detections=dets):
                    s.detections = [
                        {
                            "label": d.label,
                            "confidence": round(d.confidence, 3),
                            "bbox": {"x": d.x, "y": d.y, "w": d.w, "h": d.h},
                        }
                        for d in dets
                    ]
                case BehaviorStarted(name=name, priority=p):
                    s.current_behavior = name
                    s.behavior_priority = p
                case BehaviorEnded():
                    s.current_behavior = "none"
                    s.behavior_priority = 0
                case WakeWordDetected(timestamp=t):
                    s.last_wake_word_at = t
                case Transcript(text=text, is_final=True):
                    s.last_transcript = text
                case PersonDetected(name=name):
                    if name and name not in s.people_visible:
                        s.people_visible = [*s.people_visible, name]
                case StrangerDetected():
                    pass  # Handled by presence module directly
            s.updated_at = time.monotonic()

    async def update_people_visible(self, names: List[str]) -> None:
        """Called by PresenceDetector each frame to refresh the visible-people list."""
        async with self._lock:
            self._state.people_visible = names
            self._state.updated_at = time.monotonic()

    async def snapshot(self) -> Dict[str, Any]:
        """JSON-serialisable dict. Feed this to the LLM as robot context."""
        async with self._lock:
            d = dataclasses.asdict(self._state)
            d["current_time"] = datetime.datetime.now().strftime("%H:%M")
            d["current_date"] = datetime.datetime.now().strftime("%A, %d %B %Y")
            return d
