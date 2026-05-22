from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import ClassVar, Tuple


def _now() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class WakeWordDetected:
    topic: ClassVar[str] = "hearing.wake_word"
    confidence: float = 0.0
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class Transcript:
    topic: ClassVar[str] = "hearing.transcript"
    text: str = ""
    is_final: bool = True
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class ObstacleDetected:
    topic: ClassVar[str] = "motor.obstacle_detected"
    distance_cm: float = 0.0
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class ObstacleCleared:
    topic: ClassVar[str] = "motor.obstacle_cleared"
    distance_cm: float = 0.0
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class Detection:
    topic: ClassVar[str] = "perception.detection"
    label: str = ""
    confidence: float = 0.0
    # Normalised bounding box (0.0–1.0 relative to frame)
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class PerceptionUpdate:
    topic: ClassVar[str] = "perception.update"
    detections: Tuple[Detection, ...] = ()
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class BehaviorStarted:
    topic: ClassVar[str] = "arbiter.behavior_started"
    name: str = ""
    priority: int = 0
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class BehaviorEnded:
    topic: ClassVar[str] = "arbiter.behavior_ended"
    name: str = ""
    # "natural" | "cancelled" | "error"
    reason: str = "natural"
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class SpeakRequest:
    topic: ClassVar[str] = "speech.request"
    text: str = ""
    priority: int = 0
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class PersonDetected:
    topic: ClassVar[str] = "perception.person_detected"
    name: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class StrangerDetected:
    topic: ClassVar[str] = "perception.stranger_detected"
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class FaceEnrollRequest:
    topic: ClassVar[str] = "perception.enroll_request"
    name: str = ""
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class BatteryLow:
    topic: ClassVar[str] = "perception.battery_low"
    voltage: float = 0.0
    percent: int = 0
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class BatteryCritical:
    topic: ClassVar[str] = "perception.battery_critical"
    voltage: float = 0.0
    percent: int = 0
    timestamp: float = field(default_factory=_now, compare=False)


@dataclass(frozen=True)
class TaskRequested:
    """
    Published by the Orchestrator when a voice command is a multi-step task
    rather than a simple movement command.
    Phase 2: a TaskBehavior subscribes to this and drives the planner.
    Examples: "find a red ball", "what do you see", "explore the room".
    """
    topic: ClassVar[str] = "brain.task_requested"
    goal: str = ""
    source: str = "voice"   # "voice" | "autonomous"
    timestamp: float = field(default_factory=_now, compare=False)
