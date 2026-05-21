from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Optional


@dataclass(frozen=True)
class SceneSnapshot:
    """
    A point-in-time snapshot of what the robot perceives.

    Phase 1: sonar only.
    Phase 2: objects list populated by Hailo YOLO — nothing else changes.
    The planner and memory system consume only this dataclass.
    """

    sonar_cm: Optional[float]
    objects: tuple[str, ...] = ()   # Phase 2: ("chair [centre]", "wall [left]", ...)
    timestamp: float = field(default_factory=time.monotonic, compare=False)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def sonar_range(self) -> str:
        """Coarse bucket used for memory recall matching."""
        if self.sonar_cm is None:
            return "unknown"
        if self.sonar_cm < 15:
            return "very_close"
        if self.sonar_cm < 30:
            return "close"
        if self.sonar_cm < 50:
            return "moderate"
        return "clear"

    def to_text(self) -> str:
        """Compact human-readable description for LLM prompts."""
        parts = []
        if self.sonar_cm is not None:
            parts.append(f"front sonar: {self.sonar_cm:.0f} cm ({self.sonar_range()})")
        else:
            parts.append("front sonar: unavailable")
        if self.objects:
            parts.append(f"visible objects: {', '.join(self.objects)}")
        return "; ".join(parts)
