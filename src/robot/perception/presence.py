from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..core.bus import MessageBus
    from ..core.state import StateManager
    from .camera import Camera
    from .people_registry import PeopleRegistry

from ..core.events import (
    FaceEnrollRequest,
    PersonDetected,
    SpeakRequest,
    StrangerDetected,
)

logger = logging.getLogger(__name__)

# How often to grab and analyse a frame (seconds).
_FRAME_INTERVAL_S = 0.5

# Minimum face detection confidence to consider a detection valid.
_DET_SCORE_MIN = 0.6

# Seconds before greeting the same known person again.
_KNOWN_COOLDOWN_S = 300

# Seconds before greeting a stranger again.
_STRANGER_COOLDOWN_S = 60


class PresenceDetector:
    """
    Continuously captures frames from the camera, detects faces via
    InsightFace (ONNX/CPU), and publishes PersonDetected or
    StrangerDetected events on the message bus.

    Also listens for FaceEnrollRequest events: on the next frame after
    receiving one, it saves the most prominent face into the registry.
    """

    def __init__(
        self,
        bus: "MessageBus",
        state: "StateManager",
        camera: "Camera",
        registry: "PeopleRegistry",
    ) -> None:
        self._bus = bus
        self._state = state
        self._camera = camera
        self._registry = registry
        self._analyzer = None  # Loaded lazily in thread

        # Cooldown tracking: name -> last_greeted_at (monotonic)
        self._last_greeted: dict[str, float] = {}
        self._last_stranger_at: float = 0.0

        # If set, the next detected face will be enrolled with this name.
        self._pending_enroll: Optional[str] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Long-lived task. Cancel to stop."""
        logger.info("presence: loading face analyser (first run downloads ~150 MB)…")
        self._analyzer = await asyncio.to_thread(self._load_analyzer)
        logger.info("presence: face analyser ready")

        self._camera.start()

        # Subscribe to enroll requests in a background sub-task.
        enroll_task = asyncio.create_task(self._watch_enrollments(), name="presence.enroll")

        try:
            while True:
                t0 = time.monotonic()
                await self._process_frame()
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, _FRAME_INTERVAL_S - elapsed))
        except asyncio.CancelledError:
            raise
        finally:
            enroll_task.cancel()
            self._camera.stop()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _process_frame(self) -> None:
        frame_rgb = await self._camera.capture_frame_async()
        faces = await asyncio.to_thread(self._detect_faces, frame_rgb)

        if not faces:
            await self._state.update_people_visible([])
            return

        names_visible: list[str] = []

        for face in faces:
            if face.det_score < _DET_SCORE_MIN:
                continue

            embedding = face.normed_embedding

            # Pending enroll: save this face and announce it.
            if self._pending_enroll is not None:
                name = self._pending_enroll
                self._pending_enroll = None
                await asyncio.to_thread(self._registry.enroll, name, embedding)
                logger.info("presence: enrolled %r", name)
                self._bus.publish(SpeakRequest(text=f"Got it! I'll remember {name}."))
                names_visible.append(name)
                continue

            match = await asyncio.to_thread(self._registry.find_match, embedding)

            if match is not None:
                name, confidence = match
                names_visible.append(name)
                self._bus.publish(
                    PersonDetected(name=name, confidence=confidence)
                )
                self._maybe_greet_known(name)
            else:
                self._bus.publish(StrangerDetected())
                self._maybe_greet_stranger()

        await self._state.update_people_visible(names_visible)

    def _maybe_greet_known(self, name: str) -> None:
        now = time.monotonic()
        last = self._last_greeted.get(name, 0.0)
        if now - last >= _KNOWN_COOLDOWN_S:
            self._last_greeted[name] = now
            self._bus.publish(SpeakRequest(text=f"Hello {name}, good to see you!"))

    def _maybe_greet_stranger(self) -> None:
        now = time.monotonic()
        if now - self._last_stranger_at >= _STRANGER_COOLDOWN_S:
            self._last_stranger_at = now
            self._bus.publish(SpeakRequest(text="Hello! I don't think we've met. What's your name?"))

    async def _watch_enrollments(self) -> None:
        async with self._bus.subscribe(FaceEnrollRequest.topic) as q:
            while True:
                event: FaceEnrollRequest = await q.get()
                logger.info("presence: enroll request for %r", event.name)
                self._pending_enroll = event.name
                self._bus.publish(
                    SpeakRequest(text=f"Sure! Hold still for a moment while I take a look at you, {event.name}.")
                )

    def _detect_faces(self, frame_rgb):
        import cv2  # type: ignore[import]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return self._analyzer.get(frame_bgr)

    def _load_analyzer(self):
        from insightface.app import FaceAnalysis  # type: ignore[import]
        analyzer = FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"],
        )
        analyzer.prepare(ctx_id=0, det_size=(640, 480))
        return analyzer
