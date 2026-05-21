from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_WIDTH = 640
_HEIGHT = 480


class Camera:
    """
    Async wrapper around Picamera2 for the Camera Module 3.

    Captures RGB frames as numpy arrays (H, W, 3).
    Call start() once from within a running event loop, stop() to release.
    """

    def __init__(
        self,
        width: int = _WIDTH,
        height: int = _HEIGHT,
    ) -> None:
        self._width = width
        self._height = height
        self._cam = None

    def start(self) -> None:
        from picamera2 import Picamera2  # type: ignore[import]

        self._cam = Picamera2()
        config = self._cam.create_still_configuration(
            main={"size": (self._width, self._height), "format": "RGB888"},
            buffer_count=2,
        )
        self._cam.configure(config)
        self._cam.start()
        logger.info("camera: started (%dx%d)", self._width, self._height)

    def stop(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None
            logger.info("camera: stopped")

    def capture_frame(self) -> np.ndarray:
        """Return the latest frame as an RGB (H, W, 3) uint8 array."""
        assert self._cam is not None, "Camera.start() must be called first"
        return self._cam.capture_array("main")

    async def capture_frame_async(self) -> np.ndarray:
        """Non-blocking frame capture — runs in a thread."""
        return await asyncio.to_thread(self.capture_frame)
