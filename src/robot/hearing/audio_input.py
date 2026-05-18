from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
CHUNK_SAMPLES = 1280  # 80 ms — matches openwakeword's expected frame size
_QUEUE_MAX = 64       # ~5 s of buffering before oldest chunks are dropped


class AudioInput:
    """
    Async wrapper around a sounddevice InputStream.

    sounddevice fires its callback from a C thread.  We bridge to asyncio via
    loop.call_soon_threadsafe so callers can simply `await audio.read()` or
    iterate `async for chunk in audio` — no threads visible to the consumer.

    Call start() once (from within a running event loop), then read as needed.
    Call stop() to release the hardware.
    """

    def __init__(
        self,
        device: Optional[int | str] = None,
        chunk_samples: int = CHUNK_SAMPLES,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._device = device
        self._chunk_samples = chunk_samples
        self._sample_rate = sample_rate
        self._q: asyncio.Queue[np.ndarray] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start audio capture.  Must be called from within a running event loop."""
        if self._stream is not None:
            return
        import sounddevice as sd  # type: ignore[import]

        self._loop = asyncio.get_running_loop()
        self._q = asyncio.Queue(maxsize=_QUEUE_MAX)

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self._chunk_samples,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "audio: stream started (rate=%d, chunk=%d samples)",
            self._sample_rate,
            self._chunk_samples,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("audio: stream stopped")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    async def read(self) -> np.ndarray:
        """Return the next audio chunk.  Blocks until one is available."""
        assert self._q is not None, "AudioInput.start() must be called first"
        return await self._q.get()

    def drain(self) -> int:
        """Discard all queued chunks.  Returns number of chunks dropped."""
        assert self._q is not None
        count = 0
        while not self._q.empty():
            try:
                self._q.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        if count:
            logger.debug("audio: drained %d stale chunks", count)
        return count

    def __aiter__(self):
        return self

    async def __anext__(self) -> np.ndarray:
        return await self.read()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug("audio: %s", status)
        chunk = indata[:, 0].copy()  # mono: flatten first channel
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._put_chunk, chunk)

    def _put_chunk(self, chunk: np.ndarray) -> None:
        """Runs in the event loop thread; safe to manipulate asyncio.Queue."""
        assert self._q is not None
        if self._q.full():
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            logger.debug("audio: dropped oldest chunk (consumer lagging)")
        self._q.put_nowait(chunk)
