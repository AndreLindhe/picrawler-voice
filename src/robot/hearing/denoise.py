from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_GATE_RATIO = 1.5        # RMS must be >= noise_floor × ratio to pass the gate
_CALIBRATION_CHUNKS = 20  # ~1.6 s at 80 ms/chunk before noise floor is set


class AudioDenoiser:
    """
    Two-mode denoiser for the hearing pipeline.

    process_chunk() — per-chunk noise gate used during wake-word listening.
      Calibrates a rolling noise floor from the first ~1.6 s of audio, then
      zeros out chunks that fall below the gate threshold.  Pass-through during
      calibration so wake-word detection is not delayed.

    process_buffer() — batch denoising on the full captured utterance before
      STT.  Uses noisereduce if installed; falls back to the raw audio.

    Both modes return a new array and leave the input unchanged.
    """

    def __init__(self, sample_rate: int = 16_000, use_noisereduce: bool = True) -> None:
        self._sample_rate = sample_rate
        self._nr = None
        if use_noisereduce:
            try:
                import noisereduce as nr  # type: ignore[import]
                self._nr = nr
                logger.info("denoise: noisereduce backend available")
            except ImportError:
                logger.info("denoise: noisereduce not installed; gate fallback only")

        self._noise_floor: Optional[float] = None
        self._cal_samples: list[float] = []

    # ------------------------------------------------------------------
    # Per-chunk gate (wake-word phase)
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        rms = _rms(chunk)

        if self._noise_floor is None:
            self._cal_samples.append(rms)
            if len(self._cal_samples) >= _CALIBRATION_CHUNKS:
                self._noise_floor = float(np.mean(self._cal_samples))
                logger.info("denoise: noise floor calibrated (RMS=%.1f)", self._noise_floor)
            return chunk  # pass-through during calibration

        if rms < self._noise_floor * _GATE_RATIO:
            return np.zeros_like(chunk)
        return chunk

    # ------------------------------------------------------------------
    # Full-buffer (before STT)
    # ------------------------------------------------------------------

    def process_buffer(self, audio: np.ndarray) -> np.ndarray:
        """
        Denoise a complete utterance (float32, range ≈ [-1, 1]).
        No-op if noisereduce is not installed.
        """
        if self._nr is None:
            return audio
        try:
            return self._nr.reduce_noise(y=audio, sr=self._sample_rate, stationary=True)
        except Exception:
            logger.warning("denoise: buffer denoising failed; using raw audio", exc_info=True)
            return audio


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
