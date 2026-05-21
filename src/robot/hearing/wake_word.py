from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..core.events import Transcript, WakeWordDetected

if TYPE_CHECKING:
    from ..core.bus import MessageBus
    from ..core.state import StateManager
    from .audio_input import AudioInput
    from .stt import SpeechToText

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.5
_POST_DETECTION_COOLDOWN_S = 3.0  # ignore wake word for this long after each detection
_PING_FREQ_HZ = 880
_PING_DURATION_S = 0.25
_PING_RATE = 22050


def _play_ping() -> None:
    """Play a short confirmation tone via aplay (same ALSA chain as TTS)."""
    try:
        t = np.linspace(0, _PING_DURATION_S, int(_PING_RATE * _PING_DURATION_S), endpoint=False)
        mono = (np.sin(2 * np.pi * _PING_FREQ_HZ * t) * 0.7 * 32767).astype(np.int16)
        fade = int(len(mono) * 0.2)
        mono[-fade:] = (mono[-fade:] * np.linspace(1.0, 0.0, fade)).astype(np.int16)
        stereo = np.column_stack([mono, mono])
        subprocess.run(
            ["aplay", "-r", str(_PING_RATE), "-f", "S16_LE", "-c", "2", "-t", "raw", "-"],
            input=stereo.tobytes(),
            check=True,
            capture_output=True,
        )
    except Exception:
        logger.debug("wake_word: ping playback failed", exc_info=True)


class WakeWordListener:
    """
    Continuously reads audio from AudioInput, feeds chunks to openwakeword,
    and — on detection — hands control to SpeechToText.

    Publishes:
      WakeWordDetected  — immediately on detection
      Transcript        — after STT completes (skipped if empty)

    run() is a long-lived task; cancel it to shut down.
    AudioInput is started/stopped by this class, not by the caller.
    """

    def __init__(
        self,
        bus: "MessageBus",
        state: "StateManager",
        audio_input: "AudioInput",
        stt: "SpeechToText",
        model_paths: Optional[list[str]] = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._bus = bus
        self._state = state
        self._audio = audio_input
        self._stt = stt
        self._model_paths = model_paths or []
        self._threshold = threshold
        self._oww = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Long-lived task.  Cancel to stop."""
        model = await asyncio.to_thread(self._load_model)
        self._audio.start()
        logger.info("wake_word: listening (threshold=%.2f)", self._threshold)

        _cooldown_until: float = 0.0
        try:
            async for chunk in self._audio:
                scores: dict = await asyncio.to_thread(model.predict, chunk)
                best_score = max(scores.values()) if scores else 0.0

                if best_score >= self._threshold:
                    if time.monotonic() < _cooldown_until:
                        continue  # still in cooldown — ignore re-trigger from ping echo
                    _cooldown_until = time.monotonic() + _POST_DETECTION_COOLDOWN_S
                    logger.info("wake_word: detected (score=%.3f)", best_score)
                    self._bus.publish(WakeWordDetected(confidence=float(best_score)))
                    await asyncio.to_thread(_play_ping)
                    self._audio.drain()  # discard ping echo captured by the mic
                    await self._record_command()
                    # Drain chunks buffered during STT to avoid stale re-triggers.
                    self._audio.drain()
                    _cooldown_until = time.monotonic() + _POST_DETECTION_COOLDOWN_S

        except asyncio.CancelledError:
            raise
        finally:
            self._audio.stop()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _record_command(self) -> None:
        logger.info("wake_word: recording command…")
        try:
            text = await self._stt.record_and_transcribe(self._audio)
        except Exception:
            logger.exception("wake_word: STT failed")
            return
        if text:
            self._bus.publish(Transcript(text=text, is_final=True))
        else:
            logger.debug("wake_word: empty transcript, ignoring")

    def _load_model(self):
        if self._oww is not None:
            return self._oww
        from openwakeword.model import Model  # type: ignore[import]
        logger.info("wake_word: loading model(s) %r", self._model_paths or ["default"])
        if self._model_paths:
            self._oww = Model(wakeword_models=self._model_paths)
        else:
            self._oww = Model(inference_framework="onnx")
        return self._oww
