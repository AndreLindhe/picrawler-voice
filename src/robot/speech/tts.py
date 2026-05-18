from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.bus import MessageBus

from ..core.events import SpeakRequest

logger = logging.getLogger(__name__)

# Piper model defaults — override via TextToSpeech(model_path=...).
_DEFAULT_SAMPLE_RATE = 22050


class TextToSpeech:
    """
    Subscribes to SpeakRequest events and speaks them in order.

    Audio pipeline:  text → piper (raw PCM) → aplay (ALSA)

    If the `piper` binary is not on PATH, falls back to `espeak-ng`.
    synthesis and playback both happen in a thread so the event loop is
    never blocked.
    """

    def __init__(
        self,
        bus: "MessageBus",
        model_path: Optional[str] = None,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
    ) -> None:
        self._bus = bus
        self._model_path = model_path
        self._sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Long-lived task.  Cancel to stop."""
        async with self._bus.subscribe(SpeakRequest.topic) as q:
            while True:
                event: SpeakRequest = await q.get()
                logger.info("tts: speaking %r", event.text[:60])
                try:
                    await asyncio.to_thread(self._speak_sync, event.text)
                except Exception:
                    logger.exception("tts: playback failed")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _speak_sync(self, text: str) -> None:
        """Blocking synthesis + playback — runs in a thread."""
        if self._model_path and shutil.which("piper"):
            self._speak_piper(text)
        elif shutil.which("espeak-ng"):
            self._speak_espeak(text)
        elif shutil.which("espeak"):
            subprocess.run(["espeak", text], check=True, capture_output=True)
        else:
            logger.warning("tts: no TTS backend found (piper/espeak-ng/espeak)")

    def _speak_piper(self, text: str) -> None:
        piper = subprocess.Popen(
            ["piper", "--model", self._model_path, "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        aplay = subprocess.Popen(
            [
                "aplay",
                "-r", str(self._sample_rate),
                "-f", "S16_LE",
                "-t", "raw",
                "-",
            ],
            stdin=piper.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        piper.stdin.write(text.encode())
        piper.stdin.close()
        piper.wait()
        aplay.wait()

    def _speak_espeak(self, text: str) -> None:
        subprocess.run(
            ["espeak-ng", "-s", "140", text],
            check=True,
            capture_output=True,
        )
