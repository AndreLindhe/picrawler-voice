from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.bus import MessageBus

from ..core.events import SpeakRequest

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_RATE = 22050


def _find_piper() -> Optional[str]:
    """Find piper binary: system PATH first, then the venv that runs this process."""
    p = shutil.which("piper")
    if p:
        return p
    venv_bin = os.path.join(os.path.dirname(sys.executable), "piper")
    if os.path.isfile(venv_bin) and os.access(venv_bin, os.X_OK):
        return venv_bin
    return None


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
        piper_bin = _find_piper()
        if self._model_path and piper_bin:
            logger.info("tts: piper → aplay [%s]", self._model_path)
            self._speak_piper(text, piper_bin)
        elif self._model_path and not piper_bin:
            logger.error("tts: PIPER_MODEL is set but piper binary not found — no audio")
        elif shutil.which("espeak-ng"):
            self._speak_espeak(text)
        elif shutil.which("espeak"):
            logger.warning("tts: falling back to espeak")
            subprocess.run(["espeak", text], capture_output=True)
        else:
            logger.warning("tts: no TTS backend found (piper/espeak-ng/espeak)")

    def _speak_piper(self, text: str, piper_bin: str) -> None:
        piper = subprocess.Popen(
            [piper_bin, "--model", self._model_path, "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Piper outputs mono PCM; the ALSA plug layer upmixes to stereo for the HifiBerry DAC.
        aplay = subprocess.Popen(
            [
                "aplay",
                "-r", str(self._sample_rate),
                "-f", "S16_LE",
                "-c", "1",
                "-t", "raw",
                "-",
            ],
            stdin=piper.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        piper.stdin.write(text.encode())
        piper.stdin.close()
        piper.wait()
        aplay.wait()

        if piper.returncode != 0:
            err = (piper.stderr.read() or b"").decode(errors="replace").strip()
            logger.error("tts: piper exited %d — %s", piper.returncode, err)
        if aplay.returncode != 0:
            err = (aplay.stderr.read() or b"").decode(errors="replace").strip()
            logger.error("tts: aplay exited %d — %s", aplay.returncode, err)

    def _speak_espeak(self, text: str) -> None:
        logger.warning("tts: falling back to espeak-ng (no piper model configured)")
        result = subprocess.run(
            ["espeak-ng", "-s", "140", text],
            capture_output=True,
        )
        if result.returncode != 0:
            logger.error("tts: espeak-ng exited %d — %s", result.returncode,
                         result.stderr.decode(errors="replace").strip())
