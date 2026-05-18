from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .audio_input import AudioInput
    from .denoise import AudioDenoiser

logger = logging.getLogger(__name__)

_MAX_RECORD_S = 10.0
_SILENCE_RMS_THRESHOLD = 200    # int16 RMS — below this counts as silence
_SILENCE_CHUNKS_NEEDED = 10     # consecutive silent chunks → end-of-speech (~0.8 s)


class SpeechToText:
    """
    Records audio from AudioInput after a wake-word event, then transcribes
    it with faster-whisper.

    The Whisper model is loaded lazily on the first transcription call so
    startup is fast and tests that never transcribe skip the model download.

    Optional denoiser: if provided, process_buffer() is applied to the
    concatenated recording before Whisper sees it.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "en",
        denoiser: Optional["AudioDenoiser"] = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._denoiser = denoiser
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record_and_transcribe(
        self,
        audio_input: "AudioInput",
        max_seconds: float = _MAX_RECORD_S,
    ) -> str:
        """
        Buffer chunks from *audio_input* until end-of-speech silence or
        *max_seconds*, then transcribe and return the text (may be empty).
        """
        chunks: list[np.ndarray] = []
        silence_count = 0
        chunk_s = audio_input.chunk_samples / audio_input.sample_rate
        max_chunks = int(max_seconds / chunk_s)

        for _ in range(max_chunks):
            chunk = await audio_input.read()
            chunks.append(chunk)
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            if rms < _SILENCE_RMS_THRESHOLD:
                silence_count += 1
                if silence_count >= _SILENCE_CHUNKS_NEEDED:
                    logger.debug("stt: end-of-speech detected after %d chunks", len(chunks))
                    break
            else:
                silence_count = 0

        if not chunks:
            return ""

        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        if self._denoiser is not None:
            audio = self._denoiser.process_buffer(audio)

        text = await asyncio.to_thread(self._transcribe, audio)
        logger.info("stt: %r", text)
        return text

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore[import]
            logger.info(
                "stt: loading whisper/%s on %s (%s)",
                self._model_size,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def _transcribe(self, audio: np.ndarray) -> str:
        model = self._load_model()
        segments, _ = model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
