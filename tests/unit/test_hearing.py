"""
Unit tests for the hearing pipeline — no hardware, no ML models required.

All external I/O (sounddevice, openwakeword, faster-whisper, noisereduce) is
mocked so these tests run anywhere.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from robot.core.bus import MessageBus
from robot.core.events import Transcript, WakeWordDetected
from robot.core.state import StateManager
from robot.hearing.audio_input import AudioInput, CHUNK_SAMPLES, SAMPLE_RATE
from robot.hearing.denoise import AudioDenoiser, _CALIBRATION_CHUNKS, _GATE_RATIO
from robot.hearing.stt import SpeechToText, _SILENCE_CHUNKS_NEEDED, _SILENCE_RMS_THRESHOLD
from robot.hearing.wake_word import WakeWordListener


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silent_chunk() -> np.ndarray:
    """A chunk of all-zeros (definitely below silence threshold)."""
    return np.zeros(CHUNK_SAMPLES, dtype=np.int16)


def _loud_chunk(level: int = 2000) -> np.ndarray:
    """A chunk of constant non-zero samples (above silence threshold)."""
    return np.full(CHUNK_SAMPLES, level, dtype=np.int16)


def _make_audio_input() -> AudioInput:
    """AudioInput with a pre-created async queue (no sounddevice)."""
    ai = AudioInput.__new__(AudioInput)
    ai._device = None
    ai._chunk_samples = CHUNK_SAMPLES
    ai._sample_rate = SAMPLE_RATE
    ai._q = asyncio.Queue(maxsize=64)
    ai._loop = None
    ai._stream = None
    return ai


async def _drain_bus(bus: MessageBus, topic: str, count: int, timeout: float = 1.0) -> list:
    events = []
    async with bus.subscribe(topic) as q:
        for _ in range(count):
            events.append(await asyncio.wait_for(q.get(), timeout=timeout))
    return events


# ---------------------------------------------------------------------------
# AudioInput
# ---------------------------------------------------------------------------

class TestAudioInput:

    def test_put_chunk_enqueues(self):
        ai = _make_audio_input()
        chunk = _loud_chunk()
        ai._put_chunk(chunk)
        assert not ai._q.empty()
        assert np.array_equal(ai._q.get_nowait(), chunk)

    def test_put_chunk_drops_oldest_when_full(self):
        ai = _make_audio_input()
        ai._q = asyncio.Queue(maxsize=2)

        first = _loud_chunk(1000)
        second = _loud_chunk(2000)
        third = _loud_chunk(3000)

        ai._put_chunk(first)
        ai._put_chunk(second)
        # Queue is full — oldest (first) should be dropped
        ai._put_chunk(third)

        assert ai._q.qsize() == 2
        assert np.array_equal(ai._q.get_nowait(), second)
        assert np.array_equal(ai._q.get_nowait(), third)

    @pytest.mark.asyncio
    async def test_read_returns_chunk(self):
        ai = _make_audio_input()
        chunk = _loud_chunk()
        ai._q.put_nowait(chunk)
        result = await ai.read()
        assert np.array_equal(result, chunk)

    @pytest.mark.asyncio
    async def test_aiter_yields_chunks(self):
        ai = _make_audio_input()
        chunks = [_loud_chunk(i * 100) for i in range(3)]
        for c in chunks:
            ai._q.put_nowait(c)

        received = []
        async for chunk in ai:
            received.append(chunk)
            if len(received) == 3:
                break

        assert len(received) == 3
        for expected, actual in zip(chunks, received):
            assert np.array_equal(expected, actual)

    def test_drain_clears_queue(self):
        ai = _make_audio_input()
        for _ in range(5):
            ai._q.put_nowait(_loud_chunk())
        count = ai.drain()
        assert count == 5
        assert ai._q.empty()

    def test_drain_empty_queue_returns_zero(self):
        ai = _make_audio_input()
        assert ai.drain() == 0


# ---------------------------------------------------------------------------
# AudioDenoiser
# ---------------------------------------------------------------------------

class TestAudioDenoiser:

    def test_passthrough_during_calibration(self):
        denoiser = AudioDenoiser(use_noisereduce=False)
        chunk = _loud_chunk(100)
        result = denoiser.process_chunk(chunk)
        assert np.array_equal(result, chunk)
        assert denoiser._noise_floor is None

    def test_noise_floor_set_after_calibration(self):
        denoiser = AudioDenoiser(use_noisereduce=False)
        chunk = _loud_chunk(1000)
        for _ in range(_CALIBRATION_CHUNKS):
            denoiser.process_chunk(chunk)
        assert denoiser._noise_floor is not None
        assert denoiser._noise_floor > 0

    def test_gate_passes_loud_chunk(self):
        denoiser = AudioDenoiser(use_noisereduce=False)
        # Calibrate with a quiet signal so the gate is set low
        quiet = _loud_chunk(10)
        for _ in range(_CALIBRATION_CHUNKS):
            denoiser.process_chunk(quiet)

        loud = _loud_chunk(5000)
        result = denoiser.process_chunk(loud)
        assert np.any(result != 0), "loud chunk should pass gate"

    def test_gate_silences_quiet_chunk(self):
        denoiser = AudioDenoiser(use_noisereduce=False)
        loud = _loud_chunk(5000)
        for _ in range(_CALIBRATION_CHUNKS):
            denoiser.process_chunk(loud)

        quiet = _loud_chunk(10)  # well below floor × ratio
        result = denoiser.process_chunk(quiet)
        assert np.all(result == 0), "quiet chunk should be gated"

    def test_process_buffer_passthrough_without_noisereduce(self):
        denoiser = AudioDenoiser(use_noisereduce=False)
        audio = np.random.randn(16000).astype(np.float32)
        result = denoiser.process_buffer(audio)
        assert np.array_equal(result, audio)

    def test_process_buffer_uses_noisereduce_when_available(self):
        nr_mock = MagicMock()
        nr_mock.reduce_noise.return_value = np.zeros(16000, dtype=np.float32)

        denoiser = AudioDenoiser(use_noisereduce=False)
        denoiser._nr = nr_mock

        audio = np.random.randn(16000).astype(np.float32)
        result = denoiser.process_buffer(audio)

        nr_mock.reduce_noise.assert_called_once()
        assert np.all(result == 0)

    def test_process_buffer_fallback_on_nr_error(self):
        nr_mock = MagicMock()
        nr_mock.reduce_noise.side_effect = RuntimeError("boom")

        denoiser = AudioDenoiser(use_noisereduce=False)
        denoiser._nr = nr_mock

        audio = np.ones(16000, dtype=np.float32)
        result = denoiser.process_buffer(audio)
        assert np.array_equal(result, audio)


# ---------------------------------------------------------------------------
# SpeechToText
# ---------------------------------------------------------------------------

class TestSpeechToText:

    @pytest.mark.asyncio
    async def test_records_until_silence(self):
        ai = _make_audio_input()
        stt = SpeechToText.__new__(SpeechToText)
        stt._model = None
        stt._model_size = "tiny"
        stt._device = "cpu"
        stt._compute_type = "int8"
        stt._language = "en"
        stt._denoiser = None

        # Put some speech chunks then silence
        for _ in range(5):
            ai._q.put_nowait(_loud_chunk(2000))
        for _ in range(_SILENCE_CHUNKS_NEEDED):
            ai._q.put_nowait(_silent_chunk())

        # Patch _transcribe to avoid loading Whisper
        stt._transcribe = MagicMock(return_value="go forward")

        result = await stt.record_and_transcribe(ai)
        assert result == "go forward"
        stt._transcribe.assert_called_once()
        audio_arg = stt._transcribe.call_args[0][0]
        assert audio_arg.dtype == np.float32

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_chunks(self):
        ai = _make_audio_input()
        stt = SpeechToText.__new__(SpeechToText)
        stt._model = None
        stt._model_size = "tiny"
        stt._device = "cpu"
        stt._compute_type = "int8"
        stt._language = "en"
        stt._denoiser = None

        # Only silence so it exits immediately via the silence detector
        for _ in range(_SILENCE_CHUNKS_NEEDED):
            ai._q.put_nowait(_silent_chunk())

        stt._transcribe = MagicMock(return_value="")
        result = await stt.record_and_transcribe(ai)
        # Even all-silent chunks are buffered; transcribe is still called
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_denoiser_called_on_buffer(self):
        ai = _make_audio_input()
        for _ in range(_SILENCE_CHUNKS_NEEDED):
            ai._q.put_nowait(_loud_chunk(2000))
        for _ in range(_SILENCE_CHUNKS_NEEDED):
            ai._q.put_nowait(_silent_chunk())

        denoiser = MagicMock()
        denoiser.process_buffer = MagicMock(side_effect=lambda a: a)

        stt = SpeechToText.__new__(SpeechToText)
        stt._model = None
        stt._model_size = "tiny"
        stt._device = "cpu"
        stt._compute_type = "int8"
        stt._language = "en"
        stt._denoiser = denoiser
        stt._transcribe = MagicMock(return_value="turn left")

        await stt.record_and_transcribe(ai)
        denoiser.process_buffer.assert_called_once()

    @pytest.mark.asyncio
    async def test_stops_at_max_seconds(self):
        ai = _make_audio_input()
        # With max_seconds=0.24 and chunk_duration=0.08, only 3 chunks will be read.
        # Put exactly those 3 loud chunks so we don't exceed the queue limit.
        for _ in range(3):
            ai._q.put_nowait(_loud_chunk(2000))

        stt = SpeechToText.__new__(SpeechToText)
        stt._model = None
        stt._model_size = "tiny"
        stt._device = "cpu"
        stt._compute_type = "int8"
        stt._language = "en"
        stt._denoiser = None
        stt._transcribe = MagicMock(return_value="stop")

        # With max_seconds=0.24 and chunk_duration=0.08, only 3 chunks should be read
        result = await stt.record_and_transcribe(ai, max_seconds=0.24)
        assert result == "stop"
        audio_arg = stt._transcribe.call_args[0][0]
        expected_samples = 3 * CHUNK_SAMPLES
        assert len(audio_arg) == expected_samples


# ---------------------------------------------------------------------------
# WakeWordListener
# ---------------------------------------------------------------------------

class TestWakeWordListener:

    def _make_listener(self, ai, stt, threshold=0.5):
        bus = MessageBus()
        state = StateManager()
        listener = WakeWordListener.__new__(WakeWordListener)
        listener._bus = bus
        listener._state = state
        listener._audio = ai
        listener._stt = stt
        listener._model_paths = []
        listener._threshold = threshold
        listener._oww = None
        return listener, bus

    @pytest.mark.asyncio
    async def test_publishes_wake_word_detected(self):
        ai = _make_audio_input()
        stt = MagicMock()
        stt.record_and_transcribe = AsyncMock(return_value="go forward")

        model_mock = MagicMock()
        model_mock.predict.return_value = {"hey_robot": 0.9}

        listener, bus = self._make_listener(ai, stt)
        listener._oww = model_mock

        # Patch start/stop/drain so they're no-ops
        ai.start = MagicMock()
        ai.stop = MagicMock()
        ai.drain = MagicMock()

        # Feed one loud chunk (triggers detection), then cancel
        ai._q.put_nowait(_loud_chunk())

        async with bus.subscribe(WakeWordDetected.topic) as q:
            task = asyncio.create_task(listener.run())
            event = await asyncio.wait_for(q.get(), timeout=2.0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert isinstance(event, WakeWordDetected)
        assert event.confidence == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_publishes_transcript_after_stt(self):
        ai = _make_audio_input()
        stt = MagicMock()
        stt.record_and_transcribe = AsyncMock(return_value="turn right")

        model_mock = MagicMock()
        model_mock.predict.return_value = {"hey_robot": 0.8}

        listener, bus = self._make_listener(ai, stt)
        listener._oww = model_mock

        ai.start = MagicMock()
        ai.stop = MagicMock()
        ai.drain = MagicMock()

        ai._q.put_nowait(_loud_chunk())

        async with bus.subscribe(Transcript.topic) as q:
            task = asyncio.create_task(listener.run())
            event = await asyncio.wait_for(q.get(), timeout=2.0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert isinstance(event, Transcript)
        assert event.text == "turn right"
        assert event.is_final is True

    @pytest.mark.asyncio
    async def test_empty_transcript_not_published(self):
        ai = _make_audio_input()
        stt = MagicMock()
        stt.record_and_transcribe = AsyncMock(return_value="")

        model_mock = MagicMock()
        model_mock.predict.return_value = {"hey_robot": 0.8}

        listener, bus = self._make_listener(ai, stt)
        listener._oww = model_mock

        ai.start = MagicMock()
        ai.stop = MagicMock()
        ai.drain = MagicMock()

        # One detection chunk, then an endless stream of below-threshold chunks
        ai._q.put_nowait(_loud_chunk())
        for _ in range(10):
            ai._q.put_nowait(_loud_chunk())

        # Override predict so only the first chunk triggers detection
        call_count = 0
        original_predict = model_mock.predict

        def predict_side(chunk):
            nonlocal call_count
            call_count += 1
            return {"hey_robot": 0.8} if call_count == 1 else {"hey_robot": 0.0}

        model_mock.predict.side_effect = predict_side

        published = []
        async with bus.subscribe(Transcript.topic) as q:
            task = asyncio.create_task(listener.run())
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            while not q.empty():
                published.append(q.get_nowait())

        assert published == [], "empty transcript should not be published"

    @pytest.mark.asyncio
    async def test_below_threshold_no_detection(self):
        ai = _make_audio_input()
        stt = MagicMock()
        stt.record_and_transcribe = AsyncMock(return_value="")

        model_mock = MagicMock()
        model_mock.predict.return_value = {"hey_robot": 0.1}  # below threshold

        listener, bus = self._make_listener(ai, stt, threshold=0.5)
        listener._oww = model_mock

        ai.start = MagicMock()
        ai.stop = MagicMock()
        ai.drain = MagicMock()

        for _ in range(5):
            ai._q.put_nowait(_loud_chunk())

        published = []
        async with bus.subscribe(WakeWordDetected.topic) as q:
            task = asyncio.create_task(listener.run())
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            while not q.empty():
                published.append(q.get_nowait())

        assert published == []
        stt.record_and_transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_audio_stopped_on_cancel(self):
        ai = _make_audio_input()
        stt = MagicMock()
        stt.record_and_transcribe = AsyncMock(return_value="")

        model_mock = MagicMock()
        model_mock.predict.return_value = {"hey_robot": 0.0}

        listener, bus = self._make_listener(ai, stt)
        listener._oww = model_mock

        ai.start = MagicMock()
        ai.stop = MagicMock()
        ai.drain = MagicMock()

        for _ in range(5):
            ai._q.put_nowait(_loud_chunk())

        task = asyncio.create_task(listener.run())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        ai.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_drain_called_after_detection(self):
        ai = _make_audio_input()
        stt = MagicMock()
        stt.record_and_transcribe = AsyncMock(return_value="forward")

        model_mock = MagicMock()
        model_mock.predict.return_value = {"hey_robot": 0.9}

        listener, bus = self._make_listener(ai, stt)
        listener._oww = model_mock

        ai.start = MagicMock()
        ai.stop = MagicMock()
        ai.drain = MagicMock()

        ai._q.put_nowait(_loud_chunk())

        async with bus.subscribe(Transcript.topic) as q:
            task = asyncio.create_task(listener.run())
            await asyncio.wait_for(q.get(), timeout=2.0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        ai.drain.assert_called()
