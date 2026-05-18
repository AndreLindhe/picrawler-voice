"""
PiCrawler voice-control entry point.

Start with:
    python -m robot.main
or via the systemd service.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _main() -> None:
    from .behaviors.idle_patrol import IdlePatrol
    from .brain.arbiter import Arbiter
    from .brain.llm_client import OllamaClient
    from .brain.orchestrator import Orchestrator
    from .core.bus import MessageBus
    from .core.state import StateManager
    from .hearing.audio_input import AudioInput
    from .hearing.denoise import AudioDenoiser
    from .hearing.stt import SpeechToText
    from .hearing.wake_word import WakeWordListener
    from .motor.controller import CrawlerController
    from .motor.safety import SafetyLoop
    from .speech.tts import TextToSpeech

    # ------------------------------------------------------------------ #
    # Configuration from environment (override in systemd unit or .env)   #
    # ------------------------------------------------------------------ #
    ollama_host = os.environ.get("OLLAMA_HOST", "http://192.168.50.100:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    whisper_model = os.environ.get("WHISPER_MODEL", "tiny")
    piper_model = os.environ.get("PIPER_MODEL", "")           # path to .onnx
    wake_threshold = float(os.environ.get("WAKE_WORD_THRESHOLD", "0.5"))

    # ------------------------------------------------------------------ #
    # Core                                                                 #
    # ------------------------------------------------------------------ #
    bus = MessageBus()
    state = StateManager()

    # ------------------------------------------------------------------ #
    # Motor                                                                #
    # ------------------------------------------------------------------ #
    logger.info("main: initialising crawler controller")
    ctrl = CrawlerController()
    safety = SafetyLoop(bus, state)

    # ------------------------------------------------------------------ #
    # Hearing                                                              #
    # ------------------------------------------------------------------ #
    audio = AudioInput()
    denoiser = AudioDenoiser()
    stt = SpeechToText(model_size=whisper_model, device="cpu", denoiser=denoiser)
    wake_word = WakeWordListener(
        bus, state, audio, stt, threshold=wake_threshold
    )

    # ------------------------------------------------------------------ #
    # Speech                                                               #
    # ------------------------------------------------------------------ #
    tts = TextToSpeech(bus, model_path=piper_model or None)

    # ------------------------------------------------------------------ #
    # Brain                                                                #
    # ------------------------------------------------------------------ #
    llm = OllamaClient(ollama_host, model=ollama_model)
    arbiter = Arbiter(
        bus,
        state,
        make_patrol=lambda: IdlePatrol(bus, state, initial_turn=True),
    )
    orchestrator = Orchestrator(bus, state, arbiter, llm, ctrl)

    # ------------------------------------------------------------------ #
    # Launch                                                               #
    # ------------------------------------------------------------------ #
    tasks = [
        asyncio.create_task(safety.run(),       name="safety"),
        asyncio.create_task(wake_word.run(),    name="wake_word"),
        asyncio.create_task(tts.run(),          name="tts"),
        asyncio.create_task(arbiter.run(),      name="arbiter"),
        asyncio.create_task(orchestrator.run(), name="orchestrator"),
    ]

    logger.info("main: all tasks started — robot is alive")

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_signal(sig: int) -> None:
        logger.info("main: received signal %s — shutting down", signal.Signals(sig).name)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig)

    await stop_event.wait()

    logger.info("main: cancelling tasks…")
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("main: shutting down controller")
    await ctrl.shutdown()
    logger.info("main: done")


def main() -> None:
    _setup_logging()
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
