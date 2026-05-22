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
    from .behaviors.idle_stand import IdleStand
    from .behaviors.smart_patrol import SmartPatrol
    from .brain.arbiter import Arbiter
    from .brain.llm_client import FallbackOllamaClient
    from .brain.nav_memory import NavMemory
    from .brain.orchestrator import Orchestrator
    from .brain.planner import Planner
    from .core.bus import MessageBus
    from .core.state import StateManager
    from .hearing.audio_input import AudioInput
    from .hearing.denoise import AudioDenoiser
    from .hearing.stt import SpeechToText
    from .hearing.wake_word import WakeWordListener
    from .motor.controller import CrawlerController
    from .motor.safety import SafetyLoop
    from .perception.battery import BatteryMonitor
    from .perception.camera import Camera
    from .perception.people_registry import PeopleRegistry
    from .perception.presence import PresenceDetector
    from .speech.tts import TextToSpeech

    # ------------------------------------------------------------------ #
    # Configuration from environment (override in systemd unit or .env)   #
    # ------------------------------------------------------------------ #
    ollama_host = os.environ.get("OLLAMA_HOST", "http://192.168.50.100:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    ollama_fallback_host = os.environ.get("OLLAMA_FALLBACK_HOST", "http://localhost:11434")
    ollama_fallback_model = os.environ.get("OLLAMA_FALLBACK_MODEL", "llama3.2:1b")
    whisper_model = os.environ.get("WHISPER_MODEL", "tiny")
    piper_model = os.environ.get("PIPER_MODEL", "")           # path to .onnx
    wake_threshold = float(os.environ.get("WAKE_WORD_THRESHOLD", "0.5"))
    wake_word_model = os.environ.get("WAKE_WORD_MODEL", "hey_jarvis_v0.1")
    face_registry_path = os.environ.get("FACE_REGISTRY_PATH", "/home/penguin/people_registry")
    face_threshold = float(os.environ.get("FACE_THRESHOLD", "0.4"))
    nav_memory_path = os.environ.get("NAV_MEMORY_PATH", "/home/penguin/picrawler-voice/nav_memory.db")

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

    from robot_hat.utils import enable_speaker
    enable_speaker()
    logger.info("main: speaker amplifier enabled (GPIO 20)")

    # ------------------------------------------------------------------ #
    # Hearing                                                              #
    # ------------------------------------------------------------------ #
    audio = AudioInput()
    denoiser = AudioDenoiser()
    stt = SpeechToText(model_size=whisper_model, device="cpu", denoiser=denoiser)
    import openwakeword as _oww_pkg
    _oww_model_dir = os.path.join(os.path.dirname(_oww_pkg.__file__), "resources", "models")
    _oww_model_path = os.path.join(_oww_model_dir, f"{wake_word_model}.onnx")
    wake_word = WakeWordListener(
        bus, state, audio, stt,
        model_paths=[_oww_model_path],
        threshold=wake_threshold,
    )

    # ------------------------------------------------------------------ #
    # Speech                                                               #
    # ------------------------------------------------------------------ #
    tts = TextToSpeech(bus, model_path=piper_model or None)

    # ------------------------------------------------------------------ #
    # Perception                                                           #
    # ------------------------------------------------------------------ #
    logger.info("main: initialising perception")
    battery = BatteryMonitor(bus, state)
    registry = PeopleRegistry(base_path=face_registry_path, threshold=face_threshold)
    camera = Camera()
    presence = PresenceDetector(bus, state, camera, registry)

    # ------------------------------------------------------------------ #
    # Brain                                                                #
    # ------------------------------------------------------------------ #
    llm = FallbackOllamaClient(
        primary_url=ollama_host,
        primary_model=ollama_model,
        fallback_url=ollama_fallback_host,
        fallback_model=ollama_fallback_model,
    )
    nav_memory = NavMemory(nav_memory_path)
    planner = Planner(llm, nav_memory)

    arbiter = Arbiter(
        bus,
        state,
        make_patrol=lambda: SmartPatrol(bus, state, planner, nav_memory, safety),
    )
    orchestrator = Orchestrator(
        bus, state, arbiter, llm, ctrl,
        planner=planner,
        memory=nav_memory,
        safety=safety,
    )

    # ------------------------------------------------------------------ #
    # Launch                                                               #
    # ------------------------------------------------------------------ #
    tasks = [
        asyncio.create_task(safety.run(),       name="safety"),
        asyncio.create_task(battery.run(),      name="battery"),
        asyncio.create_task(wake_word.run(),    name="wake_word"),
        asyncio.create_task(tts.run(),          name="tts"),
        asyncio.create_task(arbiter.run(),      name="arbiter"),
        asyncio.create_task(orchestrator.run(), name="orchestrator"),
        asyncio.create_task(presence.run(),     name="presence"),
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
