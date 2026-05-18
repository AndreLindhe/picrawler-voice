"""
Arbiter smoke tests — covers the four state transitions that matter most:

1. Patrol starts automatically when nothing else is running.
2. A higher-priority voice request preempts patrol.
3. After the voice command completes naturally, patrol resumes.
   (This is the bug from the original design: without explicit task.done()
   checks the supervisor would hang waiting for a wakeup that never came.)
4. An obstacle event cancels the current behavior and blocks until cleared.
"""

import asyncio
import pytest

from robot.behaviors.base import Behavior
from robot.brain.arbiter import PRIORITY_PATROL, PRIORITY_VOICE, Arbiter
from robot.core.bus import MessageBus
from robot.core.events import BehaviorEnded, BehaviorStarted, ObstacleCleared, ObstacleDetected
from robot.core.state import StateManager


# ---------------------------------------------------------------------------
# Minimal fake behaviors (no hardware required)
# ---------------------------------------------------------------------------

class _RunOnce(Behavior):
    """Finishes immediately."""
    async def _run(self) -> None:
        pass


class _RunForever(Behavior):
    """Blocks until cancelled."""
    async def _run(self) -> None:
        await asyncio.sleep(3600)


class _SlowCleanup(Behavior):
    """Takes a moment to clean up — exercises the shield path."""
    cleaned_up: bool = False

    async def _run(self) -> None:
        await asyncio.sleep(3600)

    async def _cleanup(self) -> None:
        await asyncio.sleep(0.05)
        _SlowCleanup.cleaned_up = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state() -> tuple[MessageBus, StateManager]:
    bus = MessageBus()
    state = StateManager()
    return bus, state


async def _run_arbiter(arbiter: Arbiter, *, timeout: float = 2.0) -> None:
    """Run arbiter as a background task, cancel it when done with the test."""
    task = asyncio.create_task(arbiter.run(), name="test.arbiter")
    yield task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _collect_events(bus: MessageBus, topic: str, count: int, timeout: float = 1.0) -> list:
    events = []
    async with bus.subscribe(topic) as q:
        for _ in range(count):
            events.append(await asyncio.wait_for(q.get(), timeout=timeout))
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_patrol_starts_by_default():
    bus, state = _make_state()
    patrol_started = asyncio.Event()

    class _Patrol(_RunForever):
        async def _run(self) -> None:
            patrol_started.set()
            await super()._run()

    arbiter = Arbiter(bus, state, make_patrol=lambda: _Patrol(bus, state))
    arb_task = asyncio.create_task(arbiter.run())

    await asyncio.wait_for(patrol_started.wait(), timeout=1.0)

    arb_task.cancel()
    try:
        await arb_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_voice_preempts_patrol():
    bus, state = _make_state()
    voice_started = asyncio.Event()

    class _Voice(_RunForever):
        async def _run(self) -> None:
            voice_started.set()
            await super()._run()

    arbiter = Arbiter(bus, state, make_patrol=lambda: _RunForever(bus, state))
    arb_task = asyncio.create_task(arbiter.run())

    # Give patrol time to start
    await asyncio.sleep(0.05)

    arbiter.request(_Voice(bus, state), PRIORITY_VOICE, "user said hello")
    await asyncio.wait_for(voice_started.wait(), timeout=1.0)

    arb_task.cancel()
    try:
        await arb_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_patrol_resumes_after_voice_completes():
    """Core regression: patrol must restart after a voice command ends naturally."""
    bus, state = _make_state()

    patrol_start_count = 0
    second_patrol = asyncio.Event()

    class _CountingPatrol(_RunForever):
        async def _run(self) -> None:
            nonlocal patrol_start_count
            patrol_start_count += 1
            if patrol_start_count >= 2:
                second_patrol.set()
            await super()._run()

    arbiter = Arbiter(bus, state, make_patrol=lambda: _CountingPatrol(bus, state))
    arb_task = asyncio.create_task(arbiter.run())

    # Let first patrol start
    await asyncio.sleep(0.05)

    # Voice command that finishes on its own (no cancel needed)
    arbiter.request(_RunOnce(bus, state), PRIORITY_VOICE, "go to the door")

    # Patrol must restart after voice finishes naturally
    await asyncio.wait_for(second_patrol.wait(), timeout=2.0)

    arb_task.cancel()
    try:
        await arb_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_obstacle_cancels_behavior_and_resumes_on_clear():
    bus, state = _make_state()
    second_patrol = asyncio.Event()
    patrol_count = 0

    class _CountingPatrol(_RunForever):
        async def _run(self) -> None:
            nonlocal patrol_count
            patrol_count += 1
            if patrol_count >= 2:
                second_patrol.set()
            await super()._run()

    arbiter = Arbiter(bus, state, make_patrol=lambda: _CountingPatrol(bus, state))
    arb_task = asyncio.create_task(arbiter.run())

    await asyncio.sleep(0.05)
    assert patrol_count == 1, "first patrol should be running"

    # Simulate obstacle
    bus.publish(ObstacleDetected(distance_cm=12.0))
    await asyncio.sleep(0.1)

    # Simulate clear
    bus.publish(ObstacleCleared(distance_cm=45.0))

    # Patrol must resume
    await asyncio.wait_for(second_patrol.wait(), timeout=2.0)

    arb_task.cancel()
    try:
        await arb_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_cleanup_runs_on_cancel():
    bus, state = _make_state()
    _SlowCleanup.cleaned_up = False

    arbiter = Arbiter(bus, state, make_patrol=lambda: _SlowCleanup(bus, state))
    arb_task = asyncio.create_task(arbiter.run())

    await asyncio.sleep(0.05)

    arb_task.cancel()
    try:
        await arb_task
    except asyncio.CancelledError:
        pass

    # Give cleanup time to finish (it's shielded, runs even after cancel)
    await asyncio.sleep(0.2)
    assert _SlowCleanup.cleaned_up
