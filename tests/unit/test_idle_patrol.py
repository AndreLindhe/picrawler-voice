"""
Unit tests for IdlePatrol.  No hardware — controller is fully mocked.

Scenarios covered:
  1. stand() called on entry
  2. initial_turn causes a turn before first forward step
  3. no initial_turn skips the opening turn
  4. after STEPS_BEFORE_TURN steps a turn eventually happens (probability test)
  5. sit() called in cleanup on normal cancel
  6. sit() called in cleanup when _run() raises unexpectedly
  7. full arbiter integration: patrol starts, voice preempts, patrol resumes
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from robot.behaviors.idle_patrol import IdlePatrol
from robot.brain.arbiter import PRIORITY_VOICE, Arbiter
from robot.core.bus import MessageBus
from robot.core.state import StateManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _yield(*a, **kw) -> None:
    """Yield to the event loop once.

    AsyncMock coroutines have no real await point, so a tight
    'while True: await mock()' loop never gives the event loop a chance to
    fire timers or deliver cancellations.  Attaching this as a side_effect
    restores cooperative scheduling without adding real I/O.
    """
    await asyncio.sleep(0)


def _make_ctrl() -> AsyncMock:
    ctrl = AsyncMock()
    # Every method used in IdlePatrol._run / _cleanup must yield.
    ctrl.stand.side_effect = _yield
    ctrl.sit.side_effect = _yield
    ctrl.forward.side_effect = _yield
    ctrl.do_action.side_effect = _yield
    return ctrl


def _make_patrol(initial_turn: bool = False, ctrl=None) -> tuple[IdlePatrol, AsyncMock]:
    bus = MessageBus()
    state = StateManager()
    if ctrl is None:
        ctrl = _make_ctrl()
    patrol = IdlePatrol(bus, state, controller=ctrl, initial_turn=initial_turn)
    return patrol, ctrl


async def _run_then_cancel(coro, *, after_s: float = 0.05) -> None:
    """Run coro as a task, cancel after `after_s` seconds, await completion.

    Because mock methods yield via asyncio.sleep(0), the event loop runs
    normally and timers fire on time.  Cleanup is synchronous inside the
    task's finally block, so once asyncio.wait reports the task done we
    know cleanup has finished.  The 2 s timeout is a safety net only.
    """
    task = asyncio.create_task(coro)
    await asyncio.sleep(after_s)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=2.0)
    if not done:
        task.cancel()  # force-cancel if still stuck (should never happen)


# ---------------------------------------------------------------------------
# 1. stand() on entry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stand_called_on_start():
    patrol, ctrl = _make_patrol()
    await _run_then_cancel(patrol.run())
    ctrl.stand.assert_called_once_with(speed=IdlePatrol.POSE_SPEED)


# ---------------------------------------------------------------------------
# 2 & 3. initial_turn flag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_initial_turn_true_calls_do_action_before_forward():
    patrol, ctrl = _make_patrol(initial_turn=True)

    call_order: list[str] = []

    async def record_stand(*a, **kw):
        call_order.append("stand")
        await asyncio.sleep(0)

    async def record_do_action(*a, **kw):
        call_order.append("do_action")
        await asyncio.sleep(0)

    async def record_forward(*a, **kw):
        call_order.append("forward")
        await asyncio.sleep(0)

    ctrl.stand.side_effect = record_stand
    ctrl.do_action.side_effect = record_do_action
    ctrl.forward.side_effect = record_forward

    await _run_then_cancel(patrol.run(), after_s=0.08)

    # stand → turn (do_action) → forward …
    assert call_order[0] == "stand"
    assert call_order[1] == "do_action"
    assert "forward" in call_order


@pytest.mark.asyncio
async def test_initial_turn_false_first_call_is_forward():
    patrol, ctrl = _make_patrol(initial_turn=False)

    call_order: list[str] = []

    async def record_stand(*a, **kw):
        call_order.append("stand")
        await asyncio.sleep(0)

    async def record_do_action(*a, **kw):
        call_order.append("do_action")
        await asyncio.sleep(0)

    async def record_forward(*a, **kw):
        call_order.append("forward")
        await asyncio.sleep(0)

    ctrl.stand.side_effect = record_stand
    ctrl.do_action.side_effect = record_do_action
    ctrl.forward.side_effect = record_forward

    await _run_then_cancel(patrol.run(), after_s=0.08)

    assert call_order[0] == "stand"
    assert call_order[1] == "forward"


# ---------------------------------------------------------------------------
# 4. Turning eventually happens after enough forward steps
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_turn_occurs_after_threshold_steps():
    """
    Force random.random() to return 0.0 (always below TURN_PROBABILITY) so the
    turn is deterministic.  After STEPS_BEFORE_TURN forward steps the next
    iteration should call do_action (a turn).
    """
    patrol, ctrl = _make_patrol(initial_turn=False)

    forward_calls = 0
    do_action_after_threshold = asyncio.Event()

    async def count_forward(*a, **kw):
        nonlocal forward_calls
        forward_calls += 1
        await asyncio.sleep(0)

    async def record_do_action(*a, **kw):
        if forward_calls >= IdlePatrol.STEPS_BEFORE_TURN:
            do_action_after_threshold.set()
        await asyncio.sleep(0)

    ctrl.forward.side_effect = count_forward
    ctrl.do_action.side_effect = record_do_action

    with patch("robot.behaviors.idle_patrol.random") as mock_random:
        # random.random() → always turns, random.choice() → first turn option
        mock_random.random.return_value = 0.0
        mock_random.choice.return_value = IdlePatrol._TURN_ACTIONS[0]

        task = asyncio.create_task(patrol.run())
        try:
            await asyncio.wait_for(do_action_after_threshold.wait(), timeout=2.0)
        finally:
            task.cancel()
            await asyncio.wait({task}, timeout=2.0)

    assert forward_calls >= IdlePatrol.STEPS_BEFORE_TURN


# ---------------------------------------------------------------------------
# 5. sit() called on clean cancel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sit_called_on_cancel():
    patrol, ctrl = _make_patrol()
    await _run_then_cancel(patrol.run())
    ctrl.sit.assert_called_once_with(speed=IdlePatrol.POSE_SPEED)


# ---------------------------------------------------------------------------
# 6. sit() called even when _run() raises an unexpected error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sit_called_on_unexpected_error():
    patrol, ctrl = _make_patrol()
    # RuntimeError raised synchronously — no yield needed here.
    ctrl.forward.side_effect = RuntimeError("servo jammed")

    task = asyncio.create_task(patrol.run())
    done, _ = await asyncio.wait({task}, timeout=2.0)
    # Task should have finished (error path) — cleanup is synchronous.
    assert done, "task did not finish within timeout"
    ctrl.sit.assert_called_once_with(speed=IdlePatrol.POSE_SPEED)


# ---------------------------------------------------------------------------
# 7. Integration: patrol → voice preempts → patrol resumes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_patrol_resumes_with_fresh_instance_after_voice():
    """
    Each patrol instance is freshly created by make_patrol.  After a voice
    command completes the arbiter calls make_patrol() again — stand() should
    be called at least twice (once per patrol instance).
    """
    bus = MessageBus()
    state = StateManager()

    patrol_ctrl = _make_ctrl()
    voice_ctrl = _make_ctrl()

    patrol_instances: list[IdlePatrol] = []

    def make_patrol() -> IdlePatrol:
        p = IdlePatrol(bus, state, controller=patrol_ctrl, initial_turn=False)
        patrol_instances.append(p)
        return p

    class _VoiceCmd(IdlePatrol):
        """A one-shot behavior that exits immediately."""
        async def _run(self) -> None:
            await asyncio.sleep(0)

        async def _cleanup(self) -> None:
            await voice_ctrl.sit(speed=IdlePatrol.POSE_SPEED)

    second_patrol_started = asyncio.Event()
    stand_call_count = 0

    async def track_stand(*a, **kw):
        nonlocal stand_call_count
        stand_call_count += 1
        if stand_call_count >= 2:
            second_patrol_started.set()
        await asyncio.sleep(0)

    patrol_ctrl.stand.side_effect = track_stand

    arbiter = Arbiter(bus, state, make_patrol=make_patrol)
    arb_task = asyncio.create_task(arbiter.run())

    # Let first patrol start
    await asyncio.sleep(0.05)

    # Preempt with a voice command that finishes naturally
    arbiter.request(_VoiceCmd(bus, state, controller=voice_ctrl), PRIORITY_VOICE, "test")

    # Wait for second patrol to start
    await asyncio.wait_for(second_patrol_started.wait(), timeout=2.0)

    arb_task.cancel()
    await asyncio.wait({arb_task}, timeout=2.0)

    # Two patrol instances created, stand called twice
    assert len(patrol_instances) == 2
    assert stand_call_count >= 2
