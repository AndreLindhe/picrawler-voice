"""
Unit tests for the safety loop — no hardware required.

The sonar and Picrawler are fully mocked so these run anywhere.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot.core.bus import MessageBus
from robot.core.events import ObstacleCleared, ObstacleDetected
from robot.core.state import StateManager
from robot.motor.safety import (
    CLEAR_THRESHOLD_CM,
    OBSTACLE_THRESHOLD_CM,
    SafetyLoop,
)
from robot.motor.primitives import resolve, ALL_ACTIONS, SYNONYM_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_safety(sonar_mock) -> tuple[SafetyLoop, MessageBus, StateManager]:
    bus = MessageBus()
    state = StateManager()
    loop = SafetyLoop(bus, state, sonar=sonar_mock)
    return loop, bus, state


async def _drain(bus: MessageBus, *topics: str, count: int, timeout: float = 1.0) -> list:
    events = []
    async with bus.subscribe(*topics) as q:
        for _ in range(count):
            events.append(await asyncio.wait_for(q.get(), timeout=timeout))
    return events


# ---------------------------------------------------------------------------
# Tests: obstacle detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_obstacle_detected_below_threshold():
    sonar = MagicMock()
    sonar.read.return_value = OBSTACLE_THRESHOLD_CM - 1  # just inside threshold

    loop, bus, _ = _make_safety(sonar)

    async with bus.subscribe(ObstacleDetected.topic) as q:
        await loop._evaluate(OBSTACLE_THRESHOLD_CM - 1)
        event = await asyncio.wait_for(q.get(), timeout=0.5)

    assert isinstance(event, ObstacleDetected)
    assert event.distance_cm == OBSTACLE_THRESHOLD_CM - 1


@pytest.mark.asyncio
async def test_no_event_when_already_blocked():
    sonar = MagicMock()
    loop, bus, _ = _make_safety(sonar)
    loop._blocked = True

    events = []
    async with bus.subscribe(ObstacleDetected.topic) as q:
        await loop._evaluate(OBSTACLE_THRESHOLD_CM - 5)
        # queue should be empty — no duplicate event
        await asyncio.sleep(0.05)
        while not q.empty():
            events.append(q.get_nowait())

    assert events == []


@pytest.mark.asyncio
async def test_clear_event_after_obstacle():
    sonar = MagicMock()
    loop, bus, _ = _make_safety(sonar)

    # First, trigger blocked state
    await loop._evaluate(OBSTACLE_THRESHOLD_CM - 1)
    assert loop._blocked

    async with bus.subscribe(ObstacleCleared.topic) as q:
        await loop._evaluate(CLEAR_THRESHOLD_CM + 1)
        event = await asyncio.wait_for(q.get(), timeout=0.5)

    assert isinstance(event, ObstacleCleared)
    assert not loop._blocked


@pytest.mark.asyncio
async def test_hysteresis_no_clear_below_threshold():
    sonar = MagicMock()
    loop, bus, _ = _make_safety(sonar)
    loop._blocked = True

    # Distance between OBSTACLE and CLEAR thresholds — should NOT clear
    mid = (OBSTACLE_THRESHOLD_CM + CLEAR_THRESHOLD_CM) / 2

    events = []
    async with bus.subscribe(ObstacleCleared.topic) as q:
        await loop._evaluate(mid)
        await asyncio.sleep(0.05)
        while not q.empty():
            events.append(q.get_nowait())

    assert events == []
    assert loop._blocked  # still blocked


@pytest.mark.asyncio
async def test_repeated_sensor_failure_triggers_obstacle():
    sonar = MagicMock()
    loop, bus, _ = _make_safety(sonar)

    # Simulate enough consecutive failures to hit the fail-safe threshold
    loop._failure_count = SafetyLoop._MAX_CONSECUTIVE_FAILURES  # type: ignore[attr-defined]

    async with bus.subscribe(ObstacleDetected.topic) as q:
        await loop._evaluate(None)
        event = await asyncio.wait_for(q.get(), timeout=0.5)

    assert isinstance(event, ObstacleDetected)


@pytest.mark.asyncio
async def test_filtered_read_returns_median():
    sonar = MagicMock()
    # Alternating good/bad values — median of [10, 30, 50] → 30
    sonar.read.side_effect = [10, 30, 50]

    loop, bus, _ = _make_safety(sonar)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await loop._read_filtered()

    assert result == 30.0


@pytest.mark.asyncio
async def test_filtered_read_returns_none_on_all_failures():
    sonar = MagicMock()
    sonar.read.return_value = -1  # -1 means timeout in robot_hat

    loop, bus, _ = _make_safety(sonar)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await loop._read_filtered()

    assert result is None
    assert loop._failure_count > 0


# ---------------------------------------------------------------------------
# Tests: primitives
# ---------------------------------------------------------------------------

def test_resolve_exact_action():
    assert resolve("forward") == "forward"
    assert resolve("sit") == "sit"


def test_resolve_synonym():
    assert resolve("stop") == "sit"
    assert resolve("go forward") == "forward"
    assert resolve("spin left") == "turn left angle"


def test_resolve_case_insensitive():
    assert resolve("FORWARD") == "forward"
    assert resolve("Turn Left") == "turn left"


def test_resolve_unknown_returns_none():
    assert resolve("do a backflip") is None
    assert resolve("") is None


def test_all_synonyms_map_to_valid_actions():
    for synonym, action in SYNONYM_MAP.items():
        assert action in ALL_ACTIONS, f"synonym {synonym!r} maps to unknown action {action!r}"
