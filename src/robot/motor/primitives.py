from __future__ import annotations

"""
Named action constants and a lookup table the LLM tools layer can use to
translate natural-language intent into controller calls.

Nothing here touches hardware — import freely without side effects.
"""

# Every string accepted by Picrawler.do_action() or .do_step()
FORWARD = "forward"
BACKWARD = "backward"
TURN_LEFT = "turn left"
TURN_RIGHT = "turn right"
TURN_LEFT_ANGLE = "turn left angle"
TURN_RIGHT_ANGLE = "turn right angle"
LOOK_LEFT = "look left"
LOOK_RIGHT = "look right"
LOOK_UP = "look up"
LOOK_DOWN = "look down"
WAVE = "wave"
PUSH_UP = "push up"
DANCE = "dance"
STAND = "stand"
SIT = "sit"

# Actions that involve locomotion (as opposed to expressive/postural moves).
LOCOMOTION_ACTIONS: frozenset[str] = frozenset({
    FORWARD, BACKWARD,
    TURN_LEFT, TURN_RIGHT,
    TURN_LEFT_ANGLE, TURN_RIGHT_ANGLE,
})

# All valid action names — useful for input validation in tool handlers.
ALL_ACTIONS: frozenset[str] = frozenset({
    FORWARD, BACKWARD,
    TURN_LEFT, TURN_RIGHT,
    TURN_LEFT_ANGLE, TURN_RIGHT_ANGLE,
    LOOK_LEFT, LOOK_RIGHT, LOOK_UP, LOOK_DOWN,
    WAVE, PUSH_UP, DANCE,
    STAND, SIT,
})

# Maps natural-language synonyms → canonical action names.
# The brain/tools layer uses this to resolve LLM output.
SYNONYM_MAP: dict[str, str] = {
    "go forward":      FORWARD,
    "move forward":    FORWARD,
    "advance":         FORWARD,
    "go back":         BACKWARD,
    "go backward":     BACKWARD,
    "reverse":         BACKWARD,
    "retreat":         BACKWARD,
    "left":            TURN_LEFT,
    "turn left":       TURN_LEFT,
    "rotate left":     TURN_LEFT_ANGLE,
    "spin left":       TURN_LEFT_ANGLE,
    "right":           TURN_RIGHT,
    "turn right":      TURN_RIGHT,
    "rotate right":    TURN_RIGHT_ANGLE,
    "spin right":      TURN_RIGHT_ANGLE,
    "look left":       LOOK_LEFT,
    "look right":      LOOK_RIGHT,
    "look up":         LOOK_UP,
    "look down":       LOOK_DOWN,
    "wave":            WAVE,
    "wave hello":      WAVE,
    "push up":         PUSH_UP,
    "dance":           DANCE,
    "stand":           STAND,
    "stand up":        STAND,
    "sit":             SIT,
    "sit down":        SIT,
    "stop":            SIT,
    "halt":            SIT,
    "freeze":          SIT,
}


def resolve(intent: str) -> str | None:
    """
    Return the canonical action name for a natural-language intent string,
    or None if unrecognised.

    Case-insensitive.  Tries exact match first, then synonym lookup.
    """
    normalised = intent.strip().lower()
    if normalised in ALL_ACTIONS:
        return normalised
    return SYNONYM_MAP.get(normalised)
