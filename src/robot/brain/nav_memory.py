from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    goal        TEXT    NOT NULL,
    situation   TEXT    NOT NULL,
    sonar_range TEXT    NOT NULL,
    actions     TEXT    NOT NULL,   -- JSON array of strings
    success     INTEGER NOT NULL,   -- 1 = success, 0 = failure
    summary     TEXT    NOT NULL
);
"""


class NavMemory:
    """
    Stores navigation episodes to disk and recalls relevant ones by situation.

    Similarity in Phase 1 is coarse: match on sonar_range bucket.
    Phase 2 can extend this with object-label matching without schema changes
    (store extra keys in the situation JSON string).
    """

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()
        logger.info("nav_memory: opened %s", db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_episode(
        self,
        *,
        situation: str,
        sonar_range: str,
        goal: str,
        actions: list[str],
        success: bool,
        summary: str = "",
    ) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO episodes
                   (timestamp, goal, situation, sonar_range, actions, success, summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    goal,
                    situation,
                    sonar_range,
                    json.dumps(actions),
                    int(success),
                    summary,
                ),
            )
        logger.debug("nav_memory: saved episode goal=%r success=%s", goal, success)

    def recall(
        self,
        sonar_range: str,
        goal: str,
        n: int = 3,
    ) -> list[dict]:
        """
        Return the N most recent episodes with a matching sonar_range.
        Falls back to any sonar_range if nothing matches.
        """
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """SELECT situation, actions, success, summary
                   FROM episodes
                   WHERE sonar_range = ?
                   ORDER BY id DESC LIMIT ?""",
                (sonar_range, n),
            ).fetchall()

            if not rows:
                rows = conn.execute(
                    """SELECT situation, actions, success, summary
                       FROM episodes
                       ORDER BY id DESC LIMIT ?""",
                    (n,),
                ).fetchall()

        return [
            {
                "situation": r[0],
                "actions": json.loads(r[1]),
                "success": bool(r[2]),
                "summary": r[3],
            }
            for r in rows
        ]

    def format_for_prompt(self, episodes: list[dict]) -> str:
        """Convert recalled episodes into a compact prompt-ready string."""
        if not episodes:
            return "No relevant past experiences yet."
        lines = []
        for i, ep in enumerate(episodes, 1):
            outcome = "succeeded" if ep["success"] else "failed"
            actions = ", ".join(ep["actions"]) if ep["actions"] else "no actions"
            lines.append(f"{i}. {ep['situation']} → {actions} → {outcome}")
            if ep["summary"]:
                lines.append(f"   note: {ep['summary']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(_SCHEMA)
