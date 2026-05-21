from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "/home/penguin/people_registry"
_SIMILARITY_THRESHOLD = 0.4


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class PeopleRegistry:
    """
    Persists known face embeddings (ArcFace 512-dim vectors) alongside names.

    Storage layout:
        <base_path>/index.json   — list of {"name": str, "file": str}
        <base_path>/<uuid>.npy   — one embedding per person (averaged if multi)
    """

    def __init__(
        self,
        base_path: str = _DEFAULT_PATH,
        threshold: float = _SIMILARITY_THRESHOLD,
    ) -> None:
        self._path = Path(base_path)
        self._threshold = threshold
        # In-memory list of (name, embedding)
        self._entries: list[tuple[str, np.ndarray]] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """Save a face embedding for `name`, replacing any existing entry."""
        name = name.strip().title()
        # Remove existing entry for this name
        self._entries = [(n, e) for n, e in self._entries if n.lower() != name.lower()]
        self._entries.append((name, embedding.copy()))
        self._save()
        logger.info("registry: enrolled %r (%d people total)", name, len(self._entries))

    def find_match(
        self, embedding: np.ndarray
    ) -> Optional[tuple[str, float]]:
        """
        Return (name, similarity) for the best match above threshold,
        or None if no match is found.
        """
        best_name: Optional[str] = None
        best_sim = -1.0

        for name, stored in self._entries:
            sim = _cosine_similarity(embedding, stored)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_name is not None and best_sim >= self._threshold:
            return best_name, best_sim
        return None

    def known_names(self) -> list[str]:
        return [name for name, _ in self._entries]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        index_file = self._path / "index.json"
        if not index_file.exists():
            logger.info("registry: no existing registry at %s", self._path)
            return
        try:
            index = json.loads(index_file.read_text())
            for entry in index:
                npy_path = self._path / entry["file"]
                if npy_path.exists():
                    emb = np.load(str(npy_path))
                    self._entries.append((entry["name"], emb))
            logger.info("registry: loaded %d known people", len(self._entries))
        except Exception:
            logger.exception("registry: failed to load from %s", self._path)

    def _save(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        index = []
        for i, (name, emb) in enumerate(self._entries):
            filename = f"face_{i:04d}.npy"
            np.save(str(self._path / filename), emb)
            index.append({"name": name, "file": filename})
        (self._path / "index.json").write_text(json.dumps(index, indent=2))
        logger.debug("registry: saved %d entries", len(self._entries))
