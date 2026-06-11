# -*- coding: utf-8 -*-
"""
experience_replay.py — Cross-Module Experience Replay Buffer

Persists state-action-result trajectories to a shared JSON ledger so the
builder_brains sub-engines can learn from one another's history:

  - Thread-safe: every buffer operation is serialized behind a re-entrant
    lock, so concurrent worker threads inside one process never interleave
    read-modify-write cycles.
  - Process-safe: the JSON ledger is additionally guarded with explicit
    ``fcntl`` advisory file-descriptor locks (``LOCK_EX`` for read-modify-write
    cycles, ``LOCK_SH`` for reads), so cooperating evolve-loop processes
    cannot corrupt the file with partial writes.
  - Bounded: only the top ``max_size`` (default 200) highest-performing
    trajectories survive each write, ranked by their result score.

Primary consumer: ``decision_tree.FallbackRouter`` — when a
``Strategy.RANDOM_RESTART`` condition is met it queries this buffer for the
best matching historical trajectory parameter matrices and performs a
``WARM_BOOTSTRAP`` re-seed of the optimizer population pool instead of a
blind random restart.
"""

import fcntl
import json
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("builder_brains.experience_replay")

#: Default ledger location — kept beside this module so every process that
#: imports builder_brains shares one ledger regardless of working directory.
_DEFAULT_BUFFER_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experience_replay.json"
)

#: Hard ceiling on persisted trajectories (top performers only).
_DEFAULT_MAX_SIZE: int = 200

_READ_CHUNK_BYTES: int = 65536
_MAX_STATE_STRING_LENGTH: int = 256


class ExperienceReplayBuffer:
    """Thread-safe, process-safe persistent replay buffer.

    Each trajectory is a state-action-result triplet, optionally annotated
    with the hyperparameter matrix that produced it:

        {
          "state":            {<numeric / string snapshot of engine state>},
          "action":           "<action or strategy label>",
          "result":           <float performance score (higher is better)>,
          "parameter_matrix": {<hyperparameter configuration>},
          "recorded_at":      <unix timestamp>
        }

    The on-disk ledger is a JSON array of these triplets, sorted by ``result``
    descending and truncated to ``max_size`` on every write, so the file
    always holds the top historical trajectories.
    """

    def __init__(
        self,
        file_path: str = _DEFAULT_BUFFER_PATH,
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        self._file_path: str = str(file_path)
        self._max_size: int = max(1, int(max_size))
        self._lock = threading.RLock()
        self._ensure_file()

    # -- properties -----------------------------------------------------------

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def max_size(self) -> int:
        return self._max_size

    # -- sanitization helpers --------------------------------------------------

    @staticmethod
    def _sanitize_state(state: Any) -> Dict[str, Any]:
        """Reduce a state mapping to a JSON-safe snapshot of primitives."""
        if not isinstance(state, dict):
            return {}
        snapshot: Dict[str, Any] = {}
        for key, value in state.items():
            if isinstance(value, bool):
                snapshot[str(key)] = value
            elif isinstance(value, (int, float)):
                number = float(value)
                if math.isfinite(number):
                    snapshot[str(key)] = value if isinstance(value, int) else number
            elif isinstance(value, str):
                snapshot[str(key)] = value[:_MAX_STATE_STRING_LENGTH]
        return snapshot

    @staticmethod
    def _sanitize_matrix(matrix: Any) -> Dict[str, float]:
        """Keep only finite numeric entries from a parameter matrix."""
        if not isinstance(matrix, dict):
            return {}
        sanitized: Dict[str, float] = {}
        for key, value in matrix.items():
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                sanitized[str(key)] = number
        return sanitized

    @staticmethod
    def _coerce_result(result: Any) -> Optional[float]:
        if isinstance(result, bool):
            return 1.0 if result else 0.0
        try:
            number = float(result)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    @staticmethod
    def _result_sort_key(trajectory: Dict[str, Any]) -> float:
        value = trajectory.get("result")
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return float("-inf")

    @staticmethod
    def _parse_ledger(raw: bytes) -> List[Dict[str, Any]]:
        """Decode the on-disk ledger, tolerating corruption gracefully."""
        if not raw.strip():
            return []
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            logger.warning("Replay ledger corrupted (%s) — starting fresh", exc)
            return []
        if not isinstance(data, list):
            logger.warning("Replay ledger is not a JSON array — starting fresh")
            return []
        return [entry for entry in data if isinstance(entry, dict)]

    # -- file plumbing (explicit fcntl advisory fd locking) --------------------

    def _ensure_file(self) -> None:
        """Create the ledger atomically if it does not exist yet."""
        with self._lock:
            directory = os.path.dirname(self._file_path)
            if directory:
                try:
                    os.makedirs(directory, exist_ok=True)
                except OSError as exc:
                    logger.warning("Replay ledger directory unavailable: %s", exc)
                    return
            if os.path.exists(self._file_path):
                return
            try:
                # O_EXCL guards against a racing process creating it first.
                fd = os.open(
                    self._file_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                )
            except FileExistsError:
                return
            except OSError as exc:
                logger.warning("Replay ledger creation failed: %s", exc)
                return
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    os.write(fd, b"[]")
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @staticmethod
    def _read_fd(fd: int) -> bytes:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: List[bytes] = []
        while True:
            chunk = os.read(fd, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def _load_all(self) -> List[Dict[str, Any]]:
        """Read the full ledger under a shared advisory lock."""
        with self._lock:
            self._ensure_file()
            try:
                fd = os.open(self._file_path, os.O_RDONLY)
            except OSError as exc:
                logger.warning("Replay ledger open failed: %s", exc)
                return []
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                try:
                    raw = self._read_fd(fd)
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError as exc:
                logger.warning("Replay ledger read failed: %s", exc)
                return []
            finally:
                os.close(fd)
            return self._parse_ledger(raw)

    def _mutate_ledger(self, mutate: Any) -> bool:
        """Run one exclusive read-modify-write cycle against the ledger.

        ``mutate`` receives the decoded trajectory list and returns the list
        to persist. The entire cycle holds ``LOCK_EX`` on the ledger's file
        descriptor so concurrent processes serialize cleanly.
        """
        with self._lock:
            self._ensure_file()
            try:
                fd = os.open(self._file_path, os.O_RDWR | os.O_CREAT, 0o644)
            except OSError as exc:
                logger.warning("Replay ledger open failed: %s", exc)
                return False
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    trajectories = self._parse_ledger(self._read_fd(fd))
                    trajectories = mutate(trajectories)
                    payload = json.dumps(trajectories, indent=2).encode("utf-8")
                    os.lseek(fd, 0, os.SEEK_SET)
                    os.ftruncate(fd, 0)
                    os.write(fd, payload)
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError as exc:
                logger.warning("Replay ledger write failed: %s", exc)
                return False
            finally:
                os.close(fd)
            return True

    # -- recording --------------------------------------------------------------

    def record_trajectory(
        self,
        state: Any,
        action: Any,
        result: Any,
        parameter_matrix: Any = None,
    ) -> bool:
        """Persist one state-action-result triplet.

        Returns True when the triplet was durably written. Non-numeric or
        non-finite results are rejected (the ledger ranks by result, so an
        unrankable trajectory carries no replay value).
        """
        score = self._coerce_result(result)
        if score is None:
            logger.debug("Discarding trajectory with non-numeric result: %r", result)
            return False

        entry: Dict[str, Any] = {
            "state": self._sanitize_state(state),
            "action": str(action),
            "result": round(score, 6),
            "parameter_matrix": self._sanitize_matrix(parameter_matrix),
            "recorded_at": round(time.time(), 3),
        }

        def _apply(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            trajectories.append(entry)
            trajectories.sort(key=self._result_sort_key, reverse=True)
            return trajectories[: self._max_size]

        return self._mutate_ledger(_apply)

    def save_triplet(self, state: Any, action: Any, fitness_delta: Any) -> bool:
        """Backward-compatible alias for :meth:`record_trajectory`."""
        return self.record_trajectory(state, action, fitness_delta)

    # -- querying ----------------------------------------------------------------

    def top_trajectories(
        self,
        count: int = 1,
        action_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to ``count`` highest-result trajectories."""
        if count <= 0:
            return []
        trajectories = self._load_all()
        if action_filter is not None:
            trajectories = [
                t for t in trajectories if t.get("action") == action_filter
            ]
        trajectories.sort(key=self._result_sort_key, reverse=True)
        return trajectories[:count]

    def get_best_triplet(self) -> Optional[Dict[str, Any]]:
        """Backward-compatible alias: the single highest-result trajectory."""
        best = self.top_trajectories(count=1)
        return best[0] if best else None

    @staticmethod
    def _state_similarity(state_a: Dict[str, Any], state_b: Dict[str, Any]) -> float:
        """Similarity in [0, 1] over the keys two state snapshots share.

        Numeric values contribute a normalized closeness score (scale floor of
        1.0 guards the division), booleans and strings contribute equality.
        Disjoint snapshots score 0.0.
        """
        if not state_a or not state_b:
            return 0.0
        shared = set(state_a.keys()) & set(state_b.keys())
        if not shared:
            return 0.0
        total = 0.0
        for key in shared:
            value_a = state_a[key]
            value_b = state_b[key]
            if isinstance(value_a, bool) or isinstance(value_b, bool):
                total += 1.0 if bool(value_a) == bool(value_b) else 0.0
            elif isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                scale = max(abs(float(value_a)), abs(float(value_b)), 1.0)
                total += max(0.0, 1.0 - abs(float(value_a) - float(value_b)) / scale)
            else:
                total += 1.0 if str(value_a) == str(value_b) else 0.0
        return total / len(shared)

    def best_matching(
        self,
        state: Any,
        count: int = 1,
        action_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Trajectories ranked by similarity to ``state``, ties by result."""
        if count <= 0:
            return []
        target = self._sanitize_state(state)
        trajectories = self._load_all()
        if action_filter is not None:
            trajectories = [
                t for t in trajectories if t.get("action") == action_filter
            ]
        if not target:
            trajectories.sort(key=self._result_sort_key, reverse=True)
            return trajectories[:count]

        scored = [
            (
                self._state_similarity(target, t.get("state", {}) or {}),
                self._result_sort_key(t),
                t,
            )
            for t in trajectories
        ]
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [t for _, _, t in scored[:count]]

    def best_parameter_matrices(
        self,
        state: Any = None,
        count: int = 3,
    ) -> List[Dict[str, float]]:
        """Best matching historical trajectory parameter matrices.

        This is the WARM_BOOTSTRAP query surface: returns up to ``count``
        non-empty parameter matrices, ranked by state similarity when a query
        state is provided (falling back to raw result ranking otherwise).
        """
        if count <= 0:
            return []
        # Over-fetch so trajectories without matrices do not starve the result.
        candidates = (
            self.best_matching(state, count=max(count * 4, count))
            if state is not None
            else self.top_trajectories(count=max(count * 4, count))
        )
        matrices: List[Dict[str, float]] = []
        for trajectory in candidates:
            matrix = self._sanitize_matrix(trajectory.get("parameter_matrix"))
            if matrix:
                matrices.append(matrix)
            if len(matrices) >= count:
                break
        return matrices

    # -- maintenance ---------------------------------------------------------------

    def size(self) -> int:
        return len(self._load_all())

    def clear(self) -> bool:
        """Erase every stored trajectory (exclusive lock held throughout)."""
        return self._mutate_ledger(lambda _trajectories: [])

    def stats(self) -> Dict[str, Any]:
        trajectories = self._load_all()
        results = [
            self._result_sort_key(t)
            for t in trajectories
            if self._result_sort_key(t) != float("-inf")
        ]
        return {
            "size": len(trajectories),
            "max_size": self._max_size,
            "file_path": self._file_path,
            "best_result": round(max(results), 6) if results else 0.0,
            "mean_result": round(sum(results) / len(results), 6) if results else 0.0,
            "with_parameter_matrix": sum(
                1 for t in trajectories if t.get("parameter_matrix")
            ),
        }
