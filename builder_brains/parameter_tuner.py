# -*- coding: utf-8 -*-
"""
parameter_tuner.py — Dynamic Hyperparameter Tuning & Multi-Objective Evolutionary Engine

Implements:
  - Dynamic learning rate annealing curves (cosine warm-restart, exponential decay,
    step-decay, linear warmup)
  - Hyperparameter configuration matrices with bounded search spaces
  - Multi-objective Pareto frontier optimization (NSGA-II style mechanics):
    fast non-dominated sorting, crowding-distance diversity preservation,
    crowded tournament selection, and exact hypervolume progress tracking
  - Frontier-preserving elitism: the entire non-dominated pool survives across
    generational steps, protecting hyper-fast, hyper-accurate, and
    hyper-compact trade-off configurations alike

Fitness is a vector, not a scalar:

    F = [accuracy_score, inverse_wall_seconds (1 / execution_time), compression_ratio]

All three objectives are maximized. Individual A dominates Individual B when A
is no worse than B in every objective and strictly better in at least one.

Pipeline entry point: evaluate(metadata, hyper_params)
"""

import json
import logging
import math
import os
import random
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("builder_brains.parameter_tuner")

_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "build_manifest.json")


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def _load_manifest() -> Dict[str, Any]:
    try:
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load build_manifest.json: %s — using defaults", exc)
        return {}


def _get_tuner_params(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("hyperparameter_weights", {}).get("parameter_tuner", {})


def _get_thresholds(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("thresholds", {})


def _get_cost_ceilings(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("execution_cost_ceilings", {})


# ---------------------------------------------------------------------------
# Multi-objective fitness model
# ---------------------------------------------------------------------------

#: Objective vector layout (all objectives are maximized).
OBJECTIVE_NAMES: Tuple[str, str, str] = (
    "accuracy_score",
    "inverse_wall_seconds",
    "compression_ratio",
)
NUM_OBJECTIVES: int = len(OBJECTIVE_NAMES)

#: Saturation value for the speed objective. A measured wall time of exactly
#: zero seconds (or one small enough to overflow the inverse) maps to this cap
#: instead of raising ZeroDivisionError / producing infinity.
_INVERSE_WALL_CAP: float = 1.0e9

#: Tolerances for deduplicating re-measured configurations in the archive.
_VECTOR_TOLERANCE: float = 1e-9
_CONFIG_TOLERANCE: float = 1e-9

_ACCURACY_KEYS: Tuple[str, ...] = (
    "accuracy_score", "accuracy", "score", "success_rate", "precision",
)
_WALL_SECONDS_KEYS: Tuple[str, ...] = (
    "wall_seconds", "execution_time", "execution_seconds",
    "elapsed_seconds", "duration_seconds", "runtime_seconds",
)
_WALL_MILLIS_KEYS: Tuple[str, ...] = ("latency_ms", "wall_ms", "duration_ms")
_WALL_MICROS_KEYS: Tuple[str, ...] = ("latency_us", "wall_us", "duration_us")
_COMPRESSION_KEYS: Tuple[str, ...] = (
    "compression_ratio", "compression", "compaction_ratio", "minify_compression",
)


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort float coercion. Returns None for non-numeric or NaN input."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def safe_inverse_wall_seconds(wall_seconds: Any, cap: float = _INVERSE_WALL_CAP) -> float:
    """Convert an execution time into the inverse-wall-seconds objective.

    Division-by-zero, negative, missing, infinite, and non-numeric inputs are
    handled gracefully: a zero duration saturates at ``cap`` (an effectively
    instantaneous run), while invalid or infinite durations collapse to 0.0
    (the worst possible speed score).
    """
    seconds = _coerce_float(wall_seconds)
    if seconds is None or seconds < 0.0 or math.isinf(seconds):
        return 0.0
    if seconds == 0.0:
        return cap
    return min(cap, 1.0 / seconds)


def coerce_objective_vector(value: Any) -> List[float]:
    """Normalize arbitrary input into a non-negative objective vector.

    Scalars become accuracy-only vectors, short sequences are zero-padded,
    long sequences are truncated, and non-finite or negative elements are
    clamped so that downstream domination checks never see NaN/inf surprises.
    """
    if isinstance(value, (list, tuple)):
        raw: List[Any] = list(value)[:NUM_OBJECTIVES]
    elif value is None:
        raw = []
    else:
        raw = [value]

    vector: List[float] = []
    for element in raw:
        number = _coerce_float(element)
        if number is None or number < 0.0:
            vector.append(0.0)
        elif math.isinf(number):
            vector.append(_INVERSE_WALL_CAP)
        else:
            vector.append(min(number, _INVERSE_WALL_CAP))
    while len(vector) < NUM_OBJECTIVES:
        vector.append(0.0)
    return vector


def measurement_to_objectives(
    accuracy: Any,
    wall_seconds: Any,
    compression: Any,
) -> List[float]:
    """Map raw measurements (accuracy, execution time, compression) to F."""
    accuracy_value = _coerce_float(accuracy)
    compression_value = _coerce_float(compression)
    return coerce_objective_vector([
        max(0.0, accuracy_value) if accuracy_value is not None else 0.0,
        safe_inverse_wall_seconds(wall_seconds),
        max(0.0, compression_value) if compression_value is not None else 0.0,
    ])


def _first_numeric(record: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in record:
            value = _coerce_float(record[key])
            if value is not None:
                return value
    return None


def parse_performance_record(record: Any) -> List[float]:
    """Translate one performance-log entry into an objective vector.

    Dict records may carry accuracy under several aliases (or a boolean
    ``success`` flag), execution time in seconds / milliseconds / microseconds,
    and a compression ratio. Sequence records are interpreted as raw
    ``(accuracy, wall_seconds, compression_ratio)`` measurement rows, and bare
    scalars as accuracy-only measurements.
    """
    if isinstance(record, dict):
        accuracy = _first_numeric(record, _ACCURACY_KEYS)
        if accuracy is None and "success" in record:
            accuracy = 1.0 if record.get("success") else 0.0

        wall_seconds = _first_numeric(record, _WALL_SECONDS_KEYS)
        if wall_seconds is None:
            millis = _first_numeric(record, _WALL_MILLIS_KEYS)
            wall_seconds = millis / 1e3 if millis is not None else None
        if wall_seconds is None:
            micros = _first_numeric(record, _WALL_MICROS_KEYS)
            wall_seconds = micros / 1e6 if micros is not None else None

        compression = _first_numeric(record, _COMPRESSION_KEYS)
        return measurement_to_objectives(accuracy, wall_seconds, compression)

    if isinstance(record, (list, tuple)):
        padded = list(record) + [None] * NUM_OBJECTIVES
        return measurement_to_objectives(padded[0], padded[1], padded[2])

    return measurement_to_objectives(record, None, None)


def _synthetic_objective_vector() -> List[float]:
    """Exploration filler for population members without a measurement yet."""
    return measurement_to_objectives(
        random.uniform(0.3, 0.9),
        random.uniform(0.05, 2.5),
        random.uniform(0.1, 0.9),
    )


def collect_objective_vectors(
    metadata: Dict[str, Any],
    count: int,
) -> Tuple[List[List[float]], Dict[str, int]]:
    """Assemble one objective vector per population member (row i -> member i).

    Sources, in priority order:
      1. ``metadata["performance_log"]`` — raw measurement records
      2. ``metadata["fitness_matrix"]`` / ``metadata["objective_matrix"]`` —
         rows of ``(accuracy, wall_seconds, compression_ratio)`` measurements
      3. ``metadata["fitness_scores"]`` — legacy list; scalar entries are
         interpreted as accuracy-only measurements
      4. synthetic vectors — fill for members without measurements
    """
    sources = {
        "performance_log": 0,
        "fitness_matrix": 0,
        "legacy_scalars": 0,
        "synthetic": 0,
    }
    vectors: List[List[float]] = []
    if count <= 0:
        return vectors, sources

    log = metadata.get("performance_log")
    if isinstance(log, (list, tuple)):
        for record in list(log)[:count]:
            vectors.append(parse_performance_record(record))
            sources["performance_log"] += 1

    if len(vectors) < count:
        for key in ("fitness_matrix", "objective_matrix"):
            rows = metadata.get(key)
            if isinstance(rows, (list, tuple)):
                for row in list(rows)[: count - len(vectors)]:
                    if isinstance(row, (list, tuple)):
                        padded = list(row) + [None] * NUM_OBJECTIVES
                        vectors.append(
                            measurement_to_objectives(padded[0], padded[1], padded[2])
                        )
                        sources["fitness_matrix"] += 1
            if len(vectors) >= count:
                break

    if len(vectors) < count:
        legacy = metadata.get("fitness_scores")
        if isinstance(legacy, (list, tuple)):
            for entry in list(legacy)[: count - len(vectors)]:
                if isinstance(entry, (list, tuple)):
                    padded = list(entry) + [None] * NUM_OBJECTIVES
                    vectors.append(
                        measurement_to_objectives(padded[0], padded[1], padded[2])
                    )
                    sources["fitness_matrix"] += 1
                else:
                    vectors.append(measurement_to_objectives(entry, None, None))
                    sources["legacy_scalars"] += 1

    while len(vectors) < count:
        vectors.append(_synthetic_objective_vector())
        sources["synthetic"] += 1

    return vectors[:count], sources


def _sanitize_config(config: Any) -> Dict[str, float]:
    """Keep only finite numeric entries from an untrusted config mapping."""
    if not isinstance(config, dict):
        return {}
    sanitized: Dict[str, float] = {}
    for key, value in config.items():
        number = _coerce_float(value)
        if number is not None and math.isfinite(number):
            sanitized[str(key)] = number
    return sanitized


def _vectors_close(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
    tolerance: float = _VECTOR_TOLERANCE,
) -> bool:
    if len(vector_a) != len(vector_b):
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(vector_a, vector_b))


def _configs_close(
    config_a: Dict[str, float],
    config_b: Dict[str, float],
    tolerance: float = _CONFIG_TOLERANCE,
) -> bool:
    if set(config_a.keys()) != set(config_b.keys()):
        return False
    return all(abs(config_a[key] - config_b[key]) <= tolerance for key in config_a)


# ---------------------------------------------------------------------------
# Learning rate annealing curves
# ---------------------------------------------------------------------------

class AnnealingSchedule:
    """Collection of learning-rate schedule functions.

    All methods return the annealed learning rate for a given cycle index.
    """

    def __init__(
        self,
        initial_lr: float = 0.01,
        min_lr: float = 0.00001,
        warmup_cycles: int = 50,
        restart_period: int = 100,
        restart_multiplier: float = 2.0,
    ) -> None:
        self.initial_lr: float = max(min_lr, initial_lr)
        self.min_lr: float = min_lr
        self.warmup_cycles: int = max(0, warmup_cycles)
        self.restart_period: int = max(1, restart_period)
        self.restart_multiplier: float = max(1.0, restart_multiplier)
        self._current_period: int = self.restart_period
        self._cycle_in_period: int = 0
        self._restart_count: int = 0

    def _linear_warmup(self, cycle: int) -> float:
        if self.warmup_cycles <= 0 or cycle >= self.warmup_cycles:
            return self.initial_lr
        return self.min_lr + (self.initial_lr - self.min_lr) * (cycle / self.warmup_cycles)

    def cosine_warm_restart(self, cycle: int) -> float:
        if cycle < self.warmup_cycles:
            return self._linear_warmup(cycle)

        effective_cycle = cycle - self.warmup_cycles
        period = self.restart_period
        cumulative = 0
        restart_idx = 0

        while cumulative + period <= effective_cycle:
            cumulative += period
            restart_idx += 1
            period = int(period * self.restart_multiplier)

        cycle_in_period = effective_cycle - cumulative
        fraction = cycle_in_period / max(period, 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * fraction))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor

        return max(self.min_lr, lr)

    def exponential_decay(self, cycle: int, decay_rate: float = 0.995) -> float:
        if cycle < self.warmup_cycles:
            return self._linear_warmup(cycle)
        effective = cycle - self.warmup_cycles
        lr = self.initial_lr * (decay_rate ** effective)
        return max(self.min_lr, lr)

    def step_decay(
        self,
        cycle: int,
        drop_factor: float = 0.5,
        drop_every: int = 100,
    ) -> float:
        if cycle < self.warmup_cycles:
            return self._linear_warmup(cycle)
        effective = cycle - self.warmup_cycles
        drops = effective // max(1, drop_every)
        lr = self.initial_lr * (drop_factor ** drops)
        return max(self.min_lr, lr)

    def get_lr(self, cycle: int, schedule_name: str = "cosine_warm_restart") -> float:
        dispatch: Dict[str, Any] = {
            "cosine_warm_restart": self.cosine_warm_restart,
            "exponential_decay": self.exponential_decay,
            "step_decay": self.step_decay,
        }
        fn = dispatch.get(schedule_name, self.cosine_warm_restart)
        return fn(cycle)


# ---------------------------------------------------------------------------
# Hyperparameter configuration matrix
# ---------------------------------------------------------------------------

class ParameterBound:
    """Defines the search boundary for a single hyperparameter."""

    __slots__ = ("name", "low", "high", "log_scale", "dtype")

    def __init__(
        self,
        name: str,
        low: float,
        high: float,
        log_scale: bool = False,
        dtype: str = "float",
    ) -> None:
        self.name: str = name
        self.low: float = min(low, high)
        self.high: float = max(low, high)
        self.log_scale: bool = log_scale
        self.dtype: str = dtype

    def sample(self) -> float:
        if self.log_scale and self.low > 0:
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            value = math.exp(random.uniform(log_low, log_high))
        else:
            value = random.uniform(self.low, self.high)

        if self.dtype == "int":
            value = float(round(value))
        return value

    def clip(self, value: float) -> float:
        clipped = max(self.low, min(self.high, value))
        if self.dtype == "int":
            clipped = float(round(clipped))
        return clipped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "log_scale": self.log_scale,
            "dtype": self.dtype,
        }


class HyperparameterMatrix:
    """A bounded search space defined by a matrix of parameter constraints."""

    def __init__(self) -> None:
        self._bounds: Dict[str, ParameterBound] = {}

    def add_parameter(
        self,
        name: str,
        low: float,
        high: float,
        log_scale: bool = False,
        dtype: str = "float",
    ) -> None:
        self._bounds[name] = ParameterBound(
            name=name, low=low, high=high, log_scale=log_scale, dtype=dtype
        )

    def sample_configuration(self) -> Dict[str, float]:
        return {name: bound.sample() for name, bound in self._bounds.items()}

    def clip_configuration(self, config: Dict[str, float]) -> Dict[str, float]:
        clipped: Dict[str, float] = {}
        for name, value in config.items():
            if name in self._bounds:
                clipped[name] = self._bounds[name].clip(value)
            else:
                clipped[name] = value
        return clipped

    def parameter_names(self) -> List[str]:
        return list(self._bounds.keys())

    @property
    def dimension(self) -> int:
        return len(self._bounds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "parameters": {n: b.to_dict() for n, b in self._bounds.items()},
        }


def build_default_matrix() -> HyperparameterMatrix:
    matrix = HyperparameterMatrix()
    matrix.add_parameter("learning_rate", 1e-5, 1.0, log_scale=True)
    matrix.add_parameter("mutation_sigma", 0.001, 0.5)
    matrix.add_parameter("crossover_probability", 0.1, 0.95)
    matrix.add_parameter("population_size", 8, 256, dtype="int")
    matrix.add_parameter("tournament_k", 2, 10, dtype="int")
    matrix.add_parameter("elite_ratio", 0.05, 0.50)
    matrix.add_parameter("exploration_epsilon", 0.01, 0.50)
    matrix.add_parameter("discount_gamma", 0.80, 0.9999)
    return matrix


# ---------------------------------------------------------------------------
# Individual (vector-valued fitness)
# ---------------------------------------------------------------------------

class Individual:
    """A single candidate in the evolutionary population.

    ``fitness`` is the maximized objective vector
    ``[accuracy_score, inverse_wall_seconds, compression_ratio]``.
    ``rank`` (non-domination front index, 0 = Pareto-optimal) and
    ``crowding_distance`` are scratch values reassigned by each
    non-dominated sort.
    """

    __slots__ = (
        "config", "fitness", "rank", "crowding_distance",
        "generation", "parent_id", "uid",
    )

    _uid_counter: int = 0

    def __init__(
        self,
        config: Dict[str, float],
        fitness: Optional[Sequence[float]] = None,
        generation: int = 0,
        parent_id: Optional[int] = None,
    ) -> None:
        Individual._uid_counter += 1
        self.uid: int = Individual._uid_counter
        self.config: Dict[str, float] = dict(config)
        self.fitness: List[float] = coerce_objective_vector(fitness)
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.generation: int = generation
        self.parent_id: Optional[int] = parent_id

    @property
    def objectives(self) -> Dict[str, float]:
        return dict(zip(OBJECTIVE_NAMES, self.fitness))

    def clone(self, generation: int) -> "Individual":
        """Copy this individual into a new generation, preserving its fitness."""
        child = Individual(
            config=dict(self.config),
            fitness=list(self.fitness),
            generation=generation,
            parent_id=self.uid,
        )
        child.rank = self.rank
        child.crowding_distance = self.crowding_distance
        return child

    def to_dict(self) -> Dict[str, Any]:
        crowding = self.crowding_distance
        return {
            "uid": self.uid,
            "config": {k: round(v, 6) for k, v in self.config.items()},
            "fitness": [round(v, 6) for v in self.fitness],
            "objectives": {k: round(v, 6) for k, v in self.objectives.items()},
            "rank": self.rank,
            "crowding_distance": round(crowding, 6) if math.isfinite(crowding) else "inf",
            "generation": self.generation,
            "parent_id": self.parent_id,
        }


# ---------------------------------------------------------------------------
# Pareto dominance, non-dominated sorting & diversity machinery
# ---------------------------------------------------------------------------

def dominates(vector_a: Sequence[float], vector_b: Sequence[float]) -> bool:
    """Pareto domination for maximized objectives.

    A dominates B iff A is no worse than B in every objective and strictly
    better in at least one.
    """
    length = min(len(vector_a), len(vector_b))
    if length == 0:
        return False
    strictly_better = False
    for idx in range(length):
        if vector_a[idx] < vector_b[idx]:
            return False
        if vector_a[idx] > vector_b[idx]:
            strictly_better = True
    return strictly_better


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """NSGA-II fast non-dominated sort.

    Partitions the population into fronts and assigns ``rank`` in place:
    front 0 holds the Pareto-optimal individuals, front 1 those dominated
    only by front 0, and so on. Duplicate object references are collapsed.
    """
    unique: Dict[int, Individual] = {ind.uid: ind for ind in population}
    members = list(unique.values())
    if not members:
        return []

    dominated_by: Dict[int, List[Individual]] = {ind.uid: [] for ind in members}
    domination_count: Dict[int, int] = {ind.uid: 0 for ind in members}
    fronts: List[List[Individual]] = [[]]

    for candidate in members:
        for other in members:
            if candidate.uid == other.uid:
                continue
            if dominates(candidate.fitness, other.fitness):
                dominated_by[candidate.uid].append(other)
            elif dominates(other.fitness, candidate.fitness):
                domination_count[candidate.uid] += 1
        if domination_count[candidate.uid] == 0:
            candidate.rank = 0
            fronts[0].append(candidate)

    front_idx = 0
    while front_idx < len(fronts) and fronts[front_idx]:
        next_front: List[Individual] = []
        for candidate in fronts[front_idx]:
            for dominated in dominated_by[candidate.uid]:
                domination_count[dominated.uid] -= 1
                if domination_count[dominated.uid] == 0:
                    dominated.rank = front_idx + 1
                    next_front.append(dominated)
        front_idx += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def assign_crowding_distance(front: List[Individual]) -> None:
    """NSGA-II crowding distance assignment (in place, per front).

    Boundary individuals of every objective receive infinite distance so
    extreme trade-offs (hyper-fast / hyper-accurate / hyper-compact) are
    always preferred for preservation. Degenerate objectives whose values
    are identical across the front contribute nothing (guarding the
    normalization against division by zero).
    """
    count = len(front)
    if count == 0:
        return
    for individual in front:
        individual.crowding_distance = 0.0
    if count <= 2:
        for individual in front:
            individual.crowding_distance = math.inf
        return

    for objective_idx in range(NUM_OBJECTIVES):
        ordered = sorted(front, key=lambda ind: ind.fitness[objective_idx])
        ordered[0].crowding_distance = math.inf
        ordered[-1].crowding_distance = math.inf
        span = ordered[-1].fitness[objective_idx] - ordered[0].fitness[objective_idx]
        if span <= 0.0:
            continue
        for position in range(1, count - 1):
            individual = ordered[position]
            if math.isinf(individual.crowding_distance):
                continue
            gap = (
                ordered[position + 1].fitness[objective_idx]
                - ordered[position - 1].fitness[objective_idx]
            )
            individual.crowding_distance += gap / span


def crowded_comparison_key(individual: Individual) -> Tuple[int, float]:
    """Sort key implementing the NSGA-II crowded-comparison operator.

    Lower is better: prefer lower non-domination rank, then larger crowding
    distance (more isolated, hence more diverse) within the same rank.
    """
    return (individual.rank, -individual.crowding_distance)


def _hypervolume_2d(
    points: List[Tuple[float, float]],
    reference: Tuple[float, float],
) -> float:
    ref_x, ref_y = reference
    candidates = [(x, y) for x, y in points if x > ref_x and y > ref_y]
    if not candidates:
        return 0.0
    candidates.sort(key=lambda point: (-point[0], -point[1]))
    volume = 0.0
    best_y = ref_y
    for x, y in candidates:
        if y > best_y:
            volume += (x - ref_x) * (y - best_y)
            best_y = y
    return volume


def hypervolume(points: List[Sequence[float]], reference: Sequence[float]) -> float:
    """Exact hypervolume indicator for maximized objectives (1-3 dimensions).

    Measures the volume of objective space dominated by ``points`` relative to
    ``reference``. Monotone under Pareto improvement, which makes it a sound
    scalar progress signal for a multi-objective archive. The 3-D case uses
    the standard slicing sweep over the third objective.
    """
    if not points:
        return 0.0
    dimension = min(len(reference), min(len(point) for point in points))
    if dimension <= 0:
        return 0.0
    if dimension == 1:
        best = max(point[0] for point in points)
        return max(0.0, best - reference[0])
    if dimension == 2:
        return _hypervolume_2d(
            [(point[0], point[1]) for point in points],
            (reference[0], reference[1]),
        )

    ref_x, ref_y, ref_z = reference[0], reference[1], reference[2]
    candidates = [
        (point[0], point[1], point[2])
        for point in points
        if point[0] > ref_x and point[1] > ref_y and point[2] > ref_z
    ]
    if not candidates:
        return 0.0
    candidates.sort(key=lambda point: -point[2])

    volume = 0.0
    slab_points: List[Tuple[float, float]] = []
    for idx, (x, y, z) in enumerate(candidates):
        slab_points.append((x, y))
        next_z = candidates[idx + 1][2] if idx + 1 < len(candidates) else ref_z
        slab_height = z - next_z
        if slab_height <= 0.0:
            continue
        volume += _hypervolume_2d(slab_points, (ref_x, ref_y)) * slab_height
    return volume


def select_compromise(front: List[Individual]) -> Optional[Individual]:
    """Pick the balanced trade-off member of a Pareto front.

    Returns the individual closest (Euclidean) to the ideal point after
    min-max normalizing each objective across the front. Objectives with zero
    spread are skipped, guarding the normalization against division by zero.
    This is a reporting convenience only — selection pressure inside the
    engine remains purely Pareto-based.
    """
    if not front:
        return None
    mins = [min(ind.fitness[idx] for ind in front) for idx in range(NUM_OBJECTIVES)]
    maxs = [max(ind.fitness[idx] for ind in front) for idx in range(NUM_OBJECTIVES)]

    best_member: Optional[Individual] = None
    best_distance = math.inf
    for individual in front:
        distance_sq = 0.0
        informative = 0
        for idx in range(NUM_OBJECTIVES):
            span = maxs[idx] - mins[idx]
            if span <= 0.0:
                continue
            normalized = (individual.fitness[idx] - mins[idx]) / span
            distance_sq += (1.0 - normalized) ** 2
            informative += 1
        distance = math.sqrt(distance_sq) if informative else 0.0
        if distance < best_distance:
            best_distance = distance
            best_member = individual
    return best_member


def objective_champions(front: List[Individual]) -> Dict[str, Dict[str, Any]]:
    """Per-objective champions: the frontier member maximizing each objective."""
    champions: Dict[str, Dict[str, Any]] = {}
    for idx, name in enumerate(OBJECTIVE_NAMES):
        champion: Optional[Individual] = None
        for individual in front:
            if champion is None or individual.fitness[idx] > champion.fitness[idx]:
                champion = individual
        if champion is not None:
            champions[name] = champion.to_dict()
    return champions


# ---------------------------------------------------------------------------
# Pareto frontier survival tracking
# ---------------------------------------------------------------------------

class FitnessSurvivalTracker:
    """Track Pareto-frontier survival and progress across generations.

    Replaces the legacy scalar elite tracker with non-dominated archive
    mechanics:

      - Maintains a persistent archive holding the current Pareto frontier;
        elitism preserves this entire non-dominated pool across generational
        steps (crowding-distance truncation applies only beyond
        ``archive_capacity``, keeping objective-boundary extremes first).
      - Re-measured configurations replace their stale archive twins, and
        newly dominated members are evicted on every merge.
      - Progress is measured with the exact hypervolume indicator; an EMA of
        the hypervolume supports convergence checks, and stagnation is
        declared when neither the hypervolume improves by
        ``min_improvement_delta`` nor a novel non-dominated objective vector
        is admitted for ``stagnation_patience`` consecutive generations.
    """

    def __init__(
        self,
        archive_capacity: int = 64,
        history_window: int = 100,
        ema_alpha: float = 0.10,
        stagnation_patience: int = 200,
        min_improvement_delta: float = 0.0005,
        reference_point: Sequence[float] = (0.0,) * NUM_OBJECTIVES,
        initial_ema_hypervolume: float = 0.0,
        initial_best_hypervolume: float = 0.0,
        initial_stagnation_cycles: int = 0,
    ) -> None:
        self._archive_capacity: int = max(2, int(archive_capacity))
        self._history_window: int = max(1, history_window)
        self._ema_alpha: float = max(0.0, min(1.0, ema_alpha))
        self._stagnation_patience: int = stagnation_patience
        self._min_delta: float = min_improvement_delta

        padded_reference = list(reference_point)[:NUM_OBJECTIVES]
        while len(padded_reference) < NUM_OBJECTIVES:
            padded_reference.append(0.0)
        self._reference_point: Tuple[float, ...] = tuple(
            float(value) for value in padded_reference
        )

        self._archive: List[Individual] = []
        self._hypervolume_history: Deque[float] = deque(maxlen=self._history_window)
        self._ema_hypervolume: float = max(0.0, float(initial_ema_hypervolume))
        self._best_hypervolume: float = max(0.0, float(initial_best_hypervolume))
        self._cycles_without_improvement: int = max(0, int(initial_stagnation_cycles))
        self._total_generations: int = 0

    # -- archive maintenance ------------------------------------------------

    def _merge_into_archive(self, candidates: List[Individual]) -> bool:
        """Merge candidates into the non-dominated archive.

        Returns True when at least one novel objective vector (not present
        before the merge, within tolerance) survives into the archive.
        """
        previous_vectors = [list(ind.fitness) for ind in self._archive]

        pool: Dict[int, Individual] = {ind.uid: ind for ind in self._archive}
        for candidate in candidates:
            twin_uid: Optional[int] = None
            for uid, member in pool.items():
                if uid != candidate.uid and _configs_close(member.config, candidate.config):
                    twin_uid = uid
                    break
            if twin_uid is not None:
                # Re-measured configuration: the freshest measurement wins.
                del pool[twin_uid]
            pool[candidate.uid] = candidate

        members = list(pool.values())
        fronts = fast_non_dominated_sort(members)
        frontier = fronts[0] if fronts else []
        assign_crowding_distance(frontier)
        if len(frontier) > self._archive_capacity:
            frontier.sort(key=lambda ind: -ind.crowding_distance)
            frontier = frontier[: self._archive_capacity]
        self._archive = frontier

        for individual in self._archive:
            if not any(
                _vectors_close(individual.fitness, previous)
                for previous in previous_vectors
            ):
                return True
        return False

    def restore_archive(self, records: Any) -> int:
        """Rehydrate the archive from serialized frontier records.

        Accepts the ``pareto_frontier`` entries emitted by a previous
        ``evaluate`` call (dicts carrying ``config`` and ``fitness``).
        Returns the number of individuals restored.
        """
        if not isinstance(records, (list, tuple)):
            return 0
        restored: List[Individual] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            config = _sanitize_config(record.get("config"))
            if not config:
                continue
            generation = _coerce_float(record.get("generation"))
            restored.append(
                Individual(
                    config=config,
                    fitness=coerce_objective_vector(record.get("fitness")),
                    generation=int(generation) if generation is not None else 0,
                )
            )
        if restored:
            self._merge_into_archive(restored)
        return len(restored)

    def update_frontier(self, candidates: List[Individual]) -> List[Individual]:
        """Record one generational step: merge candidates, track hypervolume."""
        admitted_novel = self._merge_into_archive(candidates)

        current_hv = hypervolume(
            [ind.fitness for ind in self._archive], self._reference_point
        )
        self._hypervolume_history.append(current_hv)
        self._ema_hypervolume = (
            self._ema_alpha * current_hv
            + (1.0 - self._ema_alpha) * self._ema_hypervolume
        )

        improved = current_hv > self._best_hypervolume + self._min_delta
        if improved:
            self._best_hypervolume = current_hv
        if improved or admitted_novel:
            self._cycles_without_improvement = 0
        else:
            self._cycles_without_improvement += 1
        self._total_generations += 1
        return list(self._archive)

    def select_elites(self, population: List[Individual]) -> List[Individual]:
        """Return the non-dominated frontier of ``population`` ∪ archive.

        The frontier is ordered by descending crowding distance (extremes
        first) and capped at ``archive_capacity``.
        """
        pool: Dict[int, Individual] = {ind.uid: ind for ind in population}
        for member in self._archive:
            pool.setdefault(member.uid, member)
        fronts = fast_non_dominated_sort(list(pool.values()))
        if not fronts:
            return []
        frontier = fronts[0]
        assign_crowding_distance(frontier)
        frontier.sort(key=lambda ind: -ind.crowding_distance)
        return frontier[: self._archive_capacity]

    # -- observability -------------------------------------------------------

    @property
    def pareto_front(self) -> List[Individual]:
        return list(self._archive)

    @property
    def frontier_size(self) -> int:
        return len(self._archive)

    @property
    def is_stagnant(self) -> bool:
        return self._cycles_without_improvement >= self._stagnation_patience

    @property
    def best_hypervolume(self) -> float:
        return self._best_hypervolume

    @property
    def ema_hypervolume(self) -> float:
        return self._ema_hypervolume

    def stats(self) -> Dict[str, Any]:
        recent = list(self._hypervolume_history)
        objective_bests = {name: 0.0 for name in OBJECTIVE_NAMES}
        for individual in self._archive:
            for idx, name in enumerate(OBJECTIVE_NAMES):
                objective_bests[name] = max(objective_bests[name], individual.fitness[idx])
        return {
            "total_generations": self._total_generations,
            "frontier_size": len(self._archive),
            "archive_capacity": self._archive_capacity,
            "hypervolume": round(recent[-1] if recent else 0.0, 6),
            "best_hypervolume": round(self._best_hypervolume, 6),
            "ema_hypervolume": round(self._ema_hypervolume, 6),
            "objective_bests": {k: round(v, 6) for k, v in objective_bests.items()},
            "cycles_without_improvement": self._cycles_without_improvement,
            "is_stagnant": self.is_stagnant,
            "history_size": len(recent),
            "recent_mean_hypervolume": round(sum(recent) / max(len(recent), 1), 6),
            "recent_max_hypervolume": round(max(recent) if recent else 0.0, 6),
            "recent_min_hypervolume": round(min(recent) if recent else 0.0, 6),
        }


# ---------------------------------------------------------------------------
# Evolutionary operators
# ---------------------------------------------------------------------------

def tournament_select(
    population: List[Individual],
    k: int = 3,
) -> Individual:
    """Crowded tournament selection (NSGA-II).

    The winner is the competitor with the lowest non-domination rank,
    ties broken by the largest crowding distance. Callers must run
    ``fast_non_dominated_sort`` and ``assign_crowding_distance`` over the
    population beforehand so ranks and distances are current.
    """
    if not population:
        raise ValueError("tournament_select requires a non-empty population")
    if len(population) <= k:
        competitors = list(population)
    else:
        competitors = random.sample(population, k)
    return min(competitors, key=crowded_comparison_key)


def gaussian_mutate(
    config: Dict[str, float],
    sigma: float,
    matrix: HyperparameterMatrix,
) -> Dict[str, float]:
    mutated: Dict[str, float] = {}
    for name, value in config.items():
        noise = random.gauss(0, sigma)
        mutated[name] = value + noise
    return matrix.clip_configuration(mutated)


def uniform_crossover(
    parent_a: Dict[str, float],
    parent_b: Dict[str, float],
    probability: float = 0.5,
) -> Dict[str, float]:
    child: Dict[str, float] = {}
    for key in set(parent_a.keys()) | set(parent_b.keys()):
        if random.random() < probability:
            child[key] = parent_a.get(key, parent_b.get(key, 0.0))
        else:
            child[key] = parent_b.get(key, parent_a.get(key, 0.0))
    return child


def blend_crossover(
    parent_a: Dict[str, float],
    parent_b: Dict[str, float],
    alpha: float = 0.5,
) -> Dict[str, float]:
    child: Dict[str, float] = {}
    for key in set(parent_a.keys()) | set(parent_b.keys()):
        val_a = parent_a.get(key, 0.0)
        val_b = parent_b.get(key, 0.0)
        low = min(val_a, val_b) - alpha * abs(val_a - val_b)
        high = max(val_a, val_b) + alpha * abs(val_a - val_b)
        child[key] = random.uniform(low, high)
    return child


# ---------------------------------------------------------------------------
# Evolutionary engine (multi-objective, NSGA-II style)
# ---------------------------------------------------------------------------

def _objective_aggregates(population: List[Individual]) -> Dict[str, Dict[str, float]]:
    aggregates: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(OBJECTIVE_NAMES):
        values = [ind.fitness[idx] for ind in population]
        if values:
            aggregates[name] = {
                "mean": round(sum(values) / len(values), 6),
                "max": round(max(values), 6),
                "min": round(min(values), 6),
            }
        else:
            aggregates[name] = {"mean": 0.0, "max": 0.0, "min": 0.0}
    return aggregates


class EvolutionaryEngine:
    """Population-based multi-objective optimizer for hyperparameter tuning.

    Each generational step performs NSGA-II style environmental selection:

      1. The evaluated population is pooled with the tracker's persistent
         Pareto archive and partitioned via fast non-dominated sorting.
      2. The first front (the current non-dominated frontier) is handed to the
         tracker, which merges it into the archive — preserving the frontier
         pool across generational steps.
      3. The next generation is seeded with clones of the entire frontier
         (crowding-distance truncation applies only when the frontier would
         crowd out the reserved offspring quota), then filled with offspring
         bred via crowded tournament selection, crossover, and mutation.
    """

    def __init__(
        self,
        matrix: HyperparameterMatrix,
        population_size: int = 64,
        mutation_sigma: float = 0.05,
        crossover_prob: float = 0.70,
        tournament_k: int = 3,
        offspring_reserve: float = 0.125,
    ) -> None:
        self._matrix: HyperparameterMatrix = matrix
        self._pop_size: int = max(4, int(population_size))
        self._sigma: float = mutation_sigma
        self._crossover_prob: float = crossover_prob
        self._tournament_k: int = max(2, int(tournament_k))
        self._offspring_reserve: float = max(0.0, min(0.9, float(offspring_reserve)))
        self._generation: int = 0

        self._last_front_count: int = 0
        self._last_frontier_size: int = 0
        self._last_pool_aggregates: Dict[str, Dict[str, float]] = {}

        self._population: List[Individual] = []
        self._initialize_population()

    def _initialize_population(self) -> None:
        for _ in range(self._pop_size):
            config = self._matrix.sample_configuration()
            self._population.append(Individual(config=config, generation=0))

    def inject_configurations(self, configs: Any, from_tail: bool = False) -> int:
        """Overwrite population slots with externally supplied configs.

        Head injection (the default) rehydrates the population emitted by the
        previous cycle so that incoming objective-matrix rows align with the
        configurations they measured; invalid rows keep their slot so that
        alignment is preserved. Tail injection (``from_tail=True``) re-seeds
        the trailing slots — used by warm-bootstrap signals from the decision
        tree's experience replay — without disturbing the head alignment.
        Returns the number of slots successfully injected.
        """
        if not isinstance(configs, (list, tuple)):
            return 0
        injected = 0
        if from_tail:
            usable = [cfg for cfg in (_sanitize_config(raw) for raw in configs) if cfg]
            for offset, config in enumerate(usable):
                slot = len(self._population) - 1 - offset
                if slot < 0:
                    break
                self._population[slot] = Individual(
                    config=self._matrix.clip_configuration(config),
                    generation=self._generation,
                )
                injected += 1
            return injected
        for slot, raw in enumerate(configs):
            if slot >= len(self._population):
                break
            config = _sanitize_config(raw)
            if not config:
                continue
            self._population[slot] = Individual(
                config=self._matrix.clip_configuration(config),
                generation=self._generation,
            )
            injected += 1
        return injected

    def assign_fitness(self, fitness_vectors: Sequence[Any]) -> None:
        """Attach one objective vector per individual (row order == population order)."""
        for individual, vector in zip(self._population, fitness_vectors):
            individual.fitness = coerce_objective_vector(vector)

    def evolve(self, tracker: FitnessSurvivalTracker) -> List[Individual]:
        """Advance one generation using non-dominated sorting selection."""
        self._generation += 1

        pool_map: Dict[int, Individual] = {ind.uid: ind for ind in self._population}
        for member in tracker.pareto_front:
            pool_map.setdefault(member.uid, member)
        pool = list(pool_map.values())

        fronts = fast_non_dominated_sort(pool)
        for front in fronts:
            assign_crowding_distance(front)
        self._last_front_count = len(fronts)
        self._last_pool_aggregates = _objective_aggregates(pool)

        frontier = tracker.update_frontier(fronts[0] if fronts else [])
        self._last_frontier_size = len(frontier)

        # Elitism: clone the entire current Pareto frontier into the next
        # generation. A small offspring quota stays reserved so exploration
        # never fully stalls; if the frontier exceeds the clone budget the
        # most diverse members (largest crowding distance) survive, while the
        # full frontier pool remains intact inside the tracker archive.
        reserved_offspring = max(1, int(round(self._pop_size * self._offspring_reserve)))
        clone_budget = max(1, self._pop_size - reserved_offspring)
        survivors = list(frontier)
        if len(survivors) > clone_budget:
            survivors.sort(key=lambda ind: -ind.crowding_distance)
            survivors = survivors[:clone_budget]

        next_generation: List[Individual] = [
            member.clone(self._generation) for member in survivors
        ]

        while len(next_generation) < self._pop_size:
            parent_a = tournament_select(pool, self._tournament_k)
            parent_b = tournament_select(pool, self._tournament_k)

            if random.random() < self._crossover_prob:
                child_config = uniform_crossover(parent_a.config, parent_b.config)
            else:
                child_config = blend_crossover(parent_a.config, parent_b.config)

            child_config = gaussian_mutate(child_config, self._sigma, self._matrix)

            next_generation.append(
                Individual(
                    config=child_config,
                    generation=self._generation,
                    parent_id=parent_a.uid,
                )
            )

        self._population = next_generation[: self._pop_size]
        return list(self._population)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def population(self) -> List[Individual]:
        return list(self._population)

    def pareto_front(self) -> List[Individual]:
        """Non-dominated front of the current population (sorted in place)."""
        fronts = fast_non_dominated_sort(self._population)
        if not fronts:
            return []
        assign_crowding_distance(fronts[0])
        return list(fronts[0])

    def population_stats(self) -> Dict[str, Any]:
        if not self._population:
            return {"size": 0}
        return {
            "size": len(self._population),
            "generation": self._generation,
            "front_count": self._last_front_count,
            "pareto_frontier_size": self._last_frontier_size,
            "objective_aggregates": (
                self._last_pool_aggregates or _objective_aggregates(self._population)
            ),
        }


# ---------------------------------------------------------------------------
# Parameter drift detection
# ---------------------------------------------------------------------------

def detect_parameter_drift(
    current_config: Dict[str, float],
    baseline_config: Dict[str, float],
    max_drift: float = 0.30,
) -> Dict[str, Any]:
    drifted_params: List[Dict[str, Any]] = []
    total_drift: float = 0.0
    param_count: int = 0

    for key in set(current_config.keys()) | set(baseline_config.keys()):
        current_val = current_config.get(key, 0.0)
        baseline_val = baseline_config.get(key, 0.0)
        denominator = abs(baseline_val) if abs(baseline_val) > 1e-9 else 1.0
        drift = abs(current_val - baseline_val) / denominator
        total_drift += drift
        param_count += 1

        if drift > max_drift:
            drifted_params.append({
                "parameter": key,
                "current": round(current_val, 6),
                "baseline": round(baseline_val, 6),
                "drift_ratio": round(drift, 6),
            })

    avg_drift = total_drift / max(param_count, 1)

    return {
        "drifted_parameters": drifted_params,
        "drifted_count": len(drifted_params),
        "average_drift": round(avg_drift, 6),
        "max_drift_threshold": max_drift,
        "drift_exceeded": len(drifted_params) > 0,
    }


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def evaluate(metadata: Dict[str, Any], hyper_params: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full multi-objective parameter tuning pipeline.

    Fitness is the maximized vector
    ``[accuracy_score, inverse_wall_seconds, compression_ratio]``; selection
    uses non-dominated sorting with crowding-distance diversity (NSGA-II
    mechanics), and elitism preserves the entire Pareto frontier across
    generational steps.

    Parameters
    ----------
    metadata : dict
        Mutable state dictionary. Recognized input keys:
          - ``current_cycle`` (int): the current optimization cycle index.
          - ``performance_log`` (list, optional): raw measurement records;
            each entry may carry accuracy (``accuracy_score`` / ``accuracy`` /
            ``score`` / boolean ``success``), execution time (``wall_seconds``
            / ``execution_time`` / ``latency_ms`` / ``latency_us`` …), and a
            ``compression_ratio`` reading. Rows map onto population members
            in order.
          - ``fitness_matrix`` / ``objective_matrix`` (list[list], optional):
            measurement rows of ``(accuracy, wall_seconds, compression_ratio)``.
          - ``fitness_scores`` (list, optional): legacy input; scalar entries
            are interpreted as accuracy-only measurements.
          - ``pareto_frontier`` (list[dict], optional): elite archive emitted
            by the previous cycle, restored for frontier persistence.
          - ``tuned_population`` (list[dict], optional): population configs
            emitted by the previous cycle, restored so measurement rows align
            with the configurations they measured.
          - ``boost_mutation_sigma`` (bool, optional): kinetic stagnation
            signal from the decision tree; expands mutation variance once,
            scaled by ``mutation_sigma_boost_factor``.
          - ``bootstrap_seed_configs`` (list[dict], optional): historical
            parameter matrices staged by a WARM_BOOTSTRAP intercept; consumed
            to re-seed the trailing population slots.
          - ``baseline_config`` (dict, optional): reference configuration for
            drift detection.
    hyper_params : dict
        Runtime overrides merged on top of ``build_manifest.json`` defaults.

    Returns
    -------
    dict
        Enriched metadata containing the updated elite configurations
        (``elite_configurations``), the serialized Pareto frontier
        (``pareto_frontier``), the next population to measure
        (``tuned_population``), per-objective champions, the balanced
        compromise configuration, hypervolume-based survival statistics,
        drift analysis, and convergence status.
    """
    start_time: float = time.monotonic()
    logger.info("Parameter tuner pipeline started (multi-objective Pareto mode)")

    manifest: Dict[str, Any] = _load_manifest()
    params: Dict[str, Any] = {**_get_tuner_params(manifest), **hyper_params}
    thresholds: Dict[str, Any] = _get_thresholds(manifest)
    ceilings: Dict[str, Any] = _get_cost_ceilings(manifest)

    wall_limit: float = ceilings.get("parameter_tuner_max_wall_seconds", 45.0)
    initial_lr: float = float(params.get("initial_learning_rate", 0.01))
    min_lr: float = float(params.get("min_learning_rate", 0.00001))
    schedule_name: str = str(params.get("annealing_schedule", "cosine_warm_restart"))
    warmup_cycles: int = int(params.get("warmup_cycles", 50))
    restart_mult: float = float(params.get("restart_period_multiplier", 2.0))
    pop_size: int = int(params.get("population_size", 64))
    elite_ratio: float = float(params.get("elite_survival_ratio", 0.20))
    mutation_sigma: float = float(params.get("mutation_sigma", 0.05))
    crossover_prob: float = float(params.get("crossover_probability", 0.70))
    tournament_k: int = int(params.get("tournament_k", 3))

    stagnation_patience: int = int(thresholds.get("stagnation_patience_cycles", 200))
    min_improvement: float = float(thresholds.get("min_improvement_delta", 0.0005))
    max_drift: float = float(thresholds.get("parameter_drift_max", 0.30))
    convergence_tol: float = float(thresholds.get("global_convergence_tolerance", 0.0001))
    history_window: int = int(
        manifest.get("scoring", {}).get("fitness_history_window", 100)
    )
    ema_alpha: float = float(manifest.get("scoring", {}).get("ema_alpha", 0.10))

    current_cycle: int = int(metadata.get("current_cycle", 1))

    # Kinetic stagnation deflection: when the decision tree flags a velocity/
    # acceleration stall it emits ``boost_mutation_sigma`` — honor the signal
    # by expanding search variance so the optimizer escapes local minima.
    sigma_boosted = False
    if metadata.get("boost_mutation_sigma"):
        boost_factor = _coerce_float(metadata.get("mutation_sigma_boost_factor"))
        if boost_factor is None or boost_factor <= 1.0:
            boost_factor = 2.0
        mutation_sigma = min(0.5, mutation_sigma * boost_factor)
        sigma_boosted = True
        metadata["boost_mutation_sigma"] = False
        metadata["boost_mutation_sigma_consumed"] = True
        logger.info(
            "Kinetic stagnation signal honored: mutation sigma boosted to %.4f",
            mutation_sigma,
        )
    metadata["mutation_sigma_boosted"] = sigma_boosted

    # --- Stage 1: Learning rate annealing ---
    scheduler = AnnealingSchedule(
        initial_lr=initial_lr,
        min_lr=min_lr,
        warmup_cycles=warmup_cycles,
        restart_period=100,
        restart_multiplier=restart_mult,
    )
    annealed_lr = scheduler.get_lr(current_cycle, schedule_name)
    metadata["annealed_learning_rate"] = round(annealed_lr, 8)
    metadata["annealing_schedule"] = schedule_name

    lr_curve: List[float] = []
    sample_points = min(current_cycle + 1, 50)
    step = max(1, current_cycle // sample_points)
    for c in range(0, current_cycle + 1, step):
        lr_curve.append(round(scheduler.get_lr(c, schedule_name), 8))
    metadata["lr_curve_sample"] = lr_curve

    logger.info(
        "Learning rate annealed: cycle=%d, lr=%.8f, schedule=%s",
        current_cycle, annealed_lr, schedule_name,
    )

    # --- Stage 2: Build hyperparameter matrix ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit:
        matrix = build_default_matrix()
        metadata["search_space"] = matrix.to_dict()
        logger.info("Hyperparameter matrix: %d dimensions", matrix.dimension)
    else:
        matrix = build_default_matrix()

    # --- Stage 3: Multi-objective evolutionary step (Pareto frontier) ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit:
        engine = EvolutionaryEngine(
            matrix=matrix,
            population_size=pop_size,
            mutation_sigma=mutation_sigma,
            crossover_prob=crossover_prob,
            tournament_k=tournament_k,
        )

        # The manifest's elite survival ratio sizes the archive headroom so
        # the preserved frontier pool may exceed a single generation.
        archive_capacity = max(
            pop_size, int(round(pop_size * (1.0 + max(0.0, elite_ratio))))
        )
        previous_stats = metadata.get("survival_tracker_stats")
        if not isinstance(previous_stats, dict):
            previous_stats = {}
        tracker = FitnessSurvivalTracker(
            archive_capacity=archive_capacity,
            history_window=history_window,
            ema_alpha=ema_alpha,
            stagnation_patience=stagnation_patience,
            min_improvement_delta=min_improvement,
            initial_ema_hypervolume=(
                _coerce_float(previous_stats.get("ema_hypervolume")) or 0.0
            ),
            initial_best_hypervolume=(
                _coerce_float(previous_stats.get("best_hypervolume")) or 0.0
            ),
            initial_stagnation_cycles=int(
                _coerce_float(previous_stats.get("cycles_without_improvement")) or 0
            ),
        )

        restored_frontier = tracker.restore_archive(metadata.get("pareto_frontier"))
        restored_population = engine.inject_configurations(
            metadata.get("tuned_population")
        )

        # Warm bootstrap: the decision tree's FallbackRouter stages historical
        # elite parameter matrices from the experience replay buffer when a
        # random restart is intercepted. Re-seed the trailing (unmeasured)
        # population slots with them so the pool restarts from proven ground.
        bootstrap_injected = 0
        bootstrap_configs = metadata.get("bootstrap_seed_configs")
        if isinstance(bootstrap_configs, (list, tuple)) and bootstrap_configs:
            bootstrap_injected = engine.inject_configurations(
                bootstrap_configs, from_tail=True
            )
            metadata["bootstrap_seed_configs"] = []
            metadata["warm_bootstrap_consumed"] = bootstrap_injected > 0
            logger.info(
                "Warm bootstrap: re-seeded %d population slots from replay trajectories",
                bootstrap_injected,
            )
        metadata["warm_bootstrap_injected"] = bootstrap_injected

        fitness_vectors, vector_sources = collect_objective_vectors(
            metadata, len(engine.population)
        )
        engine.assign_fitness(fitness_vectors)

        new_population = engine.evolve(tracker)
        frontier = tracker.pareto_front

        metadata["objective_names"] = list(OBJECTIVE_NAMES)
        metadata["objective_vector_sources"] = vector_sources
        metadata["restored_frontier_size"] = restored_frontier
        metadata["restored_population_size"] = restored_population

        metadata["pareto_frontier"] = [ind.to_dict() for ind in frontier]
        metadata["elite_configurations"] = [
            {key: round(value, 6) for key, value in ind.config.items()}
            for ind in frontier
        ]
        metadata["tuned_population"] = [
            {key: round(value, 6) for key, value in ind.config.items()}
            for ind in new_population
        ]

        metadata["population_stats"] = engine.population_stats()
        metadata["survival_tracker_stats"] = tracker.stats()
        metadata["is_stagnant"] = tracker.is_stagnant

        compromise = select_compromise(frontier)
        if compromise is not None:
            metadata["compromise_individual"] = compromise.to_dict()
            metadata["best_config"] = {
                key: round(value, 6) for key, value in compromise.config.items()
            }
        metadata["objective_champions"] = objective_champions(frontier)

        diverse = sorted(frontier, key=lambda ind: -ind.crowding_distance)[:5]
        metadata["top_individuals"] = [ind.to_dict() for ind in diverse]

        logger.info(
            "Pareto evolution: gen=%d, frontier=%d, hypervolume=%.6f, sources=%s",
            engine.generation,
            len(frontier),
            metadata["survival_tracker_stats"]["hypervolume"],
            vector_sources,
        )
    else:
        logger.warning("Wall-time exceeded before evolutionary step")

    # --- Stage 4: Parameter drift detection ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit:
        baseline_config: Dict[str, float] = metadata.get("baseline_config", {})
        current_config: Dict[str, float] = metadata.get("best_config", {})

        if baseline_config and current_config:
            drift_result = detect_parameter_drift(
                current_config, baseline_config, max_drift=max_drift
            )
            metadata["drift_analysis"] = drift_result
            if drift_result["drift_exceeded"]:
                logger.warning(
                    "Parameter drift detected: %d parameters exceed threshold",
                    drift_result["drifted_count"],
                )
        else:
            metadata["drift_analysis"] = {"status": "no_baseline_available"}

    # --- Stage 5: Convergence check (hypervolume stability) ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit:
        tracker_stats = metadata.get("survival_tracker_stats", {})
        if not isinstance(tracker_stats, dict):
            tracker_stats = {}
        current_hv = _coerce_float(tracker_stats.get("hypervolume")) or 0.0
        ema_hv = _coerce_float(tracker_stats.get("ema_hypervolume")) or 0.0
        gap = abs(current_hv - ema_hv)
        relative_gap = gap / max(abs(current_hv), 1.0)
        converged = current_hv > 0.0 and relative_gap < convergence_tol
        metadata["convergence_check"] = {
            "hypervolume": round(current_hv, 6),
            "ema_hypervolume": round(ema_hv, 6),
            "gap": round(gap, 6),
            "relative_gap": round(relative_gap, 6),
            "tolerance": convergence_tol,
            "converged": converged,
        }
        if converged:
            logger.info(
                "Convergence detected — relative hypervolume gap %.6f < tolerance %.6f",
                relative_gap, convergence_tol,
            )

    # --- Finalize ---
    metadata["mutation_rate"] = mutation_sigma
    metadata["annealing_temp"] = round(
        initial_lr / math.log1p(current_cycle + 1), 8
    )

    wall_seconds = round(time.monotonic() - start_time, 6)
    metadata["parameter_tuner_wall_seconds"] = wall_seconds
    metadata["parameter_tuner_status"] = (
        "budget_exceeded" if wall_seconds > wall_limit else "complete"
    )

    logger.info(
        "Parameter tuner pipeline finished in %.3fs — gen=%d, frontier=%d, stagnant=%s",
        wall_seconds,
        metadata.get("population_stats", {}).get("generation", 0),
        metadata.get("survival_tracker_stats", {}).get("frontier_size", 0),
        metadata.get("is_stagnant", False),
    )
    return metadata
