# -*- coding: utf-8 -*-
"""
decision_tree.py — Multi-Layered Heuristic State Machine & Decision Engine

Implements:
  - Evaluation priority queues with configurable capacity and ordering
  - Dynamic cost-benefit metrics parsing (discount-factor weighted)
  - Multi-layered heuristic state machine with transition tracking
  - Kinetic stagnation deflection: a rolling score-history array (capacity 15)
    tracks the first derivative (velocity) and second derivative
    (acceleration) of incoming evaluation scores; when both stall for 15
    consecutive cycles the FSM emits a ``boost_mutation_sigma`` signal that
    forces the parameter tuner to expand search variance proactively
  - Fallback strategy routing (cascade + random-restart), with random
    restarts intercepted into experience-replay ``WARM_BOOTSTRAP`` re-seeds

Pipeline entry point: evaluate(metadata, hyper_params)
"""

import hashlib
import heapq
import json
import logging
import math
import os
import time
from collections import OrderedDict, deque
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Tuple

try:  # package-style import (normal pipeline usage)
    from builder_brains.experience_replay import ExperienceReplayBuffer
except ImportError:  # flat sys.path import (sandbox / evolve-loop usage)
    try:
        from experience_replay import ExperienceReplayBuffer
    except ImportError:  # buffer unavailable — warm bootstrap degrades gracefully
        ExperienceReplayBuffer = None  # type: ignore[assignment, misc]

logger = logging.getLogger("builder_brains.decision_tree")

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


def _get_dt_params(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("hyperparameter_weights", {}).get("decision_tree", {})


def _get_thresholds(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("thresholds", {})


def _get_cost_ceilings(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("execution_cost_ceilings", {})


# ---------------------------------------------------------------------------
# Strategy enumeration
# ---------------------------------------------------------------------------

class Strategy(Enum):
    CONSERVATIVE = auto()
    EXPLORATORY = auto()
    GREEDY = auto()
    BALANCED = auto()
    FALLBACK_CASCADE = auto()
    RANDOM_RESTART = auto()
    WARM_BOOTSTRAP = auto()


STRATEGY_PRIORITY: Dict[Strategy, int] = {
    Strategy.GREEDY: 0,
    Strategy.EXPLORATORY: 1,
    Strategy.BALANCED: 2,
    Strategy.CONSERVATIVE: 3,
    Strategy.FALLBACK_CASCADE: 4,
    Strategy.RANDOM_RESTART: 5,
    Strategy.WARM_BOOTSTRAP: 6,
}


# ---------------------------------------------------------------------------
# Evaluation priority queue
# ---------------------------------------------------------------------------

class EvaluationItem:
    """A single candidate action scored for the priority queue."""

    __slots__ = ("score", "cost", "label", "payload", "_insert_order")

    _global_counter: int = 0

    def __init__(
        self,
        score: float,
        cost: float,
        label: str,
        payload: Dict[str, Any],
    ) -> None:
        self.score: float = score
        self.cost: float = cost
        self.label: str = label
        self.payload: Dict[str, Any] = payload
        EvaluationItem._global_counter += 1
        self._insert_order: int = EvaluationItem._global_counter

    @property
    def net_value(self) -> float:
        return self.score - self.cost

    def __lt__(self, other: "EvaluationItem") -> bool:
        if self.net_value != other.net_value:
            return self.net_value > other.net_value
        return self._insert_order < other._insert_order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 6),
            "cost": round(self.cost, 6),
            "net_value": round(self.net_value, 6),
        }


class EvaluationPriorityQueue:
    """Bounded min-heap priority queue that retains the top-K candidates."""

    def __init__(self, capacity: int = 1024) -> None:
        self._capacity: int = max(1, capacity)
        self._heap: List[EvaluationItem] = []
        self._total_pushed: int = 0
        self._total_evicted: int = 0

    def push(self, item: EvaluationItem) -> None:
        self._total_pushed += 1
        if len(self._heap) < self._capacity:
            heapq.heappush(self._heap, item)
        else:
            evicted = heapq.heappushpop(self._heap, item)
            if evicted is not item:
                self._total_evicted += 1

    def pop(self) -> Optional[EvaluationItem]:
        if self._heap:
            return heapq.heappop(self._heap)
        return None

    def peek(self) -> Optional[EvaluationItem]:
        if self._heap:
            return self._heap[0]
        return None

    def drain_sorted(self) -> List[EvaluationItem]:
        items = sorted(self._heap)
        self._heap = []
        return items

    @property
    def size(self) -> int:
        return len(self._heap)

    def stats(self) -> Dict[str, int]:
        return {
            "capacity": self._capacity,
            "current_size": self.size,
            "total_pushed": self._total_pushed,
            "total_evicted": self._total_evicted,
        }


# ---------------------------------------------------------------------------
# Cost-benefit metrics parser
# ---------------------------------------------------------------------------

class CostBenefitAnalyzer:
    """Compute discounted cost-benefit ratios for a set of proposed actions."""

    def __init__(self, discount_gamma: float = 0.97, penalty: float = 0.02) -> None:
        self._gamma: float = max(0.0, min(1.0, discount_gamma))
        self._penalty: float = max(0.0, penalty)

    def discounted_score(self, raw_score: float, step: int) -> float:
        return raw_score * (self._gamma ** step)

    def penalized_cost(self, raw_cost: float, transitions: int) -> float:
        return raw_cost + self._penalty * transitions

    def net_benefit(
        self,
        raw_score: float,
        raw_cost: float,
        step: int,
        transitions: int,
    ) -> float:
        return self.discounted_score(raw_score, step) - self.penalized_cost(
            raw_cost, transitions
        )

    def rank_actions(
        self,
        actions: List[Dict[str, Any]],
        current_step: int,
        current_transitions: int,
    ) -> List[Dict[str, Any]]:
        ranked: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, action in enumerate(actions):
            score = float(action.get("score", 0.0))
            cost = float(action.get("cost", 0.0))
            nb = self.net_benefit(score, cost, current_step, current_transitions)
            entry = {**action, "net_benefit": round(nb, 6), "rank_order": 0}
            ranked.append((nb, idx, entry))

        ranked.sort(key=lambda t: (-t[0], t[1]))
        result: List[Dict[str, Any]] = []
        for rank, (_, _, entry) in enumerate(ranked):
            entry["rank_order"] = rank
            result.append(entry)
        return result


# ---------------------------------------------------------------------------
# Heuristic state machine
# ---------------------------------------------------------------------------

class HeuristicState:
    """A single state in the multi-layer heuristic FSM."""

    def __init__(self, name: str, layer: int) -> None:
        self.name: str = name
        self.layer: int = layer
        self.transitions: Dict[str, "HeuristicState"] = {}
        self.entry_count: int = 0
        self.cumulative_score: float = 0.0

    def add_transition(self, signal: str, target: "HeuristicState") -> None:
        self.transitions[signal] = target

    def enter(self, score: float) -> None:
        self.entry_count += 1
        self.cumulative_score += score

    @property
    def average_score(self) -> float:
        return self.cumulative_score / max(self.entry_count, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "entry_count": self.entry_count,
            "average_score": round(self.average_score, 6),
            "transitions": list(self.transitions.keys()),
        }


class HeuristicStateMachine:
    """Multi-layered deterministic finite-state machine with heuristic routing.

    Layers correspond to escalation tiers:
      Layer 0 — conservative / low-cost
      Layer 1 — exploratory / moderate-cost
      Layer 2 — greedy / high-cost
      Layer 3 — balanced / adaptive
      Layer 4 — fallback cascade
    """

    #: Rolling kinetic-history capacity and stall patience (cycles).
    KINETIC_WINDOW: int = 15
    KINETIC_STALL_PATIENCE: int = 15

    def __init__(
        self,
        layer_count: int = 5,
        velocity_epsilon: float = 0.005,
        acceleration_epsilon: float = 0.005,
        stall_patience: int = KINETIC_STALL_PATIENCE,
    ) -> None:
        self._states: Dict[str, HeuristicState] = {}
        self._current: Optional[HeuristicState] = None
        self._transition_log: List[Dict[str, Any]] = []
        self._transition_count: int = 0

        # Kinetic stagnation deflection: rolling historical tracking array
        # (max capacity 15) feeding velocity / acceleration derivatives.
        self._score_history: Deque[float] = deque(maxlen=self.KINETIC_WINDOW)
        self._velocity_epsilon: float = max(1e-12, float(velocity_epsilon))
        self._acceleration_epsilon: float = max(1e-12, float(acceleration_epsilon))
        self._stall_patience: int = max(1, int(stall_patience))
        self._kinetic_stall_cycles: int = 0
        self._kinetic_anomaly_flagged: bool = False
        self._latest_velocity: float = 0.0
        self._latest_acceleration: float = 0.0

        self._build_layers(layer_count)

    def _build_layers(self, layer_count: int) -> None:
        layer_names = [
            "conservative_scan",
            "exploratory_probe",
            "greedy_exploit",
            "balanced_adaptive",
            "fallback_cascade",
        ]
        for layer_idx in range(min(layer_count, len(layer_names))):
            state = HeuristicState(name=layer_names[layer_idx], layer=layer_idx)
            self._states[state.name] = state

        state_list = list(self._states.values())
        for i, state in enumerate(state_list):
            if i + 1 < len(state_list):
                state.add_transition("escalate", state_list[i + 1])
            if i > 0:
                state.add_transition("demote", state_list[i - 1])
            state.add_transition("hold", state)

        if state_list:
            last = state_list[-1]
            last.add_transition("restart", state_list[0])

        # Kinetic stagnation escape hatch: every layer can jump straight to
        # the exploratory tier when the boost_mutation_sigma signal fires.
        if state_list:
            variance_target = state_list[1] if len(state_list) > 1 else state_list[0]
            for state in state_list:
                state.add_transition("boost_mutation_sigma", variance_target)

        self._current = state_list[0] if state_list else None

    def current_state_name(self) -> str:
        return self._current.name if self._current else "none"

    # -- kinetic stagnation deflection ------------------------------------

    def preload_scores(self, scores: Any) -> int:
        """Rehydrate the rolling score history (persisted across cycles).

        Only refills the tracking array — stall accounting is restored
        separately via :meth:`restore_kinetic_state` so replayed history is
        never double-counted. Returns the number of scores loaded.
        """
        if not isinstance(scores, (list, tuple)):
            return 0
        loaded = 0
        for value in scores:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(number) or math.isinf(number):
                continue
            self._score_history.append(number)
            loaded += 1
        return loaded

    def restore_kinetic_state(self, stall_cycles: Any) -> None:
        """Restore the consecutive-stall counter persisted by a prior cycle."""
        try:
            cycles = int(stall_cycles)
        except (TypeError, ValueError):
            cycles = 0
        self._kinetic_stall_cycles = max(0, cycles)
        self._kinetic_anomaly_flagged = (
            self._kinetic_stall_cycles >= self._stall_patience
        )

    def record_score(self, score: float) -> Dict[str, Any]:
        """Track score kinetics: velocity (1st derivative of the incoming
        evaluation scores) and acceleration (2nd derivative).

        When the velocity approaches zero and the acceleration stalls for
        ``stall_patience`` (15) consecutive cycles, a kinetic stagnation
        anomaly is flagged and :meth:`decide_signal` emits the specialized
        ``boost_mutation_sigma`` signal.
        """
        try:
            value = float(score)
        except (TypeError, ValueError):
            value = 0.0
        if math.isnan(value) or math.isinf(value):
            value = 0.0

        self._score_history.append(value)
        history = list(self._score_history)

        velocity = history[-1] - history[-2] if len(history) >= 2 else 0.0
        previous_velocity = history[-2] - history[-3] if len(history) >= 3 else 0.0
        acceleration = velocity - previous_velocity if len(history) >= 3 else 0.0
        self._latest_velocity = velocity
        self._latest_acceleration = acceleration

        if (
            len(history) >= 3
            and abs(velocity) < self._velocity_epsilon
            and abs(acceleration) < self._acceleration_epsilon
        ):
            self._kinetic_stall_cycles += 1
        else:
            self._kinetic_stall_cycles = 0

        self._kinetic_anomaly_flagged = (
            self._kinetic_stall_cycles >= self._stall_patience
        )
        if self._kinetic_anomaly_flagged:
            logger.warning(
                "Kinetic stagnation anomaly: velocity=%.6f acceleration=%.6f "
                "stalled for %d consecutive cycles",
                velocity, acceleration, self._kinetic_stall_cycles,
            )
        return self.kinetic_snapshot()

    def consume_kinetic_anomaly(self) -> bool:
        """Acknowledge a flagged stagnation anomaly, resetting the stall
        counter so the boost signal fires once per detected plateau."""
        if not self._kinetic_anomaly_flagged:
            return False
        self._kinetic_anomaly_flagged = False
        self._kinetic_stall_cycles = 0
        return True

    @property
    def is_kinetically_stagnant(self) -> bool:
        return self._kinetic_anomaly_flagged

    @property
    def kinetic_stall_cycles(self) -> int:
        return self._kinetic_stall_cycles

    @property
    def score_history(self) -> List[float]:
        return list(self._score_history)

    def kinetic_snapshot(self) -> Dict[str, Any]:
        return {
            "history": [round(v, 6) for v in self._score_history],
            "window_capacity": self.KINETIC_WINDOW,
            "velocity": round(self._latest_velocity, 6),
            "acceleration": round(self._latest_acceleration, 6),
            "velocity_epsilon": self._velocity_epsilon,
            "acceleration_epsilon": self._acceleration_epsilon,
            "stall_cycles": self._kinetic_stall_cycles,
            "stall_patience": self._stall_patience,
            "stagnation_anomaly": self._kinetic_anomaly_flagged,
        }

    def transition(self, signal: str, score: float) -> str:
        if self._current is None:
            logger.error("State machine has no current state")
            return "error_no_state"

        target = self._current.transitions.get(signal)
        if target is None:
            logger.warning(
                "No transition for signal '%s' in state '%s' — holding",
                signal,
                self._current.name,
            )
            target = self._current

        previous_name = self._current.name
        self._current = target
        self._current.enter(score)
        self._transition_count += 1

        log_entry = {
            "step": self._transition_count,
            "from": previous_name,
            "signal": signal,
            "to": self._current.name,
            "score": round(score, 6),
        }
        self._transition_log.append(log_entry)
        logger.debug("Transition: %s", log_entry)
        return self._current.name

    def decide_signal(
        self,
        score: float,
        exploration_epsilon: float,
        exploitation_bias: float,
        confidence_minimum: float,
    ) -> str:
        if self._kinetic_anomaly_flagged:
            logger.info(
                "Kinetic stagnation interception gate tripped — emitting "
                "boost_mutation_sigma to expand search variance"
            )
            return "boost_mutation_sigma"
        if score < confidence_minimum:
            if self._current and self._current.layer >= 4:
                return "restart"
            return "escalate"
        normalized = score * exploitation_bias
        if normalized > (1.0 - exploration_epsilon):
            return "hold"
        if normalized < exploration_epsilon:
            return "escalate"
        if self._current and self._current.average_score > score:
            return "demote"
        return "hold"

    def snapshot(self) -> Dict[str, Any]:
        return {
            "current_state": self._current.to_dict() if self._current else None,
            "transition_count": self._transition_count,
            "states": {n: s.to_dict() for n, s in self._states.items()},
            "recent_transitions": self._transition_log[-10:],
            "kinetics": self.kinetic_snapshot(),
        }


# ---------------------------------------------------------------------------
# Fallback strategy router
# ---------------------------------------------------------------------------

class FallbackRouter:
    """Route actions through a cascade of fallback strategies when primary
    strategies fail to meet the confidence threshold.

    Cascade depth is bounded to prevent infinite loops.
    """

    def __init__(
        self,
        max_depth: int = 6,
        ttl_seconds: float = 300.0,
        replay_buffer: Optional["ExperienceReplayBuffer"] = None,
    ) -> None:
        self._max_depth: int = max(1, max_depth)
        self._ttl: float = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Strategy, float]] = OrderedDict()
        self._cascade_order: List[Strategy] = [
            Strategy.GREEDY,
            Strategy.EXPLORATORY,
            Strategy.BALANCED,
            Strategy.CONSERVATIVE,
            Strategy.FALLBACK_CASCADE,
            Strategy.RANDOM_RESTART,
        ]

        # Cross-module experience replay: RANDOM_RESTART conditions are
        # intercepted and converted into WARM_BOOTSTRAP re-seeds when
        # historical trajectories are available.
        self._replay_buffer: Optional["ExperienceReplayBuffer"] = replay_buffer
        if self._replay_buffer is None and ExperienceReplayBuffer is not None:
            try:
                self._replay_buffer = ExperienceReplayBuffer()
            except Exception as exc:
                logger.warning("Experience replay buffer unavailable: %s", exc)
                self._replay_buffer = None
        self._warm_bootstrap_count: int = 0

    @property
    def replay_buffer(self) -> Optional["ExperienceReplayBuffer"]:
        return self._replay_buffer

    def record_experience(
        self,
        state: Dict[str, Any],
        action: str,
        result: float,
        parameter_matrix: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Persist a state-action-result triplet for future warm bootstraps."""
        if self._replay_buffer is None:
            return False
        try:
            return self._replay_buffer.record_trajectory(
                state, action, result, parameter_matrix=parameter_matrix
            )
        except Exception as exc:
            logger.debug("Experience record skipped: %s", exc)
            return False

    def _attempt_warm_bootstrap(self, metadata: Dict[str, Any]) -> bool:
        """Intercept a RANDOM_RESTART with an intelligent WARM_BOOTSTRAP.

        Queries the replay buffer for the best matching historical trajectory
        parameter matrices and stages them in
        ``metadata["bootstrap_seed_configs"]``, where the parameter tuner
        consumes them to re-seed its optimization population pool. Returns
        True when at least one historical matrix was staged.
        """
        if self._replay_buffer is None:
            return False
        try:
            query_state = {
                "current_score": float(metadata.get("current_score", 0.0) or 0.0),
                "current_cycle": float(metadata.get("current_cycle", 0) or 0),
                "fsm_state": str(metadata.get("fsm_state", "")),
                "primary_strategy": str(metadata.get("primary_strategy", "")),
            }
            matrices = self._replay_buffer.best_parameter_matrices(
                state=query_state, count=5
            )
        except Exception as exc:
            logger.warning("Replay buffer query failed: %s", exc)
            return False
        if not matrices:
            return False

        staged = metadata.get("bootstrap_seed_configs")
        if not isinstance(staged, list):
            staged = []
        staged.extend(matrices)
        metadata["bootstrap_seed_configs"] = staged
        metadata["warm_bootstrap"] = {
            "seed_count": len(matrices),
            "source": "experience_replay",
        }
        self._warm_bootstrap_count += 1
        logger.info(
            "WARM_BOOTSTRAP engaged: staged %d historical parameter matrices "
            "for population re-seed",
            len(matrices),
        )
        return True

    def _cache_key(self, metadata_hash: str, strategy: Strategy) -> str:
        return f"{metadata_hash}::{strategy.name}"

    def _evict_stale(self) -> None:
        now = time.monotonic()
        stale_keys = [
            k for k, (_, ts) in self._cache.items() if (now - ts) > self._ttl
        ]
        for k in stale_keys:
            del self._cache[k]

    def route(
        self,
        metadata: Dict[str, Any],
        primary_strategy: Strategy,
        score: float,
        confidence_minimum: float,
    ) -> Tuple[Strategy, int]:
        self._evict_stale()

        if score >= confidence_minimum:
            return primary_strategy, 0

        meta_hash = hashlib.md5(
            json.dumps(sorted(metadata.items()), default=str).encode()
        ).hexdigest()[:12]

        depth = 0
        for fallback in self._cascade_order:
            if depth >= self._max_depth:
                break
            depth += 1

            cache_key = self._cache_key(meta_hash, fallback)
            if cache_key in self._cache:
                cached_strategy, _ = self._cache[cache_key]
                logger.debug("Fallback cache hit: %s -> %s", cache_key, cached_strategy)
                continue

            self._cache[cache_key] = (fallback, time.monotonic())
            if fallback is Strategy.RANDOM_RESTART and self._attempt_warm_bootstrap(metadata):
                logger.info(
                    "Random restart intercepted at depth %d -> WARM_BOOTSTRAP", depth
                )
                return Strategy.WARM_BOOTSTRAP, depth
            logger.info("Fallback cascade depth %d -> strategy %s", depth, fallback.name)
            return fallback, depth

        if self._attempt_warm_bootstrap(metadata):
            logger.warning(
                "Fallback cascade exhausted at depth %d — warm bootstrap restart", depth
            )
            return Strategy.WARM_BOOTSTRAP, depth
        logger.warning("Fallback cascade exhausted at depth %d — random restart", depth)
        return Strategy.RANDOM_RESTART, depth

    def stats(self) -> Dict[str, Any]:
        self._evict_stale()
        return {
            "max_depth": self._max_depth,
            "cache_size": len(self._cache),
            "ttl_seconds": self._ttl,
            "warm_bootstrap_count": self._warm_bootstrap_count,
            "replay_buffer_attached": self._replay_buffer is not None,
        }


# ---------------------------------------------------------------------------
# Action generator
# ---------------------------------------------------------------------------

def _generate_candidate_actions(
    metadata: Dict[str, Any],
    strategy: Strategy,
) -> List[Dict[str, Any]]:
    """Synthesize candidate actions based on the current strategy and metadata state."""
    actions: List[Dict[str, Any]] = []

    base_score = float(metadata.get("current_score", 0.5))
    cycle = int(metadata.get("current_cycle", 1))

    if strategy in (Strategy.GREEDY, Strategy.EXPLORATORY, Strategy.BALANCED):
        actions.append({
            "label": "increase_exploration_depth",
            "score": base_score * 1.15,
            "cost": 0.08 * math.log1p(cycle),
        })
        actions.append({
            "label": "tighten_convergence_tolerance",
            "score": base_score * 1.05,
            "cost": 0.05 * math.log1p(cycle),
        })

    if strategy in (Strategy.GREEDY, Strategy.BALANCED):
        actions.append({
            "label": "boost_mutation_sigma",
            "score": base_score * 1.20,
            "cost": 0.12 * math.log1p(cycle),
        })

    if strategy in (Strategy.CONSERVATIVE, Strategy.FALLBACK_CASCADE):
        actions.append({
            "label": "reduce_population_size",
            "score": base_score * 0.90,
            "cost": 0.02,
        })
        actions.append({
            "label": "increase_patience",
            "score": base_score * 0.95,
            "cost": 0.01,
        })

    if strategy == Strategy.EXPLORATORY:
        actions.append({
            "label": "random_hyperparameter_perturbation",
            "score": base_score * 1.10,
            "cost": 0.15,
        })

    if strategy == Strategy.RANDOM_RESTART:
        actions.append({
            "label": "full_parameter_reset",
            "score": 0.50,
            "cost": 0.25,
        })

    if strategy == Strategy.WARM_BOOTSTRAP:
        actions.append({
            "label": "warm_bootstrap_reseed",
            "score": max(base_score, 0.55) * 1.10,
            "cost": 0.10,
        })

    if not actions:
        actions.append({
            "label": "noop_hold",
            "score": base_score,
            "cost": 0.0,
        })

    return actions


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def evaluate(metadata: Dict[str, Any], hyper_params: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full decision-tree evaluation pipeline.

    Parameters
    ----------
    metadata : dict
        Mutable state dictionary.  Expected keys include ``current_score``,
        ``current_cycle``, ``exploration_depth``, etc.  Optional kinetic
        keys persisted across cycles: ``score_trajectory`` (rolling score
        history, capacity 15) and ``kinetic_stall_cycles``.
    hyper_params : dict
        Runtime overrides merged on top of ``build_manifest.json`` defaults.

    Returns
    -------
    dict
        The enriched metadata dictionary with decision outputs.  When a
        kinetic stagnation anomaly is flagged, ``boost_mutation_sigma`` /
        ``mutation_sigma_boost_factor`` are emitted for the parameter tuner;
        when a random restart is intercepted, ``bootstrap_seed_configs``
        carries historical parameter matrices for a WARM_BOOTSTRAP re-seed.
    """
    start_time: float = time.monotonic()
    logger.info("Decision-tree pipeline started")

    manifest: Dict[str, Any] = _load_manifest()
    params: Dict[str, Any] = {**_get_dt_params(manifest), **hyper_params}
    thresholds: Dict[str, Any] = _get_thresholds(manifest)
    ceilings: Dict[str, Any] = _get_cost_ceilings(manifest)

    wall_limit: float = ceilings.get("decision_tree_max_wall_seconds", 15.0)
    layer_count: int = int(params.get("heuristic_layer_count", 5))
    queue_capacity: int = int(params.get("priority_queue_capacity", 1024))
    discount_gamma: float = float(params.get("cost_benefit_discount_gamma", 0.97))
    exploration_eps: float = float(params.get("exploration_epsilon", 0.15))
    exploitation_bias: float = float(params.get("exploitation_bias", 0.85))
    fallback_depth: int = int(params.get("fallback_cascade_max_depth", 6))
    transition_penalty: float = float(params.get("state_transition_penalty", 0.02))
    route_ttl: float = float(params.get("route_cache_ttl_seconds", 300))
    confidence_min: float = float(thresholds.get("decision_confidence_minimum", 0.55))

    current_score: float = float(metadata.get("current_score", 0.5))
    current_cycle: int = int(metadata.get("current_cycle", 1))

    velocity_eps: float = float(params.get("kinetic_velocity_epsilon", 0.005))
    acceleration_eps: float = float(params.get("kinetic_acceleration_epsilon", 0.005))
    stall_patience: int = int(
        params.get("kinetic_stall_patience", HeuristicStateMachine.KINETIC_STALL_PATIENCE)
    )
    boost_factor: float = float(params.get("kinetic_boost_factor", 2.0))

    # --- Stage 1: State machine initialization and signal decision ---
    fsm = HeuristicStateMachine(
        layer_count=layer_count,
        velocity_epsilon=velocity_eps,
        acceleration_epsilon=acceleration_eps,
        stall_patience=stall_patience,
    )

    # Kinetic stagnation deflection: rehydrate the rolling score trajectory
    # persisted by previous cycles, then fold in this cycle's score so the
    # velocity / acceleration derivatives see a continuous history.
    fsm.preload_scores(metadata.get("score_trajectory"))
    fsm.restore_kinetic_state(metadata.get("kinetic_stall_cycles"))
    kinetics = fsm.record_score(current_score)
    metadata["score_kinetics"] = kinetics

    signal = fsm.decide_signal(
        score=current_score,
        exploration_epsilon=exploration_eps,
        exploitation_bias=exploitation_bias,
        confidence_minimum=confidence_min,
    )
    new_state = fsm.transition(signal, current_score)
    metadata["fsm_state"] = new_state
    metadata["fsm_signal"] = signal
    logger.info("FSM: signal=%s -> state=%s (score=%.4f)", signal, new_state, current_score)

    # When the interception gate trips, instantly emit the specialized
    # boost_mutation_sigma signal so the parameter tuner expands its search
    # variance and escapes the local minimum proactively.
    kinetic_alarm = fsm.consume_kinetic_anomaly()
    metadata["kinetic_stagnation_anomaly"] = kinetic_alarm
    if kinetic_alarm:
        metadata["boost_mutation_sigma"] = True
        metadata["mutation_sigma_boost_factor"] = boost_factor
        logger.warning(
            "Kinetic stagnation anomaly flagged — boost_mutation_sigma signal "
            "emitted (factor=%.2f)",
            boost_factor,
        )

    # Persist the rolling trajectory and stall counter for the next cycle.
    metadata["score_trajectory"] = [round(v, 6) for v in fsm.score_history]
    metadata["kinetic_stall_cycles"] = fsm.kinetic_stall_cycles

    # --- Stage 2: Map FSM state to strategy ---
    state_strategy_map: Dict[str, Strategy] = {
        "conservative_scan": Strategy.CONSERVATIVE,
        "exploratory_probe": Strategy.EXPLORATORY,
        "greedy_exploit": Strategy.GREEDY,
        "balanced_adaptive": Strategy.BALANCED,
        "fallback_cascade": Strategy.FALLBACK_CASCADE,
    }
    primary_strategy = state_strategy_map.get(new_state, Strategy.BALANCED)
    if kinetic_alarm:
        # A flagged plateau routes through the exploratory tier regardless of
        # which state the boost transition landed on.
        primary_strategy = Strategy.EXPLORATORY
    metadata["primary_strategy"] = primary_strategy.name

    # --- Stage 3: Fallback routing ---
    router = FallbackRouter(max_depth=fallback_depth, ttl_seconds=route_ttl)
    resolved_strategy, cascade_depth = router.route(
        metadata=metadata,
        primary_strategy=primary_strategy,
        score=current_score,
        confidence_minimum=confidence_min,
    )
    metadata["resolved_strategy"] = resolved_strategy.name
    metadata["fallback_cascade_depth"] = cascade_depth
    if cascade_depth > 0:
        logger.info(
            "Fallback engaged: primary=%s -> resolved=%s (depth=%d)",
            primary_strategy.name,
            resolved_strategy.name,
            cascade_depth,
        )

    # --- Stage 4: Generate & rank candidate actions ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit:
        candidates = _generate_candidate_actions(metadata, resolved_strategy)

        cba = CostBenefitAnalyzer(discount_gamma=discount_gamma, penalty=transition_penalty)
        ranked = cba.rank_actions(
            candidates,
            current_step=current_cycle,
            current_transitions=fsm._transition_count,
        )
        metadata["ranked_actions"] = ranked
        logger.info("Ranked %d candidate actions", len(ranked))
    else:
        metadata["ranked_actions"] = []
        logger.warning("Wall-time exceeded before action ranking")

    # --- Stage 5: Push to priority queue and select best ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit and metadata.get("ranked_actions"):
        pq = EvaluationPriorityQueue(capacity=queue_capacity)
        for action in metadata["ranked_actions"]:
            item = EvaluationItem(
                score=float(action.get("score", 0.0)),
                cost=float(action.get("cost", 0.0)),
                label=str(action.get("label", "unknown")),
                payload=action,
            )
            pq.push(item)

        best = pq.pop()
        if best is not None:
            metadata["selected_action"] = best.to_dict()
            metadata["selected_action_label"] = best.label
            logger.info("Selected action: %s (net_value=%.4f)", best.label, best.net_value)
        else:
            metadata["selected_action"] = None
            metadata["selected_action_label"] = "none"

        metadata["priority_queue_stats"] = pq.stats()
    else:
        metadata["selected_action"] = None
        metadata["selected_action_label"] = "none"

    # --- Stage 6: Apply decision to metadata ---
    action_label = metadata.get("selected_action_label", "none")
    exploration_depth: int = int(metadata.get("exploration_depth", 1))

    if action_label == "increase_exploration_depth":
        metadata["exploration_depth"] = min(exploration_depth + 1, 100)
        metadata["strategy_mode"] = "escalate_complexity"
    elif action_label == "tighten_convergence_tolerance":
        metadata["strategy_mode"] = "tighten_convergence"
    elif action_label == "boost_mutation_sigma":
        metadata["strategy_mode"] = "boost_mutation"
    elif action_label == "reduce_population_size":
        metadata["strategy_mode"] = "conservative_reduce"
    elif action_label == "increase_patience":
        metadata["strategy_mode"] = "patience_hold"
    elif action_label == "random_hyperparameter_perturbation":
        metadata["strategy_mode"] = "random_perturb"
    elif action_label == "full_parameter_reset":
        metadata["strategy_mode"] = "full_reset"
        metadata["exploration_depth"] = 1
    elif action_label == "warm_bootstrap_reseed":
        metadata["strategy_mode"] = "warm_bootstrap"
    else:
        metadata["strategy_mode"] = "hold"

    # --- Stage 7: Persist the state-action-result trajectory ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit:
        recorded = router.record_experience(
            state={
                "current_score": current_score,
                "current_cycle": current_cycle,
                "fsm_state": metadata.get("fsm_state", "none"),
                "primary_strategy": metadata.get("primary_strategy", "none"),
                "resolved_strategy": metadata.get("resolved_strategy", "none"),
            },
            action=action_label,
            result=current_score,
            parameter_matrix=metadata.get("best_config"),
        )
        metadata["experience_recorded"] = recorded
    else:
        metadata["experience_recorded"] = False

    # --- Finalize ---
    metadata["fsm_snapshot"] = fsm.snapshot()
    metadata["fallback_router_stats"] = router.stats()

    wall_seconds = round(time.monotonic() - start_time, 6)
    metadata["decision_tree_wall_seconds"] = wall_seconds
    metadata["decision_tree_status"] = (
        "budget_exceeded" if wall_seconds > wall_limit else "complete"
    )

    logger.info(
        "Decision-tree pipeline finished in %.3fs — strategy=%s, action=%s",
        wall_seconds,
        resolved_strategy.name,
        action_label,
    )
    return metadata
