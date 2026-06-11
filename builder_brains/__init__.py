# -*- coding: utf-8 -*-
"""
builder_brains — Machine Reasoning Engine

Exposes four production-grade sub-engines, each with a unified
``evaluate(metadata, hyper_params)`` pipeline entry point:

  * **compactor**        — AST tokenization, dead-code elimination, variable minification
  * **decision_tree**    — Heuristic state machine, priority queues, fallback routing
  * **scanner**          — Concurrent file scanning, regex profiling, crypto diffing
  * **parameter_tuner**  — Learning-rate annealing, evolutionary hyperparameter tuning
"""

from builder_brains.compactor import evaluate as compactor_evaluate
from builder_brains.decision_tree import evaluate as decision_tree_evaluate
from builder_brains.scanner import evaluate as scanner_evaluate
from builder_brains.parameter_tuner import evaluate as parameter_tuner_evaluate

__all__ = [
    "compactor_evaluate",
    "decision_tree_evaluate",
    "scanner_evaluate",
    "parameter_tuner_evaluate",
]
