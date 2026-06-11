# -*- coding: utf-8 -*-
"""
compactor.py — Advanced Code Structural Tokenization & AST Compaction Engine

Implements:
  - AST-tree trimming via Python's `ast` module (recursive dead-branch pruning)
  - Dead-code elimination (unreachable-after-return, unused imports/assigns)
  - Safe variable minification (scope-aware alpha-renaming)
  - Structural tokenization (AST node-type frequency vectors)

Pipeline entry point: evaluate(metadata, hyper_params)
"""

import ast
import hashlib
import json
import logging
import math
import os
import string
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("builder_brains.compactor")

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


def _get_compactor_params(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("hyperparameter_weights", {}).get("compactor", {})


def _get_thresholds(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("thresholds", {})


def _get_cost_ceilings(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("execution_cost_ceilings", {})


# ---------------------------------------------------------------------------
# AST structural tokenization
# ---------------------------------------------------------------------------

class StructuralTokenizer(ast.NodeVisitor):
    """Walk an AST and produce a frequency vector of node types plus depth stats."""

    def __init__(self, max_depth: int = 256) -> None:
        self.node_counts: Counter = Counter()
        self.max_depth: int = max_depth
        self._current_depth: int = 0
        self._max_observed_depth: int = 0
        self._scope_stack: List[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        node_type_name: str = type(node).__name__
        self.node_counts[node_type_name] += 1

        if self._current_depth > self._max_observed_depth:
            self._max_observed_depth = self._current_depth

        if self._current_depth >= self.max_depth:
            logger.debug(
                "Max tree-walk recursion depth %d reached at node %s",
                self.max_depth,
                node_type_name,
            )
            return

        self._current_depth += 1
        super().generic_visit(node)
        self._current_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def get_token_vector(self) -> Dict[str, int]:
        return dict(self.node_counts)

    def get_depth_stats(self) -> Dict[str, int]:
        return {
            "max_observed_depth": self._max_observed_depth,
            "total_nodes": sum(self.node_counts.values()),
            "unique_node_types": len(self.node_counts),
        }


def tokenize_source(source: str, max_depth: int = 256) -> Dict[str, Any]:
    tree = ast.parse(source)
    tokenizer = StructuralTokenizer(max_depth=max_depth)
    tokenizer.visit(tree)
    return {
        "token_vector": tokenizer.get_token_vector(),
        "depth_stats": tokenizer.get_depth_stats(),
    }


# ---------------------------------------------------------------------------
# Dead-code elimination
# ---------------------------------------------------------------------------

class _NameCollector(ast.NodeVisitor):
    """Collect all Name-load references in a subtree."""

    def __init__(self) -> None:
        self.referenced: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.referenced.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.generic_visit(node)


def _collect_loaded_names(tree: ast.AST) -> Set[str]:
    collector = _NameCollector()
    collector.visit(tree)
    return collector.referenced


class DeadCodeEliminator(ast.NodeTransformer):
    """Remove provably unreachable code and unused simple assignments.

    Passes:
      1. Statements after unconditional return/raise/break/continue
      2. Unused import aliases (single-name imports only)
      3. Unused simple assignments (``x = <expr>`` where ``x`` is never loaded)
    """

    TERMINAL_TYPES: Tuple[type, ...] = (ast.Return, ast.Raise, ast.Break, ast.Continue)

    def __init__(self, elimination_depth: int = 4) -> None:
        self.elimination_depth: int = elimination_depth
        self._pass_number: int = 0
        self.removed_nodes: int = 0
        self._loaded_names: Set[str] = set()

    def set_loaded_names(self, names: Set[str]) -> None:
        self._loaded_names = names

    def _prune_body(self, body: List[ast.stmt]) -> List[ast.stmt]:
        pruned: List[ast.stmt] = []
        terminated = False
        for stmt in body:
            if terminated:
                self.removed_nodes += 1
                continue
            pruned.append(stmt)
            if isinstance(stmt, self.TERMINAL_TYPES):
                terminated = True
        return pruned if pruned else body

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = self._prune_body(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.body = self._prune_body(node.body)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.body = self._prune_body(node.body)
        self.generic_visit(node)
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        node.body = self._prune_body(node.body)
        if node.orelse:
            node.orelse = self._prune_body(node.orelse)
        self.generic_visit(node)
        return node

    def visit_Import(self, node: ast.Import) -> Optional[ast.Import]:
        remaining = [
            alias for alias in node.names
            if (alias.asname or alias.name) in self._loaded_names
        ]
        if not remaining:
            self.removed_nodes += 1
            return None
        node.names = remaining
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.ImportFrom]:
        if node.names and not any(a.name == "*" for a in node.names):
            remaining = [
                alias for alias in node.names
                if (alias.asname or alias.name) in self._loaded_names
            ]
            if not remaining:
                self.removed_nodes += 1
                return None
            node.names = remaining
        return node

    def visit_Assign(self, node: ast.Assign) -> Optional[ast.Assign]:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name: str = node.targets[0].id
            if target_name.startswith("_") and target_name not in self._loaded_names:
                self.removed_nodes += 1
                return None
        return node

    def run_passes(self, tree: ast.Module) -> ast.Module:
        for pass_idx in range(self.elimination_depth):
            self._pass_number = pass_idx
            self._loaded_names = _collect_loaded_names(tree)
            tree = self.visit(tree)
            ast.fix_missing_locations(tree)
        return tree


def eliminate_dead_code(
    source: str,
    depth: int = 4,
) -> Dict[str, Any]:
    tree = ast.parse(source)
    original_node_count = sum(1 for _ in ast.walk(tree))

    eliminator = DeadCodeEliminator(elimination_depth=depth)
    tree = eliminator.run_passes(tree)

    cleaned_source = ast.unparse(tree)
    final_node_count = sum(1 for _ in ast.walk(ast.parse(cleaned_source)))

    return {
        "cleaned_source": cleaned_source,
        "original_node_count": original_node_count,
        "final_node_count": final_node_count,
        "removed_node_count": eliminator.removed_nodes,
        "reduction_ratio": 1.0 - (final_node_count / max(original_node_count, 1)),
    }


# ---------------------------------------------------------------------------
# AST-tree trimming (prune low-confidence branches)
# ---------------------------------------------------------------------------

class ASTTreeTrimmer(ast.NodeTransformer):
    """Prune AST branches whose estimated complexity weight falls below a
    confidence floor.  Uses a simple heuristic: the ratio of child-node count
    to total-node count.  Branches below ``confidence_floor`` are replaced with
    ``pass`` stubs.
    """

    def __init__(
        self,
        confidence_floor: float = 0.60,
        aggressiveness: float = 0.72,
    ) -> None:
        self.confidence_floor: float = confidence_floor
        self.aggressiveness: float = aggressiveness
        self.trimmed_branches: int = 0
        self._total_weight: float = 0.0

    @staticmethod
    def _subtree_weight(node: ast.AST) -> int:
        return sum(1 for _ in ast.walk(node))

    def _should_trim(self, node: ast.AST, parent_weight: int) -> bool:
        child_weight = self._subtree_weight(node)
        ratio = child_weight / max(parent_weight, 1)
        threshold = self.confidence_floor * (1.0 - self.aggressiveness * 0.5)
        return ratio < threshold

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        parent_weight = self._subtree_weight(node)
        new_body: List[ast.stmt] = []
        for child in node.body:
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                if self._should_trim(child, parent_weight):
                    self.trimmed_branches += 1
                    pass_node = ast.Pass()
                    ast.copy_location(child, pass_node)
                    new_body.append(pass_node)
                    continue
            new_body.append(child)
        node.body = new_body if new_body else [ast.Pass()]
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        parent_weight = self._subtree_weight(node)
        new_body: List[ast.stmt] = []
        for child in node.body:
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                if self._should_trim(child, parent_weight):
                    self.trimmed_branches += 1
                    pass_node = ast.Pass()
                    ast.copy_location(child, pass_node)
                    new_body.append(pass_node)
                    continue
            new_body.append(child)
        node.body = new_body if new_body else [ast.Pass()]
        self.generic_visit(node)
        return node


def trim_ast_tree(
    source: str,
    confidence_floor: float = 0.60,
    aggressiveness: float = 0.72,
) -> Dict[str, Any]:
    tree = ast.parse(source)
    original_count = sum(1 for _ in ast.walk(tree))

    trimmer = ASTTreeTrimmer(
        confidence_floor=confidence_floor,
        aggressiveness=aggressiveness,
    )
    tree = trimmer.visit(tree)
    ast.fix_missing_locations(tree)

    trimmed_source = ast.unparse(tree)
    final_count = sum(1 for _ in ast.walk(ast.parse(trimmed_source)))

    return {
        "trimmed_source": trimmed_source,
        "original_node_count": original_count,
        "final_node_count": final_count,
        "trimmed_branches": trimmer.trimmed_branches,
    }


# ---------------------------------------------------------------------------
# Safe variable minification (scope-aware alpha-renaming)
# ---------------------------------------------------------------------------

class _ScopeAnalyzer(ast.NodeVisitor):
    """Build a mapping of locally-defined names per function scope."""

    BUILTIN_NAMES: Set[str] = {
        "print", "len", "range", "enumerate", "zip", "map", "filter",
        "int", "float", "str", "bool", "list", "dict", "set", "tuple",
        "isinstance", "issubclass", "type", "super", "property",
        "staticmethod", "classmethod", "open", "hasattr", "getattr",
        "setattr", "delattr", "None", "True", "False", "Exception",
        "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
        "OSError", "IOError", "AttributeError", "ImportError", "StopIteration",
        "NotImplementedError", "AssertionError",
    }

    def __init__(self) -> None:
        self.scopes: Dict[str, Set[str]] = defaultdict(set)
        self._current_scope: str = "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        parent_scope = self._current_scope
        self._current_scope = f"{parent_scope}.{node.name}"
        for arg in node.args.args:
            self.scopes[self._current_scope].add(arg.arg)
        self.generic_visit(node)
        self._current_scope = parent_scope

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        parent_scope = self._current_scope
        self._current_scope = f"{parent_scope}.{node.name}"
        for arg in node.args.args:
            self.scopes[self._current_scope].add(arg.arg)
        self.generic_visit(node)
        self._current_scope = parent_scope

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.scopes[self._current_scope].add(target.id)
        self.generic_visit(node)


class _NameMinifier:
    """Generate short, collision-free replacement names."""

    def __init__(self, salt_bits: int = 32) -> None:
        self._counter: int = 0
        self._salt_bits: int = salt_bits
        self._mapping: Dict[str, str] = {}

    def _next_short_name(self) -> str:
        chars = string.ascii_lowercase
        idx = self._counter
        self._counter += 1
        if idx < len(chars):
            return f"_{chars[idx]}"
        parts: List[str] = []
        while idx >= 0:
            parts.append(chars[idx % len(chars)])
            idx = idx // len(chars) - 1
        return "_" + "".join(reversed(parts))

    def get_minified(self, original: str, scope: str) -> str:
        key = f"{scope}::{original}"
        if key not in self._mapping:
            self._mapping[key] = self._next_short_name()
        return self._mapping[key]

    @property
    def total_renames(self) -> int:
        return len(self._mapping)


class VariableMinifier(ast.NodeTransformer):
    """Rename local variables inside function bodies to short identifiers.

    Safety rules:
      - Never rename function/class definitions themselves
      - Never rename names that appear as global/nonlocal declarations
      - Never rename Python builtins
      - Never rename dunder names (__init__, __name__, etc.)
      - Never rename single-underscore ``_`` (convention for unused)
    """

    def __init__(
        self,
        scope_map: Dict[str, Set[str]],
        entropy_cap: float = 0.85,
        salt_bits: int = 32,
    ) -> None:
        self._scope_map: Dict[str, Set[str]] = scope_map
        self._entropy_cap: float = entropy_cap
        self._minifier: _NameMinifier = _NameMinifier(salt_bits=salt_bits)
        self._current_scope: str = "<module>"
        self._protected_names: Set[str] = set()
        self._global_nonlocal: Set[str] = set()

    def _is_safe_to_rename(self, name: str) -> bool:
        if name.startswith("__") and name.endswith("__"):
            return False
        if name == "_" or name in _ScopeAnalyzer.BUILTIN_NAMES:
            return False
        if name in self._protected_names or name in self._global_nonlocal:
            return False
        return True

    def visit_Global(self, node: ast.Global) -> ast.Global:
        self._global_nonlocal.update(node.names)
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Nonlocal:
        self._global_nonlocal.update(node.names)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self._protected_names.add(node.name)
        parent_scope = self._current_scope
        self._current_scope = f"{parent_scope}.{node.name}"
        saved_gn = set(self._global_nonlocal)
        self._global_nonlocal = set()
        self.generic_visit(node)
        self._global_nonlocal = saved_gn
        self._current_scope = parent_scope
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self._protected_names.add(node.name)
        parent_scope = self._current_scope
        self._current_scope = f"{parent_scope}.{node.name}"
        saved_gn = set(self._global_nonlocal)
        self._global_nonlocal = set()
        self.generic_visit(node)
        self._global_nonlocal = saved_gn
        self._current_scope = parent_scope
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self._protected_names.add(node.name)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        scope_locals = self._scope_map.get(self._current_scope, set())
        if node.id in scope_locals and self._is_safe_to_rename(node.id):
            node.id = self._minifier.get_minified(node.id, self._current_scope)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if self._is_safe_to_rename(node.arg):
            scope_locals = self._scope_map.get(self._current_scope, set())
            if node.arg in scope_locals:
                node.arg = self._minifier.get_minified(node.arg, self._current_scope)
        return node

    @property
    def total_renames(self) -> int:
        return self._minifier.total_renames


def minify_variables(
    source: str,
    entropy_cap: float = 0.85,
    salt_bits: int = 32,
) -> Dict[str, Any]:
    tree = ast.parse(source)

    analyzer = _ScopeAnalyzer()
    analyzer.visit(tree)

    minifier = VariableMinifier(
        scope_map=dict(analyzer.scopes),
        entropy_cap=entropy_cap,
        salt_bits=salt_bits,
    )
    tree = minifier.visit(tree)
    ast.fix_missing_locations(tree)

    minified_source = ast.unparse(tree)
    original_len = len(source)
    minified_len = len(minified_source)
    compression = 1.0 - (minified_len / max(original_len, 1))

    return {
        "minified_source": minified_source,
        "total_renames": minifier.total_renames,
        "original_length": original_len,
        "minified_length": minified_len,
        "compression_ratio": round(compression, 6),
    }


# ---------------------------------------------------------------------------
# Compaction metrics aggregation
# ---------------------------------------------------------------------------

def _compute_structural_fingerprint(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _compute_entropy(token_vector: Dict[str, int]) -> float:
    total = sum(token_vector.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in token_vector.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 6)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def evaluate(metadata: Dict[str, Any], hyper_params: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full compaction pipeline on source code stored in metadata.

    Parameters
    ----------
    metadata : dict
        Must contain ``"source_code"`` (str).  Receives enrichment keys for
        every stage: tokenization, dead-code elimination, AST trimming,
        variable minification.
    hyper_params : dict
        Runtime overrides merged on top of ``build_manifest.json`` defaults.
        Recognised keys mirror the ``compactor`` section of the manifest.

    Returns
    -------
    dict
        The enriched metadata dictionary.
    """
    start_time: float = time.monotonic()
    logger.info("Compactor pipeline started")

    manifest: Dict[str, Any] = _load_manifest()
    params: Dict[str, Any] = {**_get_compactor_params(manifest), **hyper_params}
    thresholds: Dict[str, Any] = _get_thresholds(manifest)
    ceilings: Dict[str, Any] = _get_cost_ceilings(manifest)

    wall_limit: float = ceilings.get("compactor_max_wall_seconds", 30.0)
    source: str = metadata.get("source_code", "")

    if not source.strip():
        logger.warning("Empty source_code in metadata — returning early")
        metadata["compactor_status"] = "skipped_empty_source"
        metadata["compactor_wall_seconds"] = 0.0
        return metadata

    # --- Stage 1: Structural tokenization ---
    try:
        max_depth: int = int(params.get("tree_walk_max_recursion", 256))
        tok_result = tokenize_source(source, max_depth=max_depth)
        metadata["structural_tokens"] = tok_result["token_vector"]
        metadata["depth_stats"] = tok_result["depth_stats"]
        metadata["source_entropy"] = _compute_entropy(tok_result["token_vector"])
        logger.info(
            "Tokenization complete — %d unique types, depth %d",
            tok_result["depth_stats"]["unique_node_types"],
            tok_result["depth_stats"]["max_observed_depth"],
        )
    except SyntaxError as exc:
        logger.error("Tokenization failed — source has syntax errors: %s", exc)
        metadata["compactor_status"] = "tokenization_syntax_error"
        metadata["compactor_error"] = str(exc)
        metadata["compactor_wall_seconds"] = round(time.monotonic() - start_time, 6)
        return metadata

    # --- Stage 2: Dead-code elimination ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit:
        try:
            elim_depth: int = int(params.get("dead_code_elimination_depth", 4))
            dce_result = eliminate_dead_code(source, depth=elim_depth)
            source = dce_result["cleaned_source"]
            metadata["dce_original_nodes"] = dce_result["original_node_count"]
            metadata["dce_final_nodes"] = dce_result["final_node_count"]
            metadata["dce_removed"] = dce_result["removed_node_count"]
            metadata["dce_reduction_ratio"] = dce_result["reduction_ratio"]
            logger.info(
                "Dead-code elimination: removed %d nodes (%.2f%% reduction)",
                dce_result["removed_node_count"],
                dce_result["reduction_ratio"] * 100,
            )
        except Exception as exc:
            logger.error("Dead-code elimination failed: %s", exc)
            metadata["dce_error"] = str(exc)
    else:
        logger.warning("Wall-time budget exhausted before DCE stage")

    # --- Stage 3: AST tree trimming ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit:
        try:
            confidence_floor: float = float(
                params.get("node_prune_confidence_floor", 0.60)
            )
            aggressiveness: float = float(
                params.get("ast_trim_aggressiveness", 0.72)
            )
            trim_result = trim_ast_tree(
                source,
                confidence_floor=confidence_floor,
                aggressiveness=aggressiveness,
            )
            source = trim_result["trimmed_source"]
            metadata["trim_original_nodes"] = trim_result["original_node_count"]
            metadata["trim_final_nodes"] = trim_result["final_node_count"]
            metadata["trim_branches_removed"] = trim_result["trimmed_branches"]
            logger.info(
                "AST trim: pruned %d low-confidence branches",
                trim_result["trimmed_branches"],
            )
        except Exception as exc:
            logger.error("AST tree trimming failed: %s", exc)
            metadata["trim_error"] = str(exc)
    else:
        logger.warning("Wall-time budget exhausted before AST-trim stage")

    # --- Stage 4: Variable minification ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit:
        entropy_cap: float = float(params.get("minification_entropy_cap", 0.85))
        current_entropy = metadata.get("source_entropy", 0.0)

        if current_entropy <= entropy_cap:
            try:
                salt_bits: int = int(params.get("identifier_collision_salt_bits", 32))
                min_result = minify_variables(
                    source,
                    entropy_cap=entropy_cap,
                    salt_bits=salt_bits,
                )
                source = min_result["minified_source"]
                metadata["minify_renames"] = min_result["total_renames"]
                metadata["minify_compression"] = min_result["compression_ratio"]
                logger.info(
                    "Variable minification: %d renames, %.2f%% compression",
                    min_result["total_renames"],
                    min_result["compression_ratio"] * 100,
                )
            except Exception as exc:
                logger.error("Variable minification failed: %s", exc)
                metadata["minify_error"] = str(exc)
        else:
            logger.info(
                "Skipping minification — entropy %.4f exceeds cap %.4f",
                current_entropy,
                entropy_cap,
            )
            metadata["minify_skipped_reason"] = "entropy_above_cap"
    else:
        logger.warning("Wall-time budget exhausted before minification stage")

    # --- Finalize ---
    metadata["compacted_source"] = source
    metadata["compacted_fingerprint"] = _compute_structural_fingerprint(source)
    compaction_floor: float = float(thresholds.get("compaction_ratio_floor", 0.10))
    total_reduction = metadata.get("dce_reduction_ratio", 0.0)
    metadata["compaction_passed_threshold"] = total_reduction >= compaction_floor

    wall_seconds = round(time.monotonic() - start_time, 6)
    metadata["compactor_wall_seconds"] = wall_seconds
    metadata["compactor_status"] = (
        "budget_exceeded" if wall_seconds > wall_limit else "complete"
    )

    logger.info(
        "Compactor pipeline finished in %.3fs — status=%s",
        wall_seconds,
        metadata["compactor_status"],
    )
    return metadata
