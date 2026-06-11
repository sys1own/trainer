# -*- coding: utf-8 -*-
"""
blueprint_parser.py

Parse the monolithic `blueprint.aero` configuration and normalize it into a
runtime `build_context`. Invalid or unreadable blueprints automatically fall
back to stable coefficients from `builder_brains/build_manifest.json`.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("blueprint_parser")

_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "builder_brains", "build_manifest.json"
)
_REQUIRED_SECTIONS = ("graph", "compiler", "cortex")


class BlueprintParseError(ValueError):
    """Raised when blueprint parsing or validation fails."""


def load_stable_manifest(manifest_path: str = _MANIFEST_PATH) -> Dict[str, Any]:
    """Load stable parameters from build_manifest.json."""
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.error("Failed to load fallback build_manifest.json: %s", exc)
        return {}


def detect_cycles(dependencies: Dict[str, List[str]]) -> List[str]:
    """Detect cycles in a dependency graph using DFS traversal."""
    visited: Dict[str, int] = {}
    parent: Dict[str, str] = {}

    for node in dependencies:
        visited[node] = 0

    def dfs(node: str) -> List[str]:
        visited[node] = 1
        for dependency in dependencies.get(node, []):
            if dependency not in visited:
                continue
            if visited[dependency] == 1:
                cycle = [dependency, node]
                current = node
                while current in parent and parent[current] != dependency:
                    current = parent[current]
                    cycle.append(current)
                cycle.reverse()
                return cycle
            if visited[dependency] == 0:
                parent[dependency] = node
                cycle = dfs(dependency)
                if cycle:
                    return cycle
        visited[node] = 2
        return []

    for node in dependencies:
        if visited[node] == 0:
            cycle = dfs(node)
            if cycle:
                return cycle
    return []


def parse_literal(value: str) -> Any:
    """Parse booleans, numbers, JSON literals, and plain strings."""
    cleaned = value.strip()
    if not cleaned:
        return ""
    if cleaned.lower() in ("true", "yes", "on"):
        return True
    if cleaned.lower() in ("false", "no", "off"):
        return False
    try:
        if "." in cleaned or "e" in cleaned.lower():
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        pass
    if (
        (cleaned.startswith("[") and cleaned.endswith("]"))
        or (cleaned.startswith("{") and cleaned.endswith("}"))
        or (cleaned.startswith('"') and cleaned.endswith('"'))
    ):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            if cleaned.startswith('"') and cleaned.endswith('"'):
                return cleaned[1:-1]
    return cleaned


def _coerce_dependency_map(raw_dependencies: Any) -> Dict[str, List[str]]:
    if not isinstance(raw_dependencies, dict):
        raise BlueprintParseError("[graph] dependencies must be a JSON object")

    dependency_map: Dict[str, List[str]] = {}
    for node, raw_value in raw_dependencies.items():
        node_name = str(node).strip()
        if not node_name:
            raise BlueprintParseError("[graph] dependencies contains an empty node name")
        if isinstance(raw_value, list):
            dependency_map[node_name] = [
                str(item).strip() for item in raw_value if str(item).strip()
            ]
            continue
        if isinstance(raw_value, str):
            dependency_map[node_name] = [
                item.strip() for item in raw_value.split(",") if item.strip()
            ]
            continue
        raise BlueprintParseError(
            f"[graph] dependencies for '{node_name}' must be a list or comma-separated string"
        )
    return dependency_map


def _validate_sections(sections: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    missing = [section for section in _REQUIRED_SECTIONS if section not in sections]
    if missing:
        raise BlueprintParseError(f"Missing required section(s): {', '.join(missing)}")

    graph = sections["graph"]
    if "targets" not in graph:
        raise BlueprintParseError("[graph] targets is required")
    if "dependencies" not in graph:
        raise BlueprintParseError("[graph] dependencies is required")

    targets = graph["targets"]
    if not isinstance(targets, list) or not targets:
        raise BlueprintParseError("[graph] targets must be a non-empty JSON list")

    normalized_targets: List[str] = []
    target_metadata: List[Dict[str, Any]] = []
    for target in targets:
        if isinstance(target, dict):
            target_name = str(target.get("name", "")).strip()
            if not target_name:
                raise BlueprintParseError("[graph] target objects must include a non-empty name")
            normalized_targets.append(target_name)
            target_metadata.append(dict(target))
            continue
        target_name = str(target).strip()
        if not target_name:
            raise BlueprintParseError("[graph] targets cannot contain empty values")
        normalized_targets.append(target_name)
        target_metadata.append({"name": target_name})
    graph["targets"] = normalized_targets
    graph["target_metadata"] = target_metadata

    dependency_map = _coerce_dependency_map(graph["dependencies"])
    for target in normalized_targets:
        dependency_map.setdefault(target, [])

    unknown_dependencies = sorted(
        {
            dependency
            for deps in dependency_map.values()
            for dependency in deps
            if dependency not in dependency_map
        }
    )
    if unknown_dependencies:
        raise BlueprintParseError(
            "Unknown dependency target(s): " + ", ".join(unknown_dependencies)
        )

    numeric_fields = (
        ("compiler", "tier_shifting_hotness_threshold", int),
        ("compiler", "hotspot_loop_unroll_depth", int),
        ("compiler", "pipeline_budget_seconds", (int, float)),
        ("compiler", "max_memory_mb", int),
        ("cortex", "mutation_entropy_clamp_threshold", (int, float)),
        ("cortex", "total_cooperating_agents", int),
        ("cortex", "heuristic_exploration_depth", int),
        ("cortex", "inter_core_ring_buffer_capacity", int),
    )
    for section_name, key, expected_type in numeric_fields:
        value = sections[section_name].get(key)
        if value is not None and not isinstance(value, expected_type):
            raise BlueprintParseError(f"[{section_name}] {key} has an invalid type")

    bool_fields = (
        ("graph", "allow_partial_graph"),
        ("compiler", "aot_boundary_check_elimination"),
        ("compiler", "vector_intrinsics_auto_generation"),
        ("cortex", "numa_node_locality_binding"),
    )
    for section_name, key in bool_fields:
        value = sections[section_name].get(key)
        if value is not None and not isinstance(value, bool):
            raise BlueprintParseError(f"[{section_name}] {key} must be a boolean")

    return dependency_map


def parse_blueprint_content(content: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """Parse blueprint content and validate the monolithic schema."""
    sections: Dict[str, Dict[str, Any]] = {}
    current_section: Optional[str] = None

    for idx, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue

        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            if not current_section:
                raise BlueprintParseError(f"Line {idx}: Empty section header")
            if current_section in sections:
                raise BlueprintParseError(f"Line {idx}: Duplicate section [{current_section}]")
            sections[current_section] = {}
            continue

        if "=" in line or ":" in line:
            if current_section is None:
                raise BlueprintParseError(f"Line {idx}: Key-value pair found before any section")
            separator = "=" if "=" in line else ":"
            key, value = line.split(separator, 1)
            normalized_key = key.strip()
            if not normalized_key:
                raise BlueprintParseError(f"Line {idx}: Empty key")
            sections[current_section][normalized_key] = parse_literal(value)
            continue

        raise BlueprintParseError(f"Line {idx}: Unrecognized layout structure: {line}")

    dependencies = _validate_sections(sections)
    cycle = detect_cycles(dependencies)
    if cycle:
        raise BlueprintParseError(f"Invalid instruction loop detected: {' -> '.join(cycle)}")
    return sections, dependencies


def create_fallback_context(manifest: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """Generate a stable fallback build_context from manifest values."""
    logger.warning("Reverting to last known stable build manifest state: %s", error_msg)

    weights = manifest.get("hyperparameter_weights", {})
    tuner_params = weights.get("parameter_tuner", {})
    scheduler_params = manifest.get("execution_cost_ceilings", {})
    parameters = manifest.get("parameters", {})

    build_context = {
        "workspace_status": "reverted_fallback",
        "fallback_reason": error_msg,
        "timestamp": time.time(),
        "compilation_targets": ["scanner", "decision_tree", "parameter_tuner"],
        "dependency_matrix": {},
        "active_optimizer_flags": {
            "profile_guided_optimization": "enabled_strict",
            "tier_shifting_hotness_threshold": 100,
            "hotspot_loop_unroll_depth": 32,
            "aot_boundary_check_elimination": True,
            "vector_intrinsics_auto_generation": True,
            "consensus_protocol": "raft_driven_mutation_lock",
            "mutation_entropy_clamp_threshold": float(tuner_params.get("mutation_sigma", 0.05)),
        },
        "environment_targets": {
            "execution_mode": "lock_free_polling_wheel_realtime",
            "core_affinity_mask": "0xFFFF",
            "numa_node_locality_binding": True,
            "inter_core_ring_buffer_capacity": 262144,
        },
        "resource_metrics": {
            "pipeline_budget_seconds": float(
                scheduler_params.get("total_pipeline_budget_seconds", 120.0)
            ),
            "max_memory_mb": int(scheduler_params.get("max_memory_mb", 2048)),
            "elapsed_seconds": {},
        },
        "node_configurations": {},
        "graph": {
            "entrypoint": "orchestrator",
            "targets": ["scanner", "decision_tree", "parameter_tuner"],
            "dependencies": {
                "scanner": [],
                "decision_tree": ["scanner"],
                "parameter_tuner": ["decision_tree"],
            },
            "workspace_mode": "fallback_manifest",
            "allow_partial_graph": False,
        },
    }

    for key, value in parameters.items():
        if key.startswith("tuned_"):
            build_context["active_optimizer_flags"][key.replace("tuned_", "")] = value

    return build_context


def parse_blueprint(blueprint_path: str, manifest_path: str = _MANIFEST_PATH) -> Dict[str, Any]:
    """Load blueprint.aero, validate it, and generate a normalized build_context."""
    stable_manifest = load_stable_manifest(manifest_path)

    if not os.path.exists(blueprint_path):
        return create_fallback_context(
            stable_manifest, f"Blueprint file not found at: {blueprint_path}"
        )

    try:
        with open(blueprint_path, "r", encoding="utf-8") as fh:
            content = fh.read()

        sections, dependencies = parse_blueprint_content(content)
        graph_section = sections["graph"]
        compiler_section = sections["compiler"]
        cortex_section = sections["cortex"]

        return {
            "workspace_status": "stable_active",
            "timestamp": time.time(),
            "compilation_targets": graph_section["targets"],
            "dependency_matrix": dependencies,
            "active_optimizer_flags": {
                "profile_guided_optimization": compiler_section.get(
                    "profile_guided_optimization", "enabled_strict"
                ),
                "tier_shifting_hotness_threshold": compiler_section.get(
                    "tier_shifting_hotness_threshold", 100
                ),
                "hotspot_loop_unroll_depth": compiler_section.get(
                    "hotspot_loop_unroll_depth", 32
                ),
                "aot_boundary_check_elimination": compiler_section.get(
                    "aot_boundary_check_elimination", True
                ),
                "vector_intrinsics_auto_generation": compiler_section.get(
                    "vector_intrinsics_auto_generation", True
                ),
                "consensus_protocol": cortex_section.get(
                    "consensus_protocol", "raft_driven_mutation_lock"
                ),
                "mutation_entropy_clamp_threshold": cortex_section.get(
                    "mutation_entropy_clamp_threshold", 0.05
                ),
            },
            "environment_targets": {
                "execution_mode": cortex_section.get(
                    "execution_mode", "lock_free_polling_wheel_realtime"
                ),
                "core_affinity_mask": cortex_section.get("core_affinity_mask", "0xFFFF"),
                "numa_node_locality_binding": cortex_section.get(
                    "numa_node_locality_binding", True
                ),
                "inter_core_ring_buffer_capacity": cortex_section.get(
                    "inter_core_ring_buffer_capacity", 262144
                ),
            },
            "resource_metrics": {
                "pipeline_budget_seconds": float(
                    compiler_section.get("pipeline_budget_seconds", 120.0)
                ),
                "max_memory_mb": int(compiler_section.get("max_memory_mb", 2048)),
                "elapsed_seconds": {
                    target: 0.0 for target in graph_section["targets"]
                },
            },
            "node_configurations": {},
            "graph": {
                "entrypoint": graph_section.get("entrypoint", "orchestrator"),
                "targets": graph_section["targets"],
                "target_metadata": graph_section.get("target_metadata", []),
                "dependencies": dependencies,
                "workspace_mode": graph_section.get("workspace_mode", "incremental"),
                "allow_partial_graph": graph_section.get("allow_partial_graph", False),
            },
        }
    except Exception as exc:
        logger.exception("Blueprint parsing failed for %s", blueprint_path)
        return create_fallback_context(stable_manifest, f"Parser failure: {exc}")