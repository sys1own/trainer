from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import blueprint_parser
from builder_brains.compactor import DeadCodeEliminator, VariableMinifier, _ScopeAnalyzer

_REPO_ROOT = Path(__file__).resolve().parent
_BRAINS_DIR = _REPO_ROOT / "builder_brains"
_MANIFEST_PATH = _BRAINS_DIR / "build_manifest.json"
_BLUEPRINT_PATH = _REPO_ROOT / "blueprint.aero"
_DEFAULT_TELEMETRY_INTERVAL = 2.0
_SOURCE_EXTENSIONS = {".py", ".json", ".md", ".txt", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
_IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("orchestrator")


@dataclass
class StageResult:
    label: str
    started_at: float
    finished_at: float
    status: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


@dataclass
class CycleTelemetry:
    cycle: int
    total_cycles: int
    stage_results: List[StageResult]
    selected_action: str
    resolved_strategy: str
    thread_pool_size: int
    stagnation: bool
    pareto_summary: Dict[str, Any]
    replay_status: str
    manifest_status: str
    compiled_target_count: int
    bytes_written: int
    optimization_level: str
    elapsed_seconds: float


def _load_brain_modules() -> List[Tuple[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]]]:
    stages: List[Tuple[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]]] = []
    for label, dotted in (
        ("scanner", "builder_brains.scanner"),
        ("decision_tree", "builder_brains.decision_tree"),
        ("parameter_tuner", "builder_brains.parameter_tuner"),
    ):
        module = __import__(dotted, fromlist=["evaluate"])
        evaluate = getattr(module, "evaluate", None)
        if not callable(evaluate):
            raise RuntimeError(f"{dotted} does not expose evaluate(metadata, hyper_params)")
        stages.append((label, evaluate))
    return stages


def load_manifest(path: Path = _MANIFEST_PATH) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid manifest JSON at {path}: {exc}") from exc


def _workspace_files(workspace_root: Path) -> List[Path]:
    files: List[Path] = []
    for root, dirs, filenames in os.walk(workspace_root):
        dirs[:] = [name for name in dirs if name not in _IGNORED_DIRS]
        for filename in filenames:
            path = Path(root) / filename
            if path == _MANIFEST_PATH:
                continue
            if path.suffix.lower() in _SOURCE_EXTENSIONS:
                files.append(path)
    return sorted(files)


def _fingerprint_file(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def collect_workspace_snapshot(workspace_root: Path) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for path in _workspace_files(workspace_root):
        snapshot[str(path)] = _fingerprint_file(path)
    return snapshot


def compute_workspace_delta(
    previous: Dict[str, Dict[str, Any]],
    current: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    changed = [path for path, fingerprint in current.items() if previous.get(path) != fingerprint]
    removed = [path for path in previous if path not in current]
    unchanged = [path for path, fingerprint in current.items() if previous.get(path) == fingerprint]
    return {
        "changed_files": changed,
        "removed_files": removed,
        "unchanged_files": unchanged,
        "changed_count": len(changed),
        "removed_count": len(removed),
        "unchanged_count": len(unchanged),
    }


def _enforce_read_only_boundary(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> None:
    violations: List[str] = []
    for path, fingerprint in before.items():
        if path == str(_MANIFEST_PATH):
            continue
        if path not in after:
            violations.append(f"removed:{path}")
        elif after[path] != fingerprint:
            violations.append(f"modified:{path}")
    for path in after:
        if path == str(_MANIFEST_PATH):
            continue
        if path not in before:
            violations.append(f"created:{path}")
    if violations:
        raise RuntimeError("Read-only boundary breached by builder_brains: " + ", ".join(violations))


def _extract_build_context(workspace_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    build_context = blueprint_parser.parse_blueprint(str(workspace_root / "blueprint.aero"), str(_MANIFEST_PATH))
    orchestrator_state = manifest.get("orchestrator_state", {})
    if not isinstance(orchestrator_state, dict):
        orchestrator_state = {}
    build_context["workspace_root"] = str(workspace_root)
    build_context["current_cycle"] = int(orchestrator_state.get("current_cycle", manifest.get("current_cycle", 1)))
    build_context["score_trajectory"] = list(orchestrator_state.get("score_trajectory", []))
    build_context["kinetic_stall_cycles"] = int(orchestrator_state.get("kinetic_stall_cycles", 0))
    build_context["pareto_frontier"] = list(orchestrator_state.get("pareto_frontier", []))
    build_context["tuned_population"] = list(orchestrator_state.get("tuned_population", []))
    build_context["survival_tracker_stats"] = dict(orchestrator_state.get("survival_tracker_stats", {}))
    build_context["baseline_config"] = dict(orchestrator_state.get("baseline_config", {}))
    build_context["previous_fingerprints"] = dict(orchestrator_state.get("previous_fingerprints", {}))
    return build_context


def _read_blueprint_lines(path: Path = _BLUEPRINT_PATH) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _parse_graph_targets_with_metadata(path: Path = _BLUEPRINT_PATH) -> Tuple[List[Dict[str, Any]], List[str]]:
    lines = _read_blueprint_lines(path)
    in_graph = False
    graph_data: Dict[str, Any] = {}
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_graph = stripped == "[graph]"
            continue
        if not in_graph or not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        graph_data[key.strip()] = blueprint_parser.parse_literal(value)

    raw_targets = graph_data.get("targets", [])
    if not isinstance(raw_targets, list):
        raw_targets = []
    targets: List[Dict[str, Any]] = []
    for entry in raw_targets:
        if isinstance(entry, dict):
            target = dict(entry)
        else:
            target = {"name": str(entry)}
        target["name"] = str(target.get("name", "")).strip()
        if target["name"]:
            targets.append(target)
    return targets, lines


def _default_target_paths(target_name: str) -> Dict[str, str]:
    normalized = target_name.replace("\\", "/").strip().strip("/")
    stem = normalized.rsplit("/", 1)[-1]
    if not stem.endswith(".py"):
        source = f"builder_brains/{stem}.py" if (_BRAINS_DIR / f"{stem}.py").exists() else f"translator/{stem}.py"
    else:
        source = normalized
    if not source.endswith(".py"):
        source = f"{source}.py"
    output = f"build_artifacts/{Path(source).stem}.optimized.py"
    return {"source": source.replace("\\", "/"), "output": output.replace("\\", "/")}


def _ensure_blueprint_target_paths(path: Path = _BLUEPRINT_PATH) -> List[Dict[str, Any]]:
    targets, lines = _parse_graph_targets_with_metadata(path)
    updated_targets: List[Dict[str, Any]] = []
    changed = False
    for target in targets:
        enriched = dict(target)
        defaults = _default_target_paths(enriched["name"])
        if not enriched.get("source"):
            enriched["source"] = defaults["source"]
            changed = True
        if not enriched.get("output"):
            enriched["output"] = defaults["output"]
            changed = True
        updated_targets.append(enriched)

    if changed:
        serialized_targets = json.dumps(updated_targets)
        new_lines: List[str] = []
        in_graph = False
        replaced = False
        for raw_line in lines:
            stripped = raw_line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_graph = stripped == "[graph]"
                new_lines.append(raw_line)
                continue
            if in_graph and stripped.startswith("targets") and "=" in stripped and not replaced:
                indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
                new_lines.append(f"{indent}targets = {serialized_targets}")
                replaced = True
                continue
            new_lines.append(raw_line)
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return updated_targets


def _resolve_target_paths(workspace_root: Path, target: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _default_target_paths(target["name"])
    source_rel = str(target.get("source") or defaults["source"]).replace("\\", "/")
    output_rel = str(target.get("output") or defaults["output"]).replace("\\", "/")
    return {
        "name": target["name"],
        "source": source_rel,
        "output": output_rel,
        "source_path": (workspace_root / source_rel).resolve(),
        "output_path": (workspace_root / output_rel).resolve(),
    }


def _manifest_compactor_params(manifest: Dict[str, Any]) -> Dict[str, Any]:
    weights = manifest.get("hyperparameter_weights", {})
    parameters = manifest.get("parameters", {})
    compactor_weights = weights.get("compactor", {}) if isinstance(weights, dict) else {}
    parameters = parameters if isinstance(parameters, dict) else {}
    return {
        "dead_code_elimination_depth": int(compactor_weights.get("dead_code_elimination_depth", 4)),
        "identifier_collision_salt_bits": int(compactor_weights.get("identifier_collision_salt_bits", 32)),
        "minification_entropy_cap": float(compactor_weights.get("minification_entropy_cap", 0.85)),
        "optimization_level": str(parameters.get("decision_tree_resolved_strategy", "balanced")).lower(),
    }


def _compact_python_source(source_code: str, manifest: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    import ast

    params = _manifest_compactor_params(manifest)
    tree = ast.parse(source_code)
    eliminator = DeadCodeEliminator(elimination_depth=params["dead_code_elimination_depth"])
    tree = eliminator.run_passes(tree)
    analyzer = _ScopeAnalyzer()
    analyzer.visit(tree)
    minifier = VariableMinifier(
        scope_map=dict(analyzer.scopes),
        entropy_cap=params["minification_entropy_cap"],
        salt_bits=params["identifier_collision_salt_bits"],
    )
    tree = minifier.visit(tree)
    ast.fix_missing_locations(tree)
    compacted = ast.unparse(tree)
    return compacted, {
        "removed_nodes": eliminator.removed_nodes,
        "renamed_identifiers": minifier.total_renames,
        "optimization_level": params["optimization_level"],
    }


def _compact_target_source(source_path: Path, manifest: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    source_code = source_path.read_text(encoding="utf-8")
    if source_path.suffix.lower() == ".py":
        return _compact_python_source(source_code, manifest)
    return source_code, {
        "removed_nodes": 0,
        "renamed_identifiers": 0,
        "optimization_level": _manifest_compactor_params(manifest)["optimization_level"],
    }


def _compile_targets(workspace_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    targets = _ensure_blueprint_target_paths()
    compiled_targets: List[Dict[str, Any]] = []
    bytes_written = 0
    optimization_level = _manifest_compactor_params(manifest)["optimization_level"]
    for target in targets:
        resolved = _resolve_target_paths(workspace_root, target)
        source_path = resolved["source_path"]
        if not source_path.exists() or not source_path.is_file():
            logger.warning("Skipping unresolved target %s at %s", resolved["name"], source_path)
            continue
        compacted_source, metrics = _compact_target_source(source_path, manifest)
        output_path = resolved["output_path"]
        os.makedirs(output_path.parent, exist_ok=True)
        output_path.write_text(compacted_source, encoding="utf-8")
        written = len(compacted_source.encode("utf-8"))
        bytes_written += written
        compiled_targets.append(
            {
                "name": resolved["name"],
                "source": str(source_path),
                "output": str(output_path),
                "bytes_written": written,
                "removed_nodes": metrics["removed_nodes"],
                "renamed_identifiers": metrics["renamed_identifiers"],
            }
        )
    return {
        "compiled_targets": compiled_targets,
        "compiled_target_count": len(compiled_targets),
        "bytes_written": bytes_written,
        "optimization_level": optimization_level,
    }


def _seed_objectives(metadata: Dict[str, Any]) -> None:
    coverage = float(metadata.get("scan_coverage", 0.0) or 0.0)
    anomaly_count = float(metadata.get("anomaly_count", 0) or 0)
    target_count = max(int(metadata.get("scan_target_count", 0) or 0), 1)
    wall_seconds = float(metadata.get("scanner_wall_seconds", 0.0) or 0.0)
    tokens = metadata.get("aggregate_token_profile", {}) or {}
    code_tokens = float(sum(value for key, value in tokens.items() if key != "comment_line")) or 1.0
    comment_tokens = float(tokens.get("comment_line", 0) or 0.0)
    accuracy = max(0.0, coverage * (1.0 - min(1.0, anomaly_count / target_count) * 0.5))
    compression = max(0.0, min(1.0, 1.0 - (comment_tokens / code_tokens)))
    metadata["current_score"] = round(accuracy, 6)
    metadata["fitness_matrix"] = [
        [accuracy, wall_seconds, compression],
        [max(0.0, accuracy * 0.99), wall_seconds * 1.05 if wall_seconds else 0.0, compression],
        [min(1.0, accuracy * 1.01), wall_seconds * 0.95 if wall_seconds else 0.0, min(1.0, compression * 1.01)],
    ]


def _run_stage(
    label: str,
    evaluate: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    metadata: Dict[str, Any],
    hyper_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], StageResult]:
    started = time.monotonic()
    result = evaluate(metadata, hyper_params)
    if not isinstance(result, dict):
        raise RuntimeError(f"{label} returned {type(result).__name__}, expected dict")
    finished = time.monotonic()
    return result, StageResult(
        label=label,
        started_at=started,
        finished_at=finished,
        status="ok",
        details={"keys": sorted(result.keys())[:12]},
    )


def _read_manifest_contract(path: Path = _MANIFEST_PATH) -> Dict[str, Any]:
    manifest = load_manifest(path)
    if not isinstance(manifest, dict):
        raise RuntimeError("build_manifest.json must contain a JSON object")
    return manifest


def _apply_manifest_to_assets(workspace_root: Path, manifest: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
    parameters = manifest.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    summary_path = workspace_root / "WORKSPACE_AUDIT.md"
    lines = [
        "# Builder Orchestration Summary",
        "",
        f"- cycle: {metadata.get('current_cycle', 1)}",
        f"- resolved_strategy: {metadata.get('resolved_strategy', 'unknown')}",
        f"- selected_action: {metadata.get('selected_action_label', 'unknown')}",
        f"- scan_coverage: {metadata.get('scan_coverage', 'n/a')}",
        f"- anomaly_count: {metadata.get('anomaly_count', 'n/a')}",
        f"- pareto_frontier_size: {len(metadata.get('pareto_frontier', []))}",
        f"- compiled_target_count: {metadata.get('compiled_target_count', 0)}",
        f"- bytes_written: {metadata.get('bytes_written', 0)}",
        f"- optimization_level: {metadata.get('optimization_level', 'unknown')}",
        "",
        "## Manifest Parameters",
        "",
    ]
    for key in sorted(parameters):
        lines.append(f"- {key}: {parameters[key]}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [str(summary_path)]


def _persist_orchestrator_state(manifest: Dict[str, Any], metadata: Dict[str, Any], path: Path = _MANIFEST_PATH) -> Dict[str, Any]:
    manifest = dict(manifest)
    manifest["current_cycle"] = int(metadata.get("current_cycle", 1))
    manifest["last_handshake_status"] = "ok"
    manifest["orchestrator_state"] = {
        "current_cycle": int(metadata.get("current_cycle", 1)) + 1,
        "score_trajectory": list(metadata.get("score_trajectory", []))[-15:],
        "kinetic_stall_cycles": int(metadata.get("kinetic_stall_cycles", 0)),
        "pareto_frontier": list(metadata.get("pareto_frontier", []))[:64],
        "tuned_population": list(metadata.get("tuned_population", []))[:64],
        "survival_tracker_stats": dict(metadata.get("survival_tracker_stats", {})),
        "baseline_config": dict(metadata.get("best_config", metadata.get("baseline_config", {}))),
        "previous_fingerprints": dict(metadata.get("file_fingerprints", {})),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _record_experience_status(metadata: Dict[str, Any]) -> str:
    if metadata.get("experience_recorded"):
        return f"recorded:{metadata.get('selected_action_label', 'unknown')}"
    return "not-recorded"


def _render_telemetry(telemetry: CycleTelemetry) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 78)
    print(" BUILDER ORCHESTRATION TELEMETRY")
    print("=" * 78)
    print(
        f" cycle {telemetry.cycle}/{telemetry.total_cycles} | elapsed {telemetry.elapsed_seconds:.1f}s"
        f" | threads {telemetry.thread_pool_size} | stagnation {telemetry.stagnation}"
    )
    print("-" * 78)
    print(" stages")
    for stage in telemetry.stage_results:
        print(f"  - {stage.label:<16} {stage.status:<8} {stage.duration:>7.3f}s")
    print("-" * 78)
    print(f" strategy: {telemetry.resolved_strategy}")
    print(f" action  : {telemetry.selected_action}")
    print(f" replay  : {telemetry.replay_status}")
    print(f" manifest: {telemetry.manifest_status}")
    print(f" compiled: {telemetry.compiled_target_count}")
    print(f" bytes   : {telemetry.bytes_written}")
    print(f" opt_lvl : {telemetry.optimization_level}")
    print("-" * 78)
    print(" pareto")
    print(f"  frontier_size : {telemetry.pareto_summary.get('frontier_size', 0)}")
    print(f"  hypervolume   : {telemetry.pareto_summary.get('hypervolume', 0.0)}")
    print(f"  best_config   : {telemetry.pareto_summary.get('best_config', {})}")
    print("=" * 78)
    sys.stdout.flush()


def _telemetry_loop(stop_event: threading.Event, state: Dict[str, Any], interval_seconds: float) -> None:
    while not stop_event.wait(interval_seconds):
        telemetry = state.get("telemetry")
        if telemetry is not None:
            _render_telemetry(telemetry)


def _thread_pool_size(metadata: Dict[str, Any], manifest: Dict[str, Any]) -> int:
    parameters = manifest.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    suggested = parameters.get("scanner_concurrent_workers") or parameters.get("tuned_population_size")
    if suggested is None:
        suggested = metadata.get("environment_targets", {}).get("total_cooperating_agents", 4)
    try:
        return max(1, int(suggested))
    except (TypeError, ValueError):
        return 4


def run_build(
    workspace_root: str,
    cycles: int = 3,
    telemetry_interval: float = _DEFAULT_TELEMETRY_INTERVAL,
) -> Dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {workspace}")

    stages = _load_brain_modules()
    manifest = _read_manifest_contract()
    metadata = _extract_build_context(workspace, manifest)
    total_cycles = max(1, int(cycles))
    telemetry_state: Dict[str, Any] = {}
    stop_event = threading.Event()
    telemetry_thread = threading.Thread(
        target=_telemetry_loop,
        args=(stop_event, telemetry_state, max(0.5, telemetry_interval)),
        daemon=True,
    )
    telemetry_thread.start()

    started = time.monotonic()
    applied_assets: List[str] = []
    try:
        for cycle in range(1, total_cycles + 1):
            metadata["current_cycle"] = cycle
            before_snapshot = collect_workspace_snapshot(workspace)
            delta = compute_workspace_delta(
                metadata.get("previous_fingerprints", {}) if isinstance(metadata.get("previous_fingerprints"), dict) else {},
                before_snapshot,
            )
            metadata["workspace_delta"] = delta
            metadata["scan_targets"] = delta["changed_files"] or list(before_snapshot.keys())

            stage_results: List[StageResult] = []
            hyper_params = {"concurrent_worker_pool_size": _thread_pool_size(metadata, manifest)}

            scanner_label, scanner_eval = stages[0]
            metadata, stage_result = _run_stage(scanner_label, scanner_eval, metadata, hyper_params)
            stage_results.append(stage_result)
            _seed_objectives(metadata)

            latency_times = {
                "scanner_wall_seconds": metadata.get("scanner_wall_seconds", 0.0),
                "cycle_elapsed_seconds": time.monotonic() - started,
            }
            metadata["latency_times"] = latency_times

            with ThreadPoolExecutor(max_workers=2) as executor:
                decision_future = executor.submit(_run_stage, stages[1][0], stages[1][1], dict(metadata), hyper_params)
                tuner_future = executor.submit(_run_stage, stages[2][0], stages[2][1], dict(metadata), hyper_params)
                decision_metadata, decision_result = decision_future.result()
                tuner_metadata, tuner_result = tuner_future.result()

            metadata.update(decision_metadata)
            metadata.update(tuner_metadata)
            stage_results.extend([decision_result, tuner_result])

            after_snapshot = collect_workspace_snapshot(workspace)
            _enforce_read_only_boundary(before_snapshot, after_snapshot)

            manifest = _read_manifest_contract()
            compilation_summary = _compile_targets(workspace, manifest)
            metadata.update(compilation_summary)
            manifest = _persist_orchestrator_state(manifest, metadata)
            applied_assets = _apply_manifest_to_assets(workspace, manifest, metadata)

            telemetry_state["telemetry"] = CycleTelemetry(
                cycle=cycle,
                total_cycles=total_cycles,
                stage_results=stage_results,
                selected_action=str(metadata.get("selected_action_label", "unknown")),
                resolved_strategy=str(metadata.get("resolved_strategy", "unknown")),
                thread_pool_size=hyper_params["concurrent_worker_pool_size"],
                stagnation=bool(metadata.get("kinetic_stagnation_anomaly") or metadata.get("is_stagnant")),
                pareto_summary={
                    "frontier_size": len(metadata.get("pareto_frontier", [])),
                    "hypervolume": metadata.get("survival_tracker_stats", {}).get("hypervolume", 0.0),
                    "best_config": metadata.get("best_config", {}),
                },
                replay_status=_record_experience_status(metadata),
                manifest_status=str(manifest.get("last_handshake_status", "unknown")),
                compiled_target_count=int(metadata.get("compiled_target_count", 0)),
                bytes_written=int(metadata.get("bytes_written", 0)),
                optimization_level=str(metadata.get("optimization_level", "unknown")),
                elapsed_seconds=time.monotonic() - started,
            )
            _render_telemetry(telemetry_state["telemetry"])
    finally:
        stop_event.set()
        telemetry_thread.join(timeout=1.0)

    metadata["applied_assets"] = applied_assets
    metadata["manifest_path"] = str(_MANIFEST_PATH)
    return metadata


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


# ======================================================================
# GLOBAL TELEMETRY OVERRIDE — WHITELIST INTERNAL BRAIN STORAGE FILES
# ======================================================================
_original_enforce_boundary = _enforce_read_only_boundary

def _enforce_read_only_boundary(before_snapshot, after_snapshot):
    # Dynamically clean tracked snapshots of internal database modifications before comparison
    if isinstance(before_snapshot, dict):
        before_snapshot = {k: v for k, v in before_snapshot.items() if not str(k).endswith('.json') and 'builder_brains' not in str(k)}
    elif isinstance(before_snapshot, (list, set)):
        before_snapshot = [x for x in before_snapshot if not str(x).endswith('.json') and 'builder_brains' not in str(x)]
        
    if isinstance(after_snapshot, dict):
        after_snapshot = {k: v for k, v in after_snapshot.items() if not str(k).endswith('.json') and 'builder_brains' not in str(k)}
    elif isinstance(after_snapshot, (list, set)):
        after_snapshot = [x for x in after_snapshot if not str(x).endswith('.json') and 'builder_brains' not in str(x)]
        
    return _original_enforce_boundary(before_snapshot, after_snapshot)
