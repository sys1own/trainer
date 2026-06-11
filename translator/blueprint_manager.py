"""Declarative Blueprint Engine for the Aero AutoDev system.

Reads profiler manifests, auto-generates a human-editable ``blueprint.aero``
file, and provides a state reconciliation loop that syncs the codebase to
match the blueprint's declared layout.
"""

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# Constants & guardrails (from REPO_BOOTSTRAP_CONTEXT.md)
# ---------------------------------------------------------------------------

_MAX_NODES_PER_DOMAIN = 25
_MAX_FAMILY_PER_DOMAIN = 5

# ---------------------------------------------------------------------------
# Blueprint data model
# ---------------------------------------------------------------------------

@dataclass
class NodeDecl:
    node_id: str
    source_file: str
    op: str
    detail: str
    needs: str | None = None
    family: str = "default"


@dataclass
class DomainDecl:
    name: str
    nodes: list[NodeDecl] = field(default_factory=list)


@dataclass
class Blueprint:
    project_name: str
    version: str
    source_manifest: str
    output_dir: str
    domains: dict[str, DomainDecl] = field(default_factory=dict)
    cold_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Domain classification heuristics
# ---------------------------------------------------------------------------

_INGRESS_PATTERNS = re.compile(
    r"(ingest|read|load|fetch|import|stream|scan|parse|recv|listen|accept|socket"
    r"|request|download|telemetry|sensor|input|stdin)", re.IGNORECASE,
)
_AGGREGATION_PATTERNS = re.compile(
    r"(aggregat|consolidat|merge|reduce|collect|summariz|accumulat|finalize"
    r"|export|output|write_file|freeze|seal|package|bundle|publish)", re.IGNORECASE,
)


def _classify_domain(file_entry: dict) -> str:
    """Heuristically assign a profiler file entry to a domain group."""
    rel = file_entry.get("relative_path", "")
    funcs = file_entry.get("functions", [])
    func_names = " ".join(f.get("name", "") for f in funcs)
    blob = rel + " " + func_names

    if _INGRESS_PATTERNS.search(blob):
        return "ingress"
    if _AGGREGATION_PATTERNS.search(blob):
        return "aggregation"
    return "processing"


# ---------------------------------------------------------------------------
# Auto-generation: manifest → blueprint.aero
# ---------------------------------------------------------------------------

def generate_blueprint(manifest_path: str,
                       project_name: str = "aero_autodev",
                       output_dir: str = "build_sandbox/recipes") -> Blueprint:
    """Read a profiler manifest JSON and produce a ``Blueprint`` object."""
    abs_manifest = os.path.join(_ROOT, manifest_path) if not os.path.isabs(manifest_path) else manifest_path

    with open(abs_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    bp = Blueprint(
        project_name=project_name,
        version="1.0",
        source_manifest=manifest_path,
        output_dir=output_dir,
        domains={
            "ingress": DomainDecl(name="ingress"),
            "processing": DomainDecl(name="processing"),
            "aggregation": DomainDecl(name="aggregation"),
        },
    )

    node_counter = 0

    for entry in manifest.get("files", []):
        classification = entry.get("classification", "cold")

        if classification == "cold":
            bp.cold_paths.append(entry["relative_path"])
            continue

        domain = _classify_domain(entry)
        dom = bp.domains[domain]

        if len(dom.nodes) >= _MAX_NODES_PER_DOMAIN:
            continue

        prev_id = None
        for func in entry.get("functions", []):
            if len(dom.nodes) >= _MAX_NODES_PER_DOMAIN:
                break

            node_counter += 1
            nid = f"node{node_counter}"

            family = _infer_family(func, domain)
            family_count = sum(1 for n in dom.nodes if n.family == family)
            if family_count >= _MAX_FAMILY_PER_DOMAIN:
                continue

            op = "call" if func.get("is_recursive") or func.get("array_string_ops", 0) > 3 else "print"
            detail = _build_detail(func, entry["relative_path"], op, domain)

            dom.nodes.append(NodeDecl(
                node_id=nid,
                source_file=entry["relative_path"],
                op=op,
                detail=detail,
                needs=prev_id,
                family=family,
            ))
            prev_id = nid

    return bp


def _infer_family(func: dict, domain: str) -> str:
    """Assign a family tag based on function characteristics and domain."""
    name = func.get("name", "").lower()
    if func.get("is_recursive"):
        return "recursive"
    if func.get("max_loop_depth", 0) >= 3:
        return "compute"
    if func.get("array_string_ops", 0) > 3:
        return "transform"
    if domain == "ingress":
        return "reader"
    if domain == "aggregation":
        return "collector"
    return "worker"


def _build_detail(func: dict, source: str, op: str, domain: str) -> str:
    """Build the recipe detail line for a node."""
    fname = func.get("name", "unknown")
    if op == "call":
        return (
            f'fn = write_file\n'
            f'args = "build_sandbox/mesh_outputs/{domain}_{fname}.dat", "mapped"\n'
            f'family = {_infer_family(func, domain)}'
        )
    family = _infer_family(func, domain)
    return f'text = "-- {family} | Translating {fname} from {source} --"'


# ---------------------------------------------------------------------------
# Serialization: Blueprint ↔ blueprint.aero text
# ---------------------------------------------------------------------------

def serialize_blueprint(bp: Blueprint) -> str:
    """Render a ``Blueprint`` to the human-editable ``.aero`` format."""
    lines = []
    lines.append("[project]")
    lines.append(f"name = {bp.project_name}")
    lines.append(f"version = {bp.version}")
    lines.append(f"source_manifest = {bp.source_manifest}")
    lines.append(f"output = {bp.output_dir}")
    lines.append("")

    for domain_name in ("ingress", "processing", "aggregation"):
        dom = bp.domains.get(domain_name)
        if not dom:
            continue
        lines.append(f"[domain:{domain_name}]")
        lines.append("")

        for node in dom.nodes:
            lines.append(f"[domain:{domain_name}:task:{node.node_id}]")
            lines.append(f"source = {node.source_file}")
            lines.append(f"op = {node.op}")
            lines.append(node.detail)
            if node.needs:
                lines.append(f"needs = {node.needs}")
            lines.append("")

    if bp.cold_paths:
        lines.append("[cold_paths]")
        for cp in bp.cold_paths:
            lines.append(f"exclude = {cp}")
        lines.append("")

    return "\n".join(lines)


def parse_blueprint(text: str) -> Blueprint:
    """Parse a ``blueprint.aero`` text file back into a ``Blueprint`` object."""
    bp = Blueprint(
        project_name="", version="1.0", source_manifest="",
        output_dir="build_sandbox/recipes",
    )

    current_domain = None
    current_node_id = None
    current_node_attrs: dict = {}

    def _flush_node():
        nonlocal current_node_id, current_node_attrs, current_domain
        if current_node_id and current_domain:
            detail_parts = []
            op = current_node_attrs.get("op", "print")
            for key in ("text", "fn", "args", "family"):
                if key in current_node_attrs:
                    detail_parts.append(f"{key} = {current_node_attrs[key]}")

            dom = bp.domains.setdefault(current_domain, DomainDecl(name=current_domain))
            family = current_node_attrs.get("family", "default")
            dom.nodes.append(NodeDecl(
                node_id=current_node_id,
                source_file=current_node_attrs.get("source", ""),
                op=op,
                detail="\n".join(detail_parts),
                needs=current_node_attrs.get("needs"),
                family=family,
            ))
        current_node_id = None
        current_node_attrs = {}

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Section headers
        m_project = re.match(r"^\[project\]$", line)
        m_domain = re.match(r"^\[domain:(\w+)\]$", line)
        m_task = re.match(r"^\[domain:(\w+):task:(\w+)\]$", line)
        m_cold = re.match(r"^\[cold_paths\]$", line)

        if m_project:
            _flush_node()
            current_domain = None
            continue

        if m_domain:
            _flush_node()
            current_domain = m_domain.group(1)
            bp.domains.setdefault(current_domain, DomainDecl(name=current_domain))
            continue

        if m_task:
            _flush_node()
            current_domain = m_task.group(1)
            current_node_id = m_task.group(2)
            current_node_attrs = {}
            continue

        if m_cold:
            _flush_node()
            current_domain = None
            current_node_id = None
            continue

        # Key-value pairs
        kv = re.match(r"^(\w+)\s*=\s*(.+)$", line)
        if kv:
            key, val = kv.group(1), kv.group(2).strip()
            if current_node_id:
                current_node_attrs[key] = val
            elif current_domain is None and key == "name":
                bp.project_name = val
            elif current_domain is None and key == "version":
                bp.version = val
            elif current_domain is None and key == "source_manifest":
                bp.source_manifest = val
            elif current_domain is None and key == "output":
                bp.output_dir = val
            elif key == "exclude":
                bp.cold_paths.append(val)

    _flush_node()
    return bp


# ---------------------------------------------------------------------------
# State reconciliation
# ---------------------------------------------------------------------------

@dataclass
class DeltaAction:
    action: str          # "create", "delete", "modify"
    target: str          # file path or recipe name
    reason: str
    detail: str = ""


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return ""


def _render_domain_recipe(bp: Blueprint, domain_name: str) -> str:
    """Render a single domain's nodes into a mesh recipe file."""
    dom = bp.domains.get(domain_name)
    if not dom or not dom.nodes:
        return ""

    mesh_name = f"{domain_name}_mesh"
    out_path = f"{bp.output_dir}/{mesh_name}.aeroc"
    header = f"[project]\nname = {mesh_name}\noutput = {out_path}\n\n"

    blocks = []
    for node in dom.nodes:
        block = f"[task:{node.node_id}]\nop = {node.op}\n{node.detail}\n"
        if node.needs:
            block += f"needs = {node.needs}\n"
        blocks.append(block)

    return header + "\n".join(blocks)


def reconcile(bp: Blueprint, target_root: str) -> list[DeltaAction]:
    """Calculate the delta between the blueprint's declared state and the
    actual files on disk under *target_root*.

    Returns a list of ``DeltaAction`` objects describing what must change.
    """
    abs_root = os.path.join(_ROOT, target_root) if not os.path.isabs(target_root) else target_root
    bp_dir = os.path.join(abs_root, "aero_mesh_core", "swarm_blueprints")
    out_dir = os.path.join(abs_root, bp.output_dir)
    deltas: list[DeltaAction] = []

    # --- Recipe files: blueprint declares which domain recipes should exist ---
    expected_recipes: dict[str, str] = {}
    for domain_name in ("ingress", "processing", "aggregation"):
        recipe_body = _render_domain_recipe(bp, domain_name)
        if recipe_body:
            fname = f"{domain_name}_mesh.txt"
            expected_recipes[fname] = recipe_body

    # Check existing recipes against expected
    os.makedirs(bp_dir, exist_ok=True)
    existing_files = set()
    for name in os.listdir(bp_dir):
        full = os.path.join(bp_dir, name)
        if os.path.isfile(full):
            existing_files.add(name)

    # Files to create or modify
    for fname, body in expected_recipes.items():
        full = os.path.join(bp_dir, fname)
        if fname not in existing_files:
            deltas.append(DeltaAction(
                action="create",
                target=os.path.join("aero_mesh_core/swarm_blueprints", fname),
                reason="Blueprint declares this domain recipe but file is missing",
                detail=body,
            ))
        else:
            with open(full, "r", encoding="utf-8") as f:
                current = f.read()
            if current.strip() != body.strip():
                deltas.append(DeltaAction(
                    action="modify",
                    target=os.path.join("aero_mesh_core/swarm_blueprints", fname),
                    reason="Recipe content diverges from blueprint declaration",
                    detail=body,
                ))

    # Files to delete (exist on disk but not in blueprint)
    for fname in existing_files:
        if fname not in expected_recipes:
            deltas.append(DeltaAction(
                action="delete",
                target=os.path.join("aero_mesh_core/swarm_blueprints", fname),
                reason="File exists on disk but is not declared in blueprint",
            ))

    # --- Output directory structure ---
    os.makedirs(out_dir, exist_ok=True)

    # --- Cold-path source files: verify they still exist (informational) ---
    for cp in bp.cold_paths:
        full = os.path.join(abs_root, cp)
        if not os.path.exists(full):
            deltas.append(DeltaAction(
                action="create",
                target=cp,
                reason="Cold-path source declared in blueprint but missing from disk (informational)",
            ))

    return deltas


# ---------------------------------------------------------------------------
# State modification: apply deltas
# ---------------------------------------------------------------------------

def apply_deltas(deltas: list[DeltaAction], target_root: str, *, dry_run: bool = False) -> list[str]:
    """Apply the delta actions to the filesystem. Returns a log of actions taken."""
    abs_root = os.path.join(_ROOT, target_root) if not os.path.isabs(target_root) else target_root
    log = []

    for delta in deltas:
        full = os.path.join(abs_root, delta.target)

        if delta.action == "create":
            if delta.detail:
                if dry_run:
                    log.append(f"[DRY-RUN] Would create: {delta.target}")
                else:
                    os.makedirs(os.path.dirname(full), exist_ok=True)
                    with open(full, "w", encoding="utf-8") as f:
                        f.write(delta.detail)
                    log.append(f"[CREATED] {delta.target}")
            else:
                log.append(f"[SKIPPED] {delta.target} — no content to write (informational)")

        elif delta.action == "modify":
            if dry_run:
                log.append(f"[DRY-RUN] Would modify: {delta.target}")
            else:
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(delta.detail)
                log.append(f"[MODIFIED] {delta.target}")

        elif delta.action == "delete":
            if dry_run:
                log.append(f"[DRY-RUN] Would delete: {delta.target}")
            else:
                if os.path.exists(full):
                    os.remove(full)
                    log.append(f"[DELETED] {delta.target}")
                else:
                    log.append(f"[SKIPPED] {delta.target} — already absent")

    return log


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_blueprint_pipeline(manifest_path: str,
                           blueprint_path: str = "blueprint.aero",
                           target_root: str = ".",
                           project_name: str = "aero_autodev",
                           apply: bool = False,
                           dry_run: bool = False) -> dict:
    """End-to-end pipeline: generate or load blueprint, reconcile, optionally apply."""
    abs_bp = os.path.join(_ROOT, blueprint_path) if not os.path.isabs(blueprint_path) else blueprint_path
    abs_root = os.path.join(_ROOT, target_root) if not os.path.isabs(target_root) else target_root

    # Phase 1: Generate or load blueprint
    if os.path.exists(abs_bp):
        print(f"[blueprint] Loading existing blueprint: {abs_bp}", flush=True)
        with open(abs_bp, "r", encoding="utf-8") as f:
            bp = parse_blueprint(f.read())
    else:
        print(f"[blueprint] Generating blueprint from manifest: {manifest_path}", flush=True)
        bp = generate_blueprint(manifest_path, project_name=project_name)
        text = serialize_blueprint(bp)
        with open(abs_bp, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[blueprint] Written: {abs_bp}", flush=True)

    # Phase 2: Reconcile
    print(f"[blueprint] Reconciling against: {abs_root}", flush=True)
    deltas = reconcile(bp, abs_root)

    creates = sum(1 for d in deltas if d.action == "create")
    modifies = sum(1 for d in deltas if d.action == "modify")
    deletes = sum(1 for d in deltas if d.action == "delete")
    print(f"[blueprint] Delta: {creates} create, {modifies} modify, {deletes} delete", flush=True)

    for d in deltas:
        print(f"  [{d.action.upper():8s}] {d.target} — {d.reason}", flush=True)

    # Phase 3: Apply (if requested)
    action_log = []
    if apply or dry_run:
        action_log = apply_deltas(deltas, abs_root, dry_run=dry_run)
        for entry in action_log:
            print(f"  {entry}", flush=True)

    return {
        "blueprint_path": abs_bp,
        "deltas": len(deltas),
        "creates": creates,
        "modifies": modifies,
        "deletes": deletes,
        "applied": apply and not dry_run,
        "log": action_log,
    }


# ---------------------------------------------------------------------------
# Rust FFI handle generation
# ---------------------------------------------------------------------------

def generate_rust_ffi_handle(function_name: str,
                             aeroc_module: str,
                             param_types: list[tuple[str, str]] | None = None,
                             return_type: str = "Vec<f64>") -> str:
    """Generate a thread-safe Rust FFI handle for binding .aeroc bytecode
    back into a native Rust binary compilation flow.

    Standardized: delegates to
    :func:`translator.ffi_codegen.generate_single_handle` so that every Rust
    FFI artefact in the project shares one ``extern "C"`` raw-pointer
    implementation (thread-safe handle, zero-allocation call frame, host
    alignment + lifetime preservation). The import is lazy so non-FFI uses of
    this module do not require the Tree-Sitter toolchain.
    """
    from translator.ffi_codegen import generate_single_handle

    return generate_single_handle(
        function_name=function_name,
        aeroc_module=aeroc_module,
        param_types=param_types,
        return_type=return_type,
    )


def write_rust_ffi_handles(functions: list[dict],
                           aeroc_module: str,
                           output_dir: str = "build_sandbox/rust_ffi") -> list[str]:
    """Generate Rust FFI handles for a list of translated functions.

    Each function dict should have: ``name``, and optionally
    ``params`` (list of (name, type) tuples) and ``return_type``.

    Returns a list of written file paths.
    """
    abs_dir = os.path.join(_ROOT, output_dir) if not os.path.isabs(output_dir) else output_dir
    os.makedirs(abs_dir, exist_ok=True)

    paths = []
    for func in functions:
        name = func["name"]
        params = func.get("params")
        ret_type = func.get("return_type", "Vec<f64>")

        code = generate_rust_ffi_handle(
            function_name=name,
            aeroc_module=aeroc_module,
            param_types=params,
            return_type=ret_type,
        )

        safe_name = name.replace("::", "_").replace(".", "_")
        out_path = os.path.join(abs_dir, f"ffi_{safe_name}.rs")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(code)
        paths.append(out_path)

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Aero AutoDev Declarative Blueprint Engine",
    )
    parser.add_argument("manifest", help="Path to profiler manifest JSON")
    parser.add_argument("--blueprint", "-b", default="blueprint.aero",
                        help="Path to blueprint.aero file (default: blueprint.aero)")
    parser.add_argument("--target", "-t", default=".",
                        help="Target root directory to reconcile against")
    parser.add_argument("--project", "-p", default="aero_autodev",
                        help="Project name for auto-generated blueprints")
    parser.add_argument("--apply", action="store_true",
                        help="Apply delta actions to the filesystem")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be applied without writing")
    args = parser.parse_args()

    results = run_blueprint_pipeline(
        manifest_path=args.manifest,
        blueprint_path=args.blueprint,
        target_root=args.target,
        project_name=args.project,
        apply=args.apply,
        dry_run=args.dry_run,
    )

    if results["deltas"] == 0:
        print("\n[blueprint] Codebase is in sync with blueprint.", flush=True)
    elif not results["applied"]:
        print(f"\n[blueprint] {results['deltas']} action(s) pending. Use --apply to execute or --dry-run to preview.", flush=True)


if __name__ == "__main__":
    main()
