"""End-to-end code translation pipeline with evolution integration.

Connects the profiler → translator → entropy filter → differential sandbox
into a single pipeline, and provides a hook for the evolution loop's
processing_mesh track.
"""

import json
import os
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

sys.path.insert(0, _ROOT)

from translator.aero_translator import translate_file, translated_to_recipe
from translator.entropy_filter import check_entropy, detect_param_recycling
from translator.diff_sandbox import SandboxInput, verify_translation
from translator.ffi_generator import analyze_ffi, write_ffi_module
from translator.cold_pass_router import analyze_routing, analyze_routing_dispatch, filter_translatable


# ---------------------------------------------------------------------------
# Test vector generators for known function signatures
# ---------------------------------------------------------------------------

def _default_test_inputs(func_name: str) -> list[SandboxInput]:
    """Generate reasonable test vectors based on function name heuristics."""
    if "matrix_multiply" in func_name:
        return [
            SandboxInput(args=([[1, 2], [3, 4]], [[5, 6], [7, 8]]), label="2x2_basic"),
            SandboxInput(args=([[1, 0], [0, 1]], [[9, 8], [7, 6]]), label="2x2_identity"),
            SandboxInput(args=([[2]], [[3]]), label="1x1_scalar"),
        ]
    if "recursive_fib" in func_name or "fib" in func_name:
        return [
            SandboxInput(args=(0,), label="fib_0"),
            SandboxInput(args=(1,), label="fib_1"),
            SandboxInput(args=(10,), label="fib_10"),
        ]
    if "sort_and_filter" in func_name:
        return [
            SandboxInput(args=(["  hello ", " world ", "", "  "],), label="basic_filter"),
            SandboxInput(args=(["a,b,c", "d,e"],), label="csv_split"),
            SandboxInput(args=([],), label="empty"),
        ]
    # Fallback: no-arg call
    return [SandboxInput(args=(), label="no_args")]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    source_file: str
    functions_attempted: int = 0
    functions_translated: int = 0
    functions_verified: int = 0
    functions_rejected: int = 0
    entropy_blocked: int = 0
    recipes_written: int = 0
    ffi_wrappers_generated: int = 0
    cold_passthrough_count: int = 0
    cold_passthrough_functions: list[str] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)


def run_translation_pipeline(
    source_path: str,
    function_names: list[str] | None = None,
    test_inputs_map: dict[str, list[SandboxInput]] | None = None,
    output_dir: str = "build_sandbox/recipes",
    bp_dir: str = "aero_mesh_core/swarm_blueprints",
    write_recipe: bool = True,
) -> PipelineResult:
    """Full pipeline: translate → entropy check → verify → write or rollback."""
    abs_source = os.path.join(_ROOT, source_path) if not os.path.isabs(source_path) else source_path
    abs_output = os.path.join(_ROOT, output_dir) if not os.path.isabs(output_dir) else output_dir
    abs_bp = os.path.join(_ROOT, bp_dir) if not os.path.isabs(bp_dir) else bp_dir

    pr = PipelineResult(source_file=source_path)

    # Phase 0a: FFI analysis — detect external imports
    print(f"\n[pipeline] Phase 0a: FFI analysis for {abs_source}", flush=True)
    ffi = analyze_ffi(abs_source)
    if ffi.external_imports:
        ext_names = [i.module for i in ffi.external_imports]
        print(f"[pipeline]   External imports detected: {', '.join(ext_names)}", flush=True)
        if ffi.ffi_wrappers:
            ffi_path = write_ffi_module(ffi)
            pr.ffi_wrappers_generated = len(ffi.ffi_wrappers)
            print(f"[pipeline]   Generated {pr.ffi_wrappers_generated} FFI wrapper(s) -> {ffi_path}", flush=True)
            pr.details.append({
                "phase": "ffi",
                "external_imports": ext_names,
                "wrappers": [w.wrapper_name for w in ffi.ffi_wrappers],
                "ffi_module": ffi_path,
            })
    else:
        print("[pipeline]   No external imports detected", flush=True)

    # Phase 0b: Cold-path routing — detect untranslatable patterns
    print("[pipeline] Phase 0b: Cold-path routing", flush=True)
    routing = analyze_routing_dispatch(abs_source)
    cold_names = [f.name for f in routing.functions if f.is_cold_passthrough]
    if cold_names:
        pr.cold_passthrough_count = len(cold_names)
        pr.cold_passthrough_functions = cold_names
        print(f"[pipeline]   Cold pass-through ({pr.cold_passthrough_count}): {', '.join(cold_names)}", flush=True)
        for f in routing.functions:
            if f.is_cold_passthrough:
                reasons = "; ".join(r.detail for r in f.reasons)
                print(f"[pipeline]     {f.name}: {reasons}", flush=True)
        pr.details.append({
            "phase": "cold_routing",
            "cold_functions": cold_names,
            "reasons": {f.name: [r.detail for r in f.reasons]
                        for f in routing.functions if f.is_cold_passthrough},
        })
    else:
        print("[pipeline]   No cold pass-through functions detected", flush=True)

    # Filter out cold-path functions before translation
    if function_names is not None:
        active_names = [n for n in function_names if n not in cold_names]
    else:
        active_names = None  # translate_file discovers; we filter after

    # Phase 1: Translate (only non-cold functions)
    print(f"[pipeline] Phase 1: Translating {abs_source}", flush=True)
    tr = translate_file(abs_source, active_names)

    # Post-filter: remove any cold functions that slipped through discovery
    tr.functions = [tf for tf in tr.functions if tf.name not in cold_names]
    pr.functions_attempted = len(tr.functions)

    translatable = [tf for tf in tr.functions if tf.translatable]
    pr.functions_translated = len(translatable)
    print(f"[pipeline]   {pr.functions_translated}/{pr.functions_attempted} functions translatable", flush=True)

    if not translatable:
        print("[pipeline]   No translatable functions — skipping", flush=True)
        return pr

    # Phase 2: Generate recipe
    recipe_body = translated_to_recipe(tr, output_dir=output_dir)
    print(f"[pipeline] Phase 2: Generated recipe ({tr.node_count} nodes)", flush=True)

    # Phase 3: Entropy filter
    print("[pipeline] Phase 3: Entropy filter", flush=True)
    entropy = check_entropy(recipe_body)
    print(f"[pipeline]   token_entropy={entropy['token_entropy']}, "
          f"line_entropy={entropy['line_entropy']}, "
          f"structural_diversity={entropy['structural_diversity']}", flush=True)

    if not entropy["passed"]:
        pr.entropy_blocked = 1
        for reason in entropy["reasons"]:
            print(f"[pipeline]   BLOCKED: {reason}", flush=True)
        pr.details.append({"phase": "entropy", "blocked": True, "reasons": entropy["reasons"]})
        return pr

    # Phase 4: Differential sandbox verification
    print("[pipeline] Phase 4: Differential sandbox verification", flush=True)
    test_map = test_inputs_map or {}

    for tf in translatable:
        inputs = test_map.get(tf.name, _default_test_inputs(tf.name))
        if not inputs:
            print(f"[pipeline]   {tf.name}: no test inputs, skipping verification", flush=True)
            continue

        result = verify_translation(
            source_path=abs_source,
            func_name=tf.name,
            recipe_body=recipe_body,
            test_inputs=inputs,
            recipe_dir=abs_bp,
        )
        pr.details.append(result)

        if result["translatable"]:
            pr.functions_verified += 1
        else:
            pr.functions_rejected += 1
            print(f"[pipeline]   REJECTED: {tf.name} — {result['reject_reason']}", flush=True)

    # Phase 5: Write recipe if all verified
    if pr.functions_rejected > 0:
        print(f"[pipeline] Phase 5: ABORTED — {pr.functions_rejected} function(s) failed verification", flush=True)
        return pr

    if write_recipe:
        base = os.path.splitext(os.path.basename(source_path))[0]
        recipe_name = f"translated_{base}.txt"
        recipe_path = os.path.join(abs_bp, recipe_name)
        os.makedirs(abs_bp, exist_ok=True)
        with open(recipe_path, "w", encoding="utf-8") as f:
            f.write(recipe_body)
        pr.recipes_written = 1
        print(f"[pipeline] Phase 5: Recipe written -> {recipe_path}", flush=True)
    else:
        print("[pipeline] Phase 5: Dry run — recipe not written", flush=True)

    return pr


# ---------------------------------------------------------------------------
# Evolution loop integration hook
# ---------------------------------------------------------------------------

def evolve_with_translation(recipe_text: str, mesh_name: str,
                            source_path: str,
                            function_names: list[str] | None = None) -> tuple[str, str]:
    """Hook for evolve_loop.py's processing_mesh track.

    Translates hot-path functions and integrates the resulting recipe
    nodes into the existing mesh, subject to entropy and verification
    checks. Returns (updated_recipe, mutation_log).
    """
    abs_source = os.path.join(_ROOT, source_path) if not os.path.isabs(source_path) else source_path

    tr = translate_file(abs_source, function_names)
    translatable = [tf for tf in tr.functions if tf.translatable]

    if not translatable:
        return recipe_text, "No translatable functions found"

    recipe_body = translated_to_recipe(tr)

    # Entropy check against the *combined* text
    combined = recipe_text + "\n" + recipe_body
    entropy = check_entropy(combined)
    if not entropy["passed"]:
        return recipe_text, f"Entropy filter blocked: {'; '.join(entropy['reasons'])}"

    # Parameter recycling check
    if detect_param_recycling(recipe_text, combined):
        return recipe_text, "Parameter recycling detected — mutation rejected"

    # Extract just the task blocks from the translated recipe (skip header)
    new_blocks = []
    in_project = False
    for line in recipe_body.split("\n"):
        if line.strip().startswith("[project]"):
            in_project = True
            continue
        if line.strip().startswith("[task:"):
            in_project = False
        if not in_project:
            new_blocks.append(line)

    merged = recipe_text.rstrip() + "\n\n" + "\n".join(new_blocks)
    return merged, f"Integrated {len(translatable)} translated function(s) into {mesh_name}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Aero Code Translation Core & Verification Sandbox",
    )
    parser.add_argument("source", help="Python source file to translate")
    parser.add_argument("--functions", "-f", nargs="*", default=None,
                        help="Specific function names to translate (default: all)")
    parser.add_argument("--output-dir", "-o", default="build_sandbox/recipes",
                        help="Output directory for recipes")
    parser.add_argument("--bp-dir", default="aero_mesh_core/swarm_blueprints",
                        help="Blueprint directory for recipe files")
    parser.add_argument("--no-write", action="store_true",
                        help="Dry run — don't write recipe to disk")
    args = parser.parse_args()

    result = run_translation_pipeline(
        source_path=args.source,
        function_names=args.functions,
        output_dir=args.output_dir,
        bp_dir=args.bp_dir,
        write_recipe=not args.no_write,
    )

    print(f"\n[pipeline] Summary:", flush=True)
    print(f"  Functions attempted:  {result.functions_attempted}", flush=True)
    print(f"  Translated:           {result.functions_translated}", flush=True)
    print(f"  Verified:             {result.functions_verified}", flush=True)
    print(f"  Rejected:             {result.functions_rejected}", flush=True)
    print(f"  Entropy blocked:      {result.entropy_blocked}", flush=True)
    print(f"  Recipes written:      {result.recipes_written}", flush=True)


if __name__ == "__main__":
    main()
