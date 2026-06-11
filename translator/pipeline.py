"""End-to-end translation pipeline: scan -> identify -> map -> compile."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from translator.hotpath_scanner import scan_directory, identify_hotpaths
from translator.bytecode_mapper import hotpath_to_recipe
from meta_compiler import compile_recipe


def run_pipeline(target_dir: str, output_dir: str = "build_sandbox/recipes",
                 bp_dir: str = "aero_mesh_core/swarm_blueprints") -> dict:
    """Execute the full scan-identify-map-compile pipeline.

    Returns a summary dict with counts and any errors encountered.
    """
    abs_target = os.path.join(_ROOT, target_dir) if not os.path.isabs(target_dir) else target_dir
    abs_output = os.path.join(_ROOT, output_dir) if not os.path.isabs(output_dir) else output_dir
    abs_bp = os.path.join(_ROOT, bp_dir) if not os.path.isabs(bp_dir) else bp_dir

    os.makedirs(abs_output, exist_ok=True)
    os.makedirs(abs_bp, exist_ok=True)

    # --- Phase 1: Scan ---
    scanned = scan_directory(abs_target)
    print(f"[translator] Scanned {len(scanned)} files in {target_dir}", flush=True)

    # --- Phase 2: Identify hot-paths ---
    hotpaths = identify_hotpaths(scanned)
    print(f"[translator] Identified {len(hotpaths)} hot-path group(s)", flush=True)

    results = {"scanned": len(scanned), "hotpaths": len(hotpaths),
               "recipes_generated": 0, "compiled_ok": 0, "errors": []}

    # --- Phase 3: Map to recipes ---
    for hp in hotpaths:
        recipe = hotpath_to_recipe(hp, output_dir=output_dir)
        recipe_path = os.path.join(abs_bp, f"{recipe.name}.txt")
        with open(recipe_path, "w", encoding="utf-8") as f:
            f.write(recipe.body)
        results["recipes_generated"] += 1
        print(f"[translator] Generated recipe: {recipe.name} ({len(hp.source_files)} sources)", flush=True)

        # --- Phase 4: Compile ---
        try:
            compile_recipe(recipe_path)
            results["compiled_ok"] += 1
        except Exception as exc:
            results["errors"].append({"recipe": recipe.name, "error": str(exc)})
            print(f"[translator] Compile FAILED for {recipe.name}: {exc}", flush=True)

    print(f"[translator] Pipeline complete: {results['compiled_ok']}/{results['recipes_generated']} recipes compiled OK", flush=True)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aero Codebase Translator Pipeline")
    parser.add_argument("target", help="Target directory to scan for source files")
    parser.add_argument("--output-dir", default="build_sandbox/recipes",
                        help="Output directory for compiled bytecode bundles")
    parser.add_argument("--bp-dir", default="aero_mesh_core/swarm_blueprints",
                        help="Blueprint storage directory")
    args = parser.parse_args()

    results = run_pipeline(args.target, args.output_dir, args.bp_dir)
    if results["errors"]:
        print(f"\n[translator] {len(results['errors'])} error(s) encountered.")
        sys.exit(1)


if __name__ == "__main__":
    main()
