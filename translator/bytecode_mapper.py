"""Map identified hot-paths into Aero mesh recipe text that the
meta_compiler can validate and the evolution loop can optimise."""

import os
from dataclasses import dataclass

_MAX_FAMILY_INSTANCES = 5
_MAX_NODES_PER_MESH = 25


@dataclass
class MeshRecipe:
    name: str
    output_path: str
    body: str


def _task_block(task_id: str, op: str, detail: str,
                needs: str | None = None) -> str:
    block = f"[task:{task_id}]\nop = {op}\n{detail}\n"
    if needs:
        block += f"needs = {needs}\n"
    return block


def hotpath_to_recipe(hotpath, *, output_dir: str = "build_sandbox/recipes") -> MeshRecipe:
    """Convert a single ``HotPath`` into a compilable mesh recipe.

    Generates an ingest task per source file (capped by guardrails) and a
    final aggregation task that depends on the last ingest node.
    """
    mesh_name = f"translated_{hotpath.pattern_id}"
    out_path = os.path.join(output_dir, f"{mesh_name}.aeroc")

    header = f"[project]\nname = {mesh_name}\noutput = {out_path}\n\n"

    blocks = []
    prev_id = None
    family_counts: dict[str, int] = {}
    node_counter = 0

    # init task
    blocks.append(_task_block(
        "init", "print",
        f'text = "-- Initializing translator pipeline for {hotpath.label} --"',
    ))
    prev_id = "init"
    node_counter += 1

    for src in hotpath.source_files:
        if node_counter >= _MAX_NODES_PER_MESH - 1:
            break

        family = "reader"
        family_counts.setdefault(family, 0)
        if family_counts[family] >= _MAX_FAMILY_INSTANCES:
            continue
        family_counts[family] += 1

        tid = f"node{node_counter}"
        basename = os.path.basename(src)
        blocks.append(_task_block(
            tid, "print",
            f'text = "-- reader | Scanning hot-path source: {basename} --"',
            needs=prev_id,
        ))
        prev_id = tid
        node_counter += 1

    # aggregation task
    if node_counter < _MAX_NODES_PER_MESH:
        blocks.append(_task_block(
            f"node{node_counter}", "call",
            f'fn = write_file\nargs = "build_sandbox/mesh_outputs/{mesh_name}_index.dat", "mapped"',
            needs=prev_id,
        ))

    body = header + "\n".join(blocks)
    return MeshRecipe(name=mesh_name, output_path=out_path, body=body)
