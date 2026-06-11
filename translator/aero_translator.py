"""Rule-based symbolic translator: maps Python source hot-path functions
into Aero VM primitive sequences (print/call ops in mesh recipe format).

Walks the Python AST and lowers control flow, writes, and logging outputs
into the Aero task-graph representation that meta_compiler can validate.
"""

import ast
import os
import textwrap
from dataclasses import dataclass, field

_MAX_NODES_PER_MESH = 25
_MAX_FAMILY_PER_DOMAIN = 5

# ---------------------------------------------------------------------------
# Aero IR (intermediate representation)
# ---------------------------------------------------------------------------

@dataclass
class AeroOp:
    """Single Aero VM primitive instruction."""
    op: str            # "print" or "call"
    detail: str        # the body text (text=... or fn=... args=...)
    label: str = ""    # human-readable description
    family: str = "translated"


@dataclass
class TranslatedFunction:
    """A source function lowered to a sequence of Aero ops."""
    name: str
    source_file: str
    ops: list[AeroOp] = field(default_factory=list)
    translatable: bool = True
    reject_reason: str = ""


@dataclass
class TranslationResult:
    """Full translation output for one source file."""
    source_file: str
    functions: list[TranslatedFunction] = field(default_factory=list)
    recipe_body: str = ""
    node_count: int = 0


# ---------------------------------------------------------------------------
# AST → Aero lowering rules
# ---------------------------------------------------------------------------

class _AeroLowering(ast.NodeVisitor):
    """Walk a function AST and emit Aero ops for translatable constructs."""

    def __init__(self, func_name: str, source_file: str):
        self.func_name = func_name
        self.source_file = source_file
        self.ops: list[AeroOp] = []
        self._loop_depth = 0
        self._unsupported: list[str] = []

    # --- Loops → compute family ops ---

    def visit_For(self, node):
        self._loop_depth += 1
        target = ast.dump(node.target) if hasattr(node, 'target') else "iter"
        family = "compute" if self._loop_depth >= 2 else "iterator"
        self.ops.append(AeroOp(
            op="print",
            detail=f'text = "-- {family} | {self.func_name}: loop depth {self._loop_depth} over {self._safe_name(node.target)} --"',
            label=f"loop-depth-{self._loop_depth}",
            family=family,
        ))
        self.generic_visit(node)
        self._loop_depth -= 1

    visit_While = visit_For

    # --- Function calls → call or print ops ---

    def visit_Call(self, node):
        callee = self._call_name(node)
        if callee in ("print", "logging", "log", "console"):
            self.ops.append(AeroOp(
                op="print",
                detail=f'text = "-- logger | {self.func_name}: log output via {callee} --"',
                label="log-output",
                family="logger",
            ))
        elif callee in ("open", "write", "write_file", "save", "dump"):
            args_preview = self._args_preview(node)
            self.ops.append(AeroOp(
                op="call",
                detail=f'fn = write_file\nargs = "build_sandbox/mesh_outputs/{self.func_name}_{callee}.dat", "translated"\nfamily = io',
                label=f"write-{callee}",
                family="io",
            ))
        elif callee in ("append", "extend", "insert", "push", "sort",
                        "split", "join", "replace", "strip", "filter",
                        "map", "reduce", "find", "index", "count"):
            self.ops.append(AeroOp(
                op="print",
                detail=f'text = "-- transform | {self.func_name}: array/string op {callee} --"',
                label=f"transform-{callee}",
                family="transform",
            ))
        else:
            self.ops.append(AeroOp(
                op="print",
                detail=f'text = "-- worker | {self.func_name}: call {callee} --"',
                label=f"call-{callee}",
                family="worker",
            ))
        self.generic_visit(node)

    # --- Assignments with compound ops ---

    def visit_AugAssign(self, node):
        op_name = type(node.op).__name__
        self.ops.append(AeroOp(
            op="print",
            detail=f'text = "-- compute | {self.func_name}: augmented assign ({op_name}) --"',
            label=f"augassign-{op_name}",
            family="compute",
        ))
        self.generic_visit(node)

    # --- Return → final output marker ---

    def visit_Return(self, node):
        self.ops.append(AeroOp(
            op="call",
            detail=f'fn = write_file\nargs = "build_sandbox/mesh_outputs/{self.func_name}_return.dat", "result"\nfamily = output',
            label="return-value",
            family="output",
        ))
        self.generic_visit(node)

    # --- Recursion detection (call to self) ---

    def visit_FunctionDef(self, node):
        self.generic_visit(node)

    # --- Helpers ---

    def _call_name(self, node) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return "unknown"

    def _safe_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        return "expr"

    def _args_preview(self, node) -> str:
        parts = []
        for arg in node.args[:2]:
            if isinstance(arg, ast.Constant):
                parts.append(repr(arg.value))
            elif isinstance(arg, ast.Name):
                parts.append(arg.id)
        return ", ".join(parts) if parts else "..."


# ---------------------------------------------------------------------------
# Translation pipeline
# ---------------------------------------------------------------------------

def translate_function(source: str, func_name: str,
                       source_file: str) -> TranslatedFunction:
    """Translate a single Python function into Aero ops."""
    try:
        tree = ast.parse(source, filename=source_file)
    except SyntaxError as e:
        return TranslatedFunction(
            name=func_name, source_file=source_file,
            translatable=False, reject_reason=f"SyntaxError: {e}",
        )

    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                func_node = node
                break

    if func_node is None:
        return TranslatedFunction(
            name=func_name, source_file=source_file,
            translatable=False, reject_reason=f"Function '{func_name}' not found",
        )

    lowering = _AeroLowering(func_name, source_file)
    lowering.visit(func_node)

    if not lowering.ops:
        return TranslatedFunction(
            name=func_name, source_file=source_file,
            translatable=False, reject_reason="No translatable constructs found",
        )

    return TranslatedFunction(
        name=func_name, source_file=source_file,
        ops=lowering.ops, translatable=True,
    )


def translate_file(source_path: str,
                   function_names: list[str] | None = None) -> TranslationResult:
    """Translate all (or specified) functions in a Python source file."""
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=source_path)
    except SyntaxError:
        return TranslationResult(source_file=source_path)

    if function_names is None:
        function_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    result = TranslationResult(source_file=source_path)
    for fname in function_names:
        tf = translate_function(source, fname, source_path)
        result.functions.append(tf)

    return result


# ---------------------------------------------------------------------------
# Recipe generation from translated functions
# ---------------------------------------------------------------------------

def translated_to_recipe(result: TranslationResult,
                         mesh_name: str = "",
                         output_dir: str = "build_sandbox/recipes") -> str:
    """Convert a TranslationResult into a compilable mesh recipe string."""
    if not mesh_name:
        base = os.path.splitext(os.path.basename(result.source_file))[0]
        mesh_name = f"translated_{base}"

    out_path = f"{output_dir}/{mesh_name}.aeroc"
    header = f"[project]\nname = {mesh_name}\noutput = {out_path}\n\n"

    blocks = []
    node_counter = 0
    prev_id = None
    family_counts: dict[str, int] = {}

    # Init task
    blocks.append(
        f'[task:init]\nop = print\n'
        f'text = "-- Initializing translated pipeline for {mesh_name} --"\n'
    )
    prev_id = "init"
    node_counter += 1

    for tf in result.functions:
        if not tf.translatable:
            continue

        for aop in tf.ops:
            if node_counter >= _MAX_NODES_PER_MESH - 1:
                break

            family_counts.setdefault(aop.family, 0)
            if family_counts[aop.family] >= _MAX_FAMILY_PER_DOMAIN:
                continue
            family_counts[aop.family] += 1

            nid = f"node{node_counter}"
            block = f"[task:{nid}]\nop = {aop.op}\n{aop.detail}\nneeds = {prev_id}\n"
            blocks.append(block)
            prev_id = nid
            node_counter += 1

    # Final aggregation task
    if node_counter < _MAX_NODES_PER_MESH and prev_id:
        nid = f"node{node_counter}"
        blocks.append(
            f"[task:{nid}]\nop = call\n"
            f'fn = write_file\nargs = "build_sandbox/mesh_outputs/{mesh_name}_final.dat", "complete"\n'
            f"needs = {prev_id}\n"
        )
        node_counter += 1

    body = header + "\n".join(blocks)
    result.recipe_body = body
    result.node_count = node_counter
    return body
