"""FFI wrapper auto-generation for external library dependencies.

When the profiler identifies hot-path functions that depend on external
imports (numpy, scipy, ctypes, etc.), this module generates lightweight
FFI wrappers that allow Aero bytecode to delegate to native system
libraries without attempting to translate their internals.
"""

import ast
import importlib
import os
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# Known stdlib modules (not external)
# ---------------------------------------------------------------------------

_STDLIB_MODULES = frozenset([
    "abc", "argparse", "ast", "asyncio", "atexit", "base64", "bisect",
    "builtins", "calendar", "cmath", "codecs", "collections", "colorsys",
    "compileall", "concurrent", "configparser", "contextlib", "copy",
    "copyreg", "csv", "ctypes", "dataclasses", "datetime", "decimal",
    "difflib", "dis", "doctest", "email", "enum", "errno", "faulthandler",
    "filecmp", "fileinput", "fnmatch", "fractions", "ftplib", "functools",
    "gc", "getpass", "gettext", "glob", "graphlib", "gzip", "hashlib",
    "heapq", "hmac", "html", "http", "idlelib", "imaplib", "importlib",
    "inspect", "io", "ipaddress", "itertools", "json", "keyword",
    "linecache", "locale", "logging", "lzma", "mailbox", "marshal",
    "math", "mimetypes", "mmap", "multiprocessing", "netrc", "numbers",
    "operator", "os", "pathlib", "pdb", "pickle", "pickletools",
    "platform", "plistlib", "poplib", "posixpath", "pprint",
    "profile", "pstats", "py_compile", "pyclbr", "pydoc", "queue",
    "quopri", "random", "re", "readline", "reprlib", "rlcompleter",
    "runpy", "sched", "secrets", "select", "selectors", "shelve",
    "shlex", "shutil", "signal", "site", "smtplib", "sndhdr", "socket",
    "socketserver", "sqlite3", "ssl", "stat", "statistics", "string",
    "stringprep", "struct", "subprocess", "sunau", "symtable", "sys",
    "sysconfig", "syslog", "tabnanny", "tarfile", "tempfile", "test",
    "textwrap", "threading", "time", "timeit", "tkinter", "token",
    "tokenize", "tomllib", "trace", "traceback", "tracemalloc",
    "tty", "turtle", "turtledemo", "types", "typing", "unicodedata",
    "unittest", "urllib", "uuid", "venv", "warnings", "wave",
    "weakref", "webbrowser", "winreg", "winsound", "wsgiref",
    "xml", "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib",
    "_thread", "__future__",
])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImportInfo:
    """Describes a single import found in a source file."""
    module: str
    names: list[str] = field(default_factory=list)
    alias: str | None = None
    is_external: bool = False
    is_from_import: bool = False
    lineno: int = 0


@dataclass
class FFIWrapper:
    """Generated FFI wrapper for an external dependency."""
    module: str
    function_name: str
    wrapper_name: str
    param_names: list[str] = field(default_factory=list)
    return_type: str = "any"
    wrapper_code: str = ""
    recipe_block: str = ""


@dataclass
class FFIAnalysis:
    """Complete FFI analysis for a source file."""
    source_file: str
    all_imports: list[ImportInfo] = field(default_factory=list)
    external_imports: list[ImportInfo] = field(default_factory=list)
    ffi_wrappers: list[FFIWrapper] = field(default_factory=list)
    functions_needing_ffi: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Import detection
# ---------------------------------------------------------------------------

def _is_external(module_name: str) -> bool:
    """Check if a module is external (not stdlib, not local)."""
    top = module_name.split(".")[0]
    if top in _STDLIB_MODULES:
        return False
    if top.startswith("_") and top.lstrip("_") in _STDLIB_MODULES:
        return False
    return True


def extract_imports(source: str, filepath: str = "<string>") -> list[ImportInfo]:
    """Parse a Python source file and extract all import statements."""
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(ImportInfo(
                    module=alias.name,
                    alias=alias.asname,
                    is_external=_is_external(alias.name),
                    lineno=node.lineno,
                ))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names = [a.name for a in (node.names or [])]
                imports.append(ImportInfo(
                    module=node.module,
                    names=names,
                    is_external=_is_external(node.module),
                    is_from_import=True,
                    lineno=node.lineno,
                ))
    return imports


# ---------------------------------------------------------------------------
# Function-level dependency mapping
# ---------------------------------------------------------------------------

class _ExternalCallFinder(ast.NodeVisitor):
    """Find calls to external modules within a function body."""

    def __init__(self, external_modules: set[str], import_aliases: dict[str, str]):
        self.external_modules = external_modules
        self.import_aliases = import_aliases
        self.external_calls: list[dict] = []

    def visit_Call(self, node):
        name = self._resolve_call(node)
        if name:
            top = name.split(".")[0]
            resolved = self.import_aliases.get(top, top)
            if resolved in self.external_modules or top in self.external_modules:
                self.external_calls.append({
                    "call": name,
                    "module": resolved,
                    "lineno": node.lineno,
                })
        self.generic_visit(node)

    def _resolve_call(self, node) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            parts = []
            n = node.func
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if isinstance(n, ast.Name):
                parts.append(n.id)
            return ".".join(reversed(parts))
        return ""


def map_function_externals(source: str, filepath: str,
                           imports: list[ImportInfo]) -> dict[str, list[dict]]:
    """Map each function to its external library calls.

    Returns ``{func_name: [{"call": ..., "module": ..., "lineno": ...}]}``.
    """
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return {}

    external_modules = {i.module.split(".")[0] for i in imports if i.is_external}
    aliases = {}
    for i in imports:
        if i.alias:
            aliases[i.alias] = i.module.split(".")[0]
        if i.is_from_import:
            for name in i.names:
                aliases[name] = i.module.split(".")[0]

    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        finder = _ExternalCallFinder(external_modules, aliases)
        finder.visit(node)
        if finder.external_calls:
            result[node.name] = finder.external_calls

    return result


# ---------------------------------------------------------------------------
# FFI wrapper generation
# ---------------------------------------------------------------------------

def generate_ffi_wrapper(module: str, call_name: str,
                         func_node: ast.FunctionDef | None = None) -> FFIWrapper:
    """Generate a lightweight FFI wrapper for an external library call."""
    safe_name = call_name.replace(".", "_")
    wrapper_name = f"ffi_{safe_name}"

    param_names = []
    if func_node:
        for arg in func_node.args.args:
            param_names.append(arg.arg)
    if not param_names:
        param_names = ["*args", "**kwargs"]

    params_str = ", ".join(param_names)
    passthrough_str = ", ".join(p for p in param_names if not p.startswith("*"))
    if not passthrough_str:
        passthrough_str = "*args, **kwargs"

    wrapper_code = (
        f"def {wrapper_name}({params_str}):\n"
        f"    \"\"\"FFI wrapper: delegates to {call_name} via {module}\"\"\"\n"
        f"    import {module.split('.')[0]}\n"
        f"    _result = {call_name}({passthrough_str})\n"
        f"    return _result\n"
    )

    recipe_block = (
        f"[task:{wrapper_name}]\n"
        f"op = call\n"
        f'fn = write_file\n'
        f'args = "build_sandbox/mesh_outputs/ffi_{safe_name}.dat", "ffi_passthrough"\n'
        f"family = ffi\n"
    )

    return FFIWrapper(
        module=module,
        function_name=call_name,
        wrapper_name=wrapper_name,
        param_names=param_names,
        wrapper_code=wrapper_code,
        recipe_block=recipe_block,
    )


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_ffi(source_path: str) -> FFIAnalysis:
    """Analyze a source file for external dependencies and generate FFI wrappers."""
    abs_path = os.path.join(_ROOT, source_path) if not os.path.isabs(source_path) else source_path

    with open(abs_path, "r", encoding="utf-8") as f:
        source = f.read()

    imports = extract_imports(source, abs_path)
    external = [i for i in imports if i.is_external]
    func_externals = map_function_externals(source, abs_path, imports)

    analysis = FFIAnalysis(
        source_file=source_path,
        all_imports=imports,
        external_imports=external,
        functions_needing_ffi=func_externals,
    )

    seen_calls = set()
    for func_name, calls in func_externals.items():
        for call_info in calls:
            call_key = call_info["call"]
            if call_key in seen_calls:
                continue
            seen_calls.add(call_key)

            wrapper = generate_ffi_wrapper(
                module=call_info["module"],
                call_name=call_key,
            )
            analysis.ffi_wrappers.append(wrapper)

    return analysis


def write_ffi_module(analysis: FFIAnalysis,
                     output_dir: str = "build_sandbox/ffi_wrappers") -> str | None:
    """Write all generated FFI wrappers to a single Python module.

    Returns the output file path, or None if no wrappers were generated.
    """
    if not analysis.ffi_wrappers:
        return None

    abs_dir = os.path.join(_ROOT, output_dir) if not os.path.isabs(output_dir) else output_dir
    os.makedirs(abs_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(analysis.source_file))[0]
    out_path = os.path.join(abs_dir, f"ffi_{base}.py")

    lines = [
        f'"""Auto-generated FFI wrappers for {analysis.source_file}',
        f'',
        f'External dependencies: {", ".join(i.module for i in analysis.external_imports)}',
        f'"""',
        f'',
    ]

    for wrapper in analysis.ffi_wrappers:
        lines.append(wrapper.wrapper_code)
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path
