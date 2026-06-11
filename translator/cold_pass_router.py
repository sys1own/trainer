"""Cold-path pass-through router.

Detects untranslatable code patterns — multi-threading, hardware drivers,
inline assembly, ctypes/cffi interop, GPU kernels — and marks entire
function blocks as "Cold Path Pass-Through" to guarantee 100% behavioral
equivalence by leaving them untouched.
"""

import ast
import os
import re
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_THREADING_MODULES = frozenset([
    "threading", "multiprocessing", "concurrent", "concurrent.futures",
    "asyncio", "_thread", "queue", "subprocess",
])

_HARDWARE_MODULES = frozenset([
    "ctypes", "cffi", "sysconfig", "mmap", "fcntl", "termios",
    "resource", "signal", "pty", "tty",
])

_GPU_MODULES = frozenset([
    "cuda", "pycuda", "numba", "cupy", "torch", "tensorflow",
    "jax", "opencl", "pyopencl",
])

_UNSAFE_PATTERNS = re.compile(
    r"(__asm__|__attribute__|volatile|register\s|"
    r"ctypes\.CDLL|ctypes\.windll|ctypes\.cdll|"
    r"cffi\.FFI|ffi\.cdef|ffi\.dlopen|"
    r"mmap\.mmap|"
    r"POINTER\(|byref\(|addressof\(|cast\(|"
    r"\.cuda\(\)|\.to\(['\"]cuda|cuda\.synchronize|"
    r"numba\.jit|numba\.cuda|@jit|@cuda\.jit)", re.IGNORECASE,
)

# Rust-specific patterns for cold-path detection
_RS_MEMORY_PATTERNS = re.compile(
    r"(unsafe\s*\{|unsafe\s+fn|unsafe\s+impl"
    r"|\*mut\s|\*const\s"
    r"|ManuallyDrop|MaybeUninit|NonNull"
    r"|std::alloc::|alloc::|dealloc|Layout::from_size_align"
    r"|std::mem::transmute|std::mem::forget|mem::transmute|mem::forget"
    r"|std::ptr::|ptr::null|ptr::write|ptr::read"
    r"|Box::from_raw|Box::into_raw"
    r"|Arc::from_raw|Rc::from_raw"
    r"|Pin<|Unpin)",
)

_RS_THREADING_PATTERNS = re.compile(
    r"(std::thread|thread::spawn|thread::JoinHandle"
    r"|std::sync::|Mutex<|RwLock<|Arc<.*Mutex|Condvar|Barrier"
    r"|tokio::|async_std::|futures::"
    r"|rayon::|crossbeam::"
    r"|async\s+fn|async\s+move|\.await)",
)

_RS_LIFETIME_PATTERNS = re.compile(
    r"(<'[a-z]|&'[a-z]|where\s.*'[a-z]\s*:)",
)


@dataclass
class ColdPathReason:
    """Describes why a function was marked as a cold pass-through."""
    category: str      # "threading", "hardware", "gpu", "unsafe_pattern"
    detail: str
    lineno: int = 0


@dataclass
class FunctionRouting:
    """Routing decision for a single function."""
    name: str
    lineno: int
    is_cold_passthrough: bool = False
    reasons: list[ColdPathReason] = field(default_factory=list)
    external_deps: list[str] = field(default_factory=list)


@dataclass
class RoutingAnalysis:
    """Complete routing analysis for a source file."""
    source_file: str
    functions: list[FunctionRouting] = field(default_factory=list)
    cold_count: int = 0
    hot_count: int = 0


# ---------------------------------------------------------------------------
# AST-based detection
# ---------------------------------------------------------------------------

class _UntranslatableDetector(ast.NodeVisitor):
    """Walk a function AST and detect untranslatable patterns."""

    def __init__(self, external_modules: set[str]):
        self.external_modules = external_modules
        self.reasons: list[ColdPathReason] = []
        self.external_deps: list[str] = []

    def visit_Import(self, node):
        for alias in node.names:
            top = alias.name.split(".")[0]
            self._check_module(top, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            top = node.module.split(".")[0]
            self._check_module(top, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node):
        name = self._call_name(node)
        top = name.split(".")[0]

        # Thread/process creation
        if any(t in name for t in ("Thread", "Process", "Pool", "Executor")):
            self.reasons.append(ColdPathReason(
                category="threading",
                detail=f"Thread/process creation: {name}",
                lineno=node.lineno,
            ))

        # Check against external module calls
        if top in self.external_modules:
            self.external_deps.append(name)
            self.reasons.append(ColdPathReason(
                category="external_dep",
                detail=f"External library call: {name}",
                lineno=node.lineno,
            ))

        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                name = self._call_name(item.context_expr)
                if any(t in name for t in ("ThreadPoolExecutor", "ProcessPoolExecutor",
                                           "Pool", "Lock", "Semaphore", "Barrier")):
                    self.reasons.append(ColdPathReason(
                        category="threading",
                        detail=f"Context-managed concurrency: {name}",
                        lineno=node.lineno,
                    ))
        self.generic_visit(node)

    def visit_Await(self, node):
        self.reasons.append(ColdPathReason(
            category="threading",
            detail="Async await expression",
            lineno=node.lineno,
        ))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.reasons.append(ColdPathReason(
            category="threading",
            detail=f"Async function definition: {node.name}",
            lineno=node.lineno,
        ))
        self.generic_visit(node)

    def _check_module(self, module: str, lineno: int):
        if module in _THREADING_MODULES:
            self.reasons.append(ColdPathReason(
                category="threading",
                detail=f"Threading/concurrency module: {module}",
                lineno=lineno,
            ))
        if module in _HARDWARE_MODULES:
            self.reasons.append(ColdPathReason(
                category="hardware",
                detail=f"Hardware/driver module: {module}",
                lineno=lineno,
            ))
        if module in _GPU_MODULES:
            self.reasons.append(ColdPathReason(
                category="gpu",
                detail=f"GPU/accelerator module: {module}",
                lineno=lineno,
            ))

    def _call_name(self, node) -> str:
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


def _check_source_patterns(source: str, func_start: int,
                           func_end: int) -> list[ColdPathReason]:
    """Regex-based scan for unsafe patterns in function source text."""
    lines = source.split("\n")
    func_text = "\n".join(lines[func_start - 1:func_end])
    reasons = []

    for m in _UNSAFE_PATTERNS.finditer(func_text):
        reasons.append(ColdPathReason(
            category="unsafe_pattern",
            detail=f"Unsafe pattern: {m.group(0)[:50]}",
            lineno=func_start + func_text[:m.start()].count("\n"),
        ))

    return reasons


# ---------------------------------------------------------------------------
# Routing analysis
# ---------------------------------------------------------------------------

def analyze_routing(source_path: str,
                    external_modules: set[str] | None = None) -> RoutingAnalysis:
    """Analyze a source file and determine routing for each function.

    Functions with threading, hardware, GPU, or unsafe patterns are marked
    as cold pass-through. All others remain candidates for translation.
    """
    abs_path = os.path.join(_ROOT, source_path) if not os.path.isabs(source_path) else source_path

    with open(abs_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=abs_path)
    except SyntaxError:
        return RoutingAnalysis(source_file=source_path)

    if external_modules is None:
        external_modules = set()

    # Collect file-level imports into external_modules for call detection
    from translator.ffi_generator import _is_external
    import_aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in (_THREADING_MODULES | _HARDWARE_MODULES | _GPU_MODULES):
                    external_modules.add(top)
                if _is_external(alias.name):
                    external_modules.add(top)
                if alias.asname:
                    import_aliases[alias.asname] = top
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in (_THREADING_MODULES | _HARDWARE_MODULES | _GPU_MODULES):
                external_modules.add(top)
            if _is_external(node.module):
                external_modules.add(top)
            for alias in (node.names or []):
                import_aliases[alias.name] = top
                if alias.asname:
                    import_aliases[alias.asname] = top

    # Add aliases to external_modules so call resolution works
    for alias, mod in import_aliases.items():
        if mod in external_modules:
            external_modules.add(alias)

    analysis = RoutingAnalysis(source_file=source_path)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        routing = FunctionRouting(
            name=node.name,
            lineno=node.lineno,
        )

        # AST-based detection
        detector = _UntranslatableDetector(external_modules)
        detector.visit(node)
        routing.reasons.extend(detector.reasons)
        routing.external_deps.extend(detector.external_deps)

        # Regex-based detection
        end_lineno = getattr(node, "end_lineno", node.lineno + 10)
        pattern_reasons = _check_source_patterns(source, node.lineno, end_lineno)
        routing.reasons.extend(pattern_reasons)

        if routing.reasons:
            routing.is_cold_passthrough = True
            analysis.cold_count += 1
        else:
            analysis.hot_count += 1

        analysis.functions.append(routing)

    return analysis


# ---------------------------------------------------------------------------
# Rust routing analysis
# ---------------------------------------------------------------------------

def analyze_routing_rust(source_path: str) -> RoutingAnalysis:
    """Analyze a Rust source file and determine routing for each function.

    Marks functions with unsafe blocks, raw pointers, lifetime constraints,
    threading primitives, or explicit memory management as cold pass-through.
    Pure loop-heavy math/tensor functions remain hot.
    """
    from translator.code_profiler import _rs_extract_functions, RS_FUNC_DECL

    abs_path = os.path.join(_ROOT, source_path) if not os.path.isabs(source_path) else source_path

    with open(abs_path, "r", encoding="utf-8") as f:
        source = f.read()

    analysis = RoutingAnalysis(source_file=source_path)
    functions = _rs_extract_functions(source)

    for func in functions:
        name = func["name"]
        body = func["body"]
        start_line = func["start_line"]

        routing = FunctionRouting(name=name, lineno=start_line)

        # Memory / unsafe detection
        for m in _RS_MEMORY_PATTERNS.finditer(body):
            routing.reasons.append(ColdPathReason(
                category="rust_memory",
                detail=f"Memory/unsafe pattern: {m.group(0)[:50]}",
                lineno=start_line + body[:m.start()].count("\n"),
            ))

        # Threading detection
        for m in _RS_THREADING_PATTERNS.finditer(body):
            routing.reasons.append(ColdPathReason(
                category="rust_threading",
                detail=f"Threading/async pattern: {m.group(0)[:50]}",
                lineno=start_line + body[:m.start()].count("\n"),
            ))

        # Lifetime constraint detection
        # Check the full function signature (source around start_line)
        sig_start = max(0, start_line - 1)
        sig_end = min(len(source.split("\n")), start_line + 3)
        sig_text = "\n".join(source.split("\n")[sig_start:sig_end])
        for m in _RS_LIFETIME_PATTERNS.finditer(sig_text):
            routing.reasons.append(ColdPathReason(
                category="rust_lifetime",
                detail=f"Lifetime constraint: {m.group(0)[:50]}",
                lineno=start_line,
            ))

        if routing.reasons:
            routing.is_cold_passthrough = True
            analysis.cold_count += 1
        else:
            analysis.hot_count += 1

        analysis.functions.append(routing)

    return analysis


# ---------------------------------------------------------------------------
# Unified routing dispatcher
# ---------------------------------------------------------------------------

def analyze_routing_dispatch(source_path: str,
                             external_modules: set[str] | None = None) -> RoutingAnalysis:
    """Route to the correct analyzer based on file extension."""
    if source_path.endswith(".rs"):
        return analyze_routing_rust(source_path)
    return analyze_routing(source_path, external_modules)


# ---------------------------------------------------------------------------
# Integration: filter function list for translation
# ---------------------------------------------------------------------------

def filter_translatable(source_path: str,
                        function_names: list[str],
                        external_modules: set[str] | None = None) -> tuple[list[str], list[FunctionRouting]]:
    """Split function names into translatable and cold pass-through lists.

    Returns ``(translatable_names, cold_routings)``.
    """
    routing = analyze_routing_dispatch(source_path, external_modules)
    cold_map = {f.name: f for f in routing.functions if f.is_cold_passthrough}

    translatable = []
    cold = []
    for name in function_names:
        if name in cold_map:
            cold.append(cold_map[name])
        else:
            translatable.append(name)

    return translatable, cold
