"""Static analyzer and hot-path profiler for real codebases.

Traverses a target directory, performs call-graph and complexity analysis
on Python and JavaScript source files, classifies code segments into
hot-paths (candidates for Aero translation) and cold-paths (boilerplate),
and exports a JSON performance-density manifest.
"""

import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".py", ".js", ".rs"}

COLD_PATH_INDICATORS = {
    "config_patterns": re.compile(
        r"(config|settings|\.env|constants|setup\.py|setup\.cfg"
        r"|package\.json|tsconfig|\.eslintrc|webpack"
        r"|Cargo\.toml|Cargo\.lock|build\.rs)", re.IGNORECASE,
    ),
    "ui_patterns": re.compile(
        r"(template|component|\.html|\.css|\.scss|render\s*\()", re.IGNORECASE,
    ),
}

JS_FUNC_DECL = re.compile(
    r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>|\w+\s*=>))",
)
JS_FUNC_CALL = re.compile(r"\b(\w+)\s*\(")
JS_LOOP = re.compile(r"\b(for|while|do)\b")
JS_ARRAY_STRING_OPS = re.compile(
    r"\.\s*(map|filter|reduce|forEach|find|some|every|flatMap|sort"
    r"|push|pop|shift|unshift|splice|concat|slice"
    r"|split|join|replace|match|search|substring|indexOf|includes)\s*\(",
)


@dataclass
class FunctionProfile:
    name: str
    lineno: int
    end_lineno: int
    max_loop_depth: int = 0
    is_recursive: bool = False
    calls: list = field(default_factory=list)
    array_string_ops: int = 0
    complexity_score: float = 0.0


@dataclass
class FileProfile:
    path: str
    relative_path: str
    language: str
    size: int
    fingerprint: str
    functions: list = field(default_factory=list)
    max_nesting_depth: int = 0
    total_array_string_ops: int = 0
    recursive_functions: list = field(default_factory=list)
    performance_density: float = 0.0
    classification: str = "cold"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def file_fingerprint(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def is_cold_by_name(path: str) -> bool:
    for pattern in COLD_PATH_INDICATORS.values():
        if pattern.search(path):
            return True
    return False


# ---------------------------------------------------------------------------
# Python analyzer (AST-based)
# ---------------------------------------------------------------------------

class _LoopDepthVisitor(ast.NodeVisitor):
    """Walk an AST subtree and return the maximum loop nesting depth."""

    def __init__(self):
        self.max_depth = 0
        self._current = 0

    def _enter_loop(self, node):
        self._current += 1
        if self._current > self.max_depth:
            self.max_depth = self._current
        self.generic_visit(node)
        self._current -= 1

    visit_For = _enter_loop
    visit_While = _enter_loop

    # list/set/dict/generator comprehensions contain implicit loops
    visit_ListComp = _enter_loop
    visit_SetComp = _enter_loop
    visit_DictComp = _enter_loop
    visit_GeneratorExp = _enter_loop


class _CallCollector(ast.NodeVisitor):
    """Collect all function call names inside an AST subtree."""

    def __init__(self):
        self.calls: list[str] = []
        self.array_string_ops = 0

    _HOTOPS = frozenset([
        "append", "extend", "insert", "pop", "remove", "sort", "reverse",
        "split", "join", "replace", "find", "index", "count", "strip",
        "startswith", "endswith", "encode", "decode", "format",
        "map", "filter", "reduce", "sorted", "zip", "enumerate",
    ])

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
            if node.func.id in self._HOTOPS:
                self.array_string_ops += 1
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
            if node.func.attr in self._HOTOPS:
                self.array_string_ops += 1
        self.generic_visit(node)


def analyze_python(source: str, filepath: str) -> list[FunctionProfile]:
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    profiles = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        ld = _LoopDepthVisitor()
        ld.visit(node)

        cc = _CallCollector()
        cc.visit(node)

        is_recursive = node.name in cc.calls

        end_lineno = getattr(node, "end_lineno", node.lineno)
        span = max(end_lineno - node.lineno + 1, 1)

        score = (
            ld.max_depth * 3.0
            + cc.array_string_ops * 1.5
            + (5.0 if is_recursive else 0.0)
            + len(cc.calls) * 0.2
            + span * 0.05
        )

        profiles.append(FunctionProfile(
            name=node.name,
            lineno=node.lineno,
            end_lineno=end_lineno,
            max_loop_depth=ld.max_depth,
            is_recursive=is_recursive,
            calls=cc.calls,
            array_string_ops=cc.array_string_ops,
            complexity_score=round(score, 2),
        ))

    return profiles


# ---------------------------------------------------------------------------
# JavaScript analyzer (regex-based heuristic)
# ---------------------------------------------------------------------------

def _js_loop_depth(source: str) -> int:
    """Estimate maximum loop nesting depth via brace tracking."""
    max_depth = 0
    depth = 0
    in_loop = []
    i = 0
    lines = source.split("\n")
    for line in lines:
        stripped = line.strip()
        if JS_LOOP.search(stripped):
            in_loop.append(True)
        for ch in stripped:
            if ch == "{":
                depth += 1
                if in_loop:
                    if depth > max_depth:
                        max_depth = depth
            elif ch == "}":
                depth -= 1
                if in_loop and depth < len(in_loop):
                    in_loop.pop()
    return max(max_depth - 1, 0)


def analyze_javascript(source: str, filepath: str) -> list[FunctionProfile]:
    profiles = []
    func_names = set()
    for m in JS_FUNC_DECL.finditer(source):
        name = m.group(1) or m.group(2)
        if name:
            func_names.add(name)

    # Filter out function declarations from call matches
    _JS_DECL_LINE = re.compile(r"(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?function)\s*\(")
    raw_calls = []
    for line in source.split("\n"):
        stripped = line.strip()
        if _JS_DECL_LINE.search(stripped):
            continue
        raw_calls.extend(JS_FUNC_CALL.findall(stripped))
    all_calls = [c for c in raw_calls if c not in ("function", "if", "for", "while", "switch", "catch")]
    arr_ops = len(JS_ARRAY_STRING_OPS.findall(source))
    loop_depth = _js_loop_depth(source)

    for name in func_names:
        calls_from = [c for c in all_calls if c != name]
        is_recursive = name in all_calls
        call_count = all_calls.count(name)

        score = (
            loop_depth * 3.0
            + arr_ops * 1.5
            + (5.0 if is_recursive else 0.0)
            + call_count * 0.2
        )

        profiles.append(FunctionProfile(
            name=name,
            lineno=0,
            end_lineno=0,
            max_loop_depth=loop_depth,
            is_recursive=is_recursive,
            calls=calls_from[:20],
            array_string_ops=arr_ops,
            complexity_score=round(score, 2),
        ))

    if not func_names and (loop_depth > 0 or arr_ops > 0):
        score = loop_depth * 3.0 + arr_ops * 1.5
        profiles.append(FunctionProfile(
            name="<module>",
            lineno=0,
            end_lineno=0,
            max_loop_depth=loop_depth,
            calls=all_calls[:20],
            array_string_ops=arr_ops,
            complexity_score=round(score, 2),
        ))

    return profiles


# ---------------------------------------------------------------------------
# Rust analyzer (regex-based tokenizer)
# ---------------------------------------------------------------------------

# Matches: fn name(args) -> RetType { ... }, pub fn, pub(crate) fn, async fn, etc.
RS_FUNC_DECL = re.compile(
    r"(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)"
)
RS_STRUCT_DECL = re.compile(r"(?:pub(?:\s*\([^)]*\))?\s+)?struct\s+(\w+)")
RS_IMPL_BLOCK = re.compile(r"impl(?:<[^>]*>)?\s+(\w+)")
RS_LOOP = re.compile(r"\b(for|while|loop)\b")
RS_FUNC_CALL = re.compile(r"\b(\w+)\s*\(")
RS_ARRAY_OPS = re.compile(
    r"\.(iter|into_iter|map|filter|fold|reduce|collect|for_each|flatten"
    r"|flat_map|zip|enumerate|sum|product|any|all|find|position"
    r"|sort|sort_by|sort_unstable|push|pop|extend|retain|drain"
    r"|split|join|contains|starts_with|ends_with|replace|trim)\s*\(",
)

# Rust memory / lifetime / unsafe patterns for cold-path detection
RS_LIFETIME_PATTERNS = re.compile(
    r"(unsafe\s+\{|unsafe\s+fn|unsafe\s+impl"
    r"|\*mut\s|\*const\s"
    r"|ManuallyDrop|MaybeUninit|NonNull|Pin<"
    r"|std::alloc::|alloc::|dealloc|Layout::from_size_align"
    r"|std::mem::transmute|std::mem::forget|mem::transmute|mem::forget"
    r"|std::ptr::|ptr::null|ptr::write|ptr::read"
    r"|Box::from_raw|Box::into_raw"
    r"|Arc::from_raw|Rc::from_raw)",
)

RS_THREADING_PATTERNS = re.compile(
    r"(std::thread|thread::spawn|thread::JoinHandle"
    r"|std::sync::|Mutex|RwLock|Arc<|Condvar|Barrier"
    r"|tokio::|async_std::|futures::"
    r"|rayon::|crossbeam::)",
)

# Hot-path math/tensor patterns in Rust
RS_MATH_TENSOR_PATTERNS = re.compile(
    r"(\bfor\b.*\bin\b.*\.\.|\bwhile\b"
    r"|\[.*\].*\[.*\]"
    r"|ndarray|nalgebra|tensor|matrix|eigen|lapack"
    r"|f32|f64|usize|isize"
    r"|\+=|\-=|\*=|/="
    r"|\.sin\(|\.cos\(|\.exp\(|\.sqrt\(|\.abs\(|\.powi\(|\.powf\()",
)


def _rs_extract_functions(source: str) -> list[dict]:
    """Extract function boundaries from Rust source using brace matching."""
    functions = []
    for m in RS_FUNC_DECL.finditer(source):
        name = m.group(1)
        start_pos = m.start()
        start_line = source[:start_pos].count("\n") + 1

        # Find opening brace
        brace_pos = source.find("{", m.end())
        if brace_pos == -1:
            continue

        # Match braces to find function end
        depth = 1
        pos = brace_pos + 1
        while pos < len(source) and depth > 0:
            ch = source[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1

        end_line = source[:pos].count("\n") + 1
        body = source[brace_pos:pos]

        functions.append({
            "name": name,
            "start_line": start_line,
            "end_line": end_line,
            "body": body,
        })

    return functions


def _rs_loop_depth(body: str) -> int:
    """Estimate maximum loop nesting depth in a Rust function body."""
    max_depth = 0
    depth = 0
    brace_depth = 0
    loop_braces = []

    lines = body.split("\n")
    for line in lines:
        stripped = line.strip()
        if RS_LOOP.search(stripped):
            # Find the next { after the loop keyword
            loop_braces.append(brace_depth + stripped.count("{"))
            depth += 1
            if depth > max_depth:
                max_depth = depth

        for ch in stripped:
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if loop_braces and brace_depth < loop_braces[-1]:
                    loop_braces.pop()
                    depth = max(depth - 1, 0)

    return max_depth


def _rs_is_memory_cold(body: str) -> bool:
    """Check if a Rust function contains memory/lifetime patterns."""
    return bool(RS_LIFETIME_PATTERNS.search(body))


def _rs_is_threading_cold(body: str) -> bool:
    """Check if a Rust function contains threading/async patterns."""
    return bool(RS_THREADING_PATTERNS.search(body))


def _rs_is_math_hot(body: str) -> bool:
    """Check if a Rust function contains math/tensor hot-path patterns."""
    return bool(RS_MATH_TENSOR_PATTERNS.search(body))


def analyze_rust(source: str, filepath: str) -> list[FunctionProfile]:
    """Analyze a Rust source file for function calls, loops, and complexity."""
    functions = _rs_extract_functions(source)
    structs = RS_STRUCT_DECL.findall(source)

    profiles = []
    for func in functions:
        name = func["name"]
        body = func["body"]

        loop_depth = _rs_loop_depth(body)
        calls = RS_FUNC_CALL.findall(body)
        calls = [c for c in calls if c not in ("for", "while", "loop", "if",
                                                "match", "fn", "let", "mut",
                                                "return", "struct", "impl",
                                                "pub", "use", "mod")]
        is_recursive = name in calls
        arr_ops = len(RS_ARRAY_OPS.findall(body))

        is_memory_cold = _rs_is_memory_cold(body)
        is_threading_cold = _rs_is_threading_cold(body)
        is_math_hot = _rs_is_math_hot(body)

        span = func["end_line"] - func["start_line"] + 1
        score = (
            loop_depth * 3.0
            + arr_ops * 1.5
            + (5.0 if is_recursive else 0.0)
            + len(calls) * 0.2
            + span * 0.05
            + (3.0 if is_math_hot else 0.0)
            - (10.0 if is_memory_cold else 0.0)
            - (10.0 if is_threading_cold else 0.0)
        )

        profiles.append(FunctionProfile(
            name=name,
            lineno=func["start_line"],
            end_lineno=func["end_line"],
            max_loop_depth=loop_depth,
            is_recursive=is_recursive,
            calls=calls[:20],
            array_string_ops=arr_ops,
            complexity_score=round(max(score, 0.0), 2),
        ))

    return profiles


# ---------------------------------------------------------------------------
# Directory traversal & profiling
# ---------------------------------------------------------------------------

def traverse_directory(target_dir: str) -> list[dict]:
    """Walk *target_dir* and return a flat list of file-tree entries."""
    tree = []
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in sorted(files):
            full = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            tree.append({
                "path": full,
                "relative": os.path.relpath(full, target_dir),
                "extension": ext,
                "size": os.path.getsize(full),
            })
    return tree


def profile_file(entry: dict, target_dir: str) -> FileProfile | None:
    """Run static analysis on a single source file."""
    ext = entry["extension"]
    if ext not in SUPPORTED_EXTENSIONS:
        return None

    path = entry["path"]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError:
        return None

    lang_map = {".py": "python", ".js": "javascript", ".rs": "rust"}
    lang = lang_map.get(ext, "unknown")
    if lang == "python":
        functions = analyze_python(source, path)
    elif lang == "rust":
        functions = analyze_rust(source, path)
    else:
        functions = analyze_javascript(source, path)

    max_nesting = max((fn.max_loop_depth for fn in functions), default=0)
    total_ops = sum(fn.array_string_ops for fn in functions)
    recursive = [fn.name for fn in functions if fn.is_recursive]
    density = sum(fn.complexity_score for fn in functions)

    fp = FileProfile(
        path=path,
        relative_path=entry["relative"],
        language=lang,
        size=entry["size"],
        fingerprint=file_fingerprint(path),
        functions=functions,
        max_nesting_depth=max_nesting,
        total_array_string_ops=total_ops,
        recursive_functions=recursive,
        performance_density=round(density, 2),
    )
    return fp


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_HOT_THRESHOLD = 5.0


def classify_profiles(profiles: list[FileProfile]) -> list[FileProfile]:
    """Tag each profile as 'hot' or 'cold'."""
    for fp in profiles:
        if is_cold_by_name(fp.relative_path):
            fp.classification = "cold"
            continue
        if fp.performance_density >= _HOT_THRESHOLD:
            fp.classification = "hot"
        else:
            fp.classification = "cold"
    return profiles


# ---------------------------------------------------------------------------
# Manifest export
# ---------------------------------------------------------------------------

def export_manifest(profiles: list[FileProfile], output_path: str) -> str:
    """Write the profiling results as a JSON manifest and return the path."""
    manifest = {
        "version": "1.0",
        "total_files": len(profiles),
        "hot_paths": sum(1 for p in profiles if p.classification == "hot"),
        "cold_paths": sum(1 for p in profiles if p.classification == "cold"),
        "files": [],
    }

    for fp in profiles:
        entry = {
            "relative_path": fp.relative_path,
            "language": fp.language,
            "size_bytes": fp.size,
            "fingerprint": fp.fingerprint,
            "classification": fp.classification,
            "performance_density": fp.performance_density,
            "max_nesting_depth": fp.max_nesting_depth,
            "total_array_string_ops": fp.total_array_string_ops,
            "recursive_functions": fp.recursive_functions,
            "functions": [
                {
                    "name": fn.name,
                    "lineno": fn.lineno,
                    "end_lineno": fn.end_lineno,
                    "max_loop_depth": fn.max_loop_depth,
                    "is_recursive": fn.is_recursive,
                    "array_string_ops": fn.array_string_ops,
                    "complexity_score": fn.complexity_score,
                }
                for fn in fp.functions
            ],
        }
        manifest["files"].append(entry)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_profiler(target_dir: str,
                 output_path: str = "build_sandbox/mesh_outputs/profiler_manifest.json") -> dict:
    """Full profiler pipeline: traverse -> analyze -> classify -> export."""
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)

    abs_target = os.path.join(_root, target_dir) if not os.path.isabs(target_dir) else target_dir
    abs_output = os.path.join(_root, output_path) if not os.path.isabs(output_path) else output_path

    print(f"[profiler] Scanning: {abs_target}", flush=True)
    tree = traverse_directory(abs_target)
    print(f"[profiler] Found {len(tree)} files in tree", flush=True)

    profiles = []
    for entry in tree:
        fp = profile_file(entry, abs_target)
        if fp is not None:
            profiles.append(fp)

    print(f"[profiler] Analyzed {len(profiles)} source files", flush=True)

    classify_profiles(profiles)
    hot = [p for p in profiles if p.classification == "hot"]
    cold = [p for p in profiles if p.classification == "cold"]
    print(f"[profiler] Classification: {len(hot)} hot, {len(cold)} cold", flush=True)

    manifest_path = export_manifest(profiles, abs_output)
    print(f"[profiler] Manifest exported: {manifest_path}", flush=True)

    return {
        "total_files_scanned": len(tree),
        "source_files_analyzed": len(profiles),
        "hot_paths": len(hot),
        "cold_paths": len(cold),
        "manifest_path": manifest_path,
        "hot_files": [p.relative_path for p in hot],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Aero AutoDev Static Analyzer & Hot-Path Profiler",
    )
    parser.add_argument("target", help="Target directory to profile")
    parser.add_argument(
        "--output", "-o",
        default="build_sandbox/mesh_outputs/profiler_manifest.json",
        help="Output path for the JSON manifest",
    )
    args = parser.parse_args()

    results = run_profiler(args.target, args.output)

    if results["hot_paths"]:
        print(f"\n[profiler] Hot-path files ({results['hot_paths']}):")
        for hp in results["hot_files"]:
            print(f"  -> {hp}")
    else:
        print("\n[profiler] No hot-paths detected in target.")


if __name__ == "__main__":
    main()
