"""Tree-Sitter backed Rust AST utilities for the source-modification layer.

This module replaces the legacy regex / brace-matching / line-slice /
``str.replace`` style of source mutation with formal syntax-tree node
isolation. Every transform is expressed as a *byte-exact splice* against a
``function_item`` node's index boundaries, which guarantees zero syntax bleed
into neighbouring structs, macros, or impl blocks.

The grammar is loaded from the ``tree-sitter-rust`` package. Parsing is a hard
dependency of the modification layer by design: we never fall back to text
matching for code edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import tree_sitter_rust as _tsr
    from tree_sitter import Language, Parser
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "translator.rust_ast requires 'tree-sitter' and 'tree-sitter-rust'. "
        "Install them with: pip install tree-sitter tree-sitter-rust"
    ) from exc


_RUST_LANGUAGE = Language(_tsr.language())


def _parser() -> Parser:
    return Parser(_RUST_LANGUAGE)


def parse(source: str):
    """Parse Rust *source* into a Tree-Sitter syntax tree."""
    return _parser().parse(source.encode("utf-8"))


# ---------------------------------------------------------------------------
# Identifier sanitisation (no str.replace on source text)
# ---------------------------------------------------------------------------

def safe_ident(name: str) -> str:
    """Sanitise a symbol name into a valid Rust identifier.

    Operates character-by-character on a *symbol name* (never on source
    text), so it carries none of the bleed risk of ``str.replace`` masks.
    """
    out = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if out and out[0].isdigit():
        out = "_" + out
    return out or "_anon"


# ---------------------------------------------------------------------------
# FFI type mapping (driven by AST type nodes, not regex)
# ---------------------------------------------------------------------------

def map_ffi_type(rust_type: str) -> tuple[str, str]:
    """Map a Rust parameter type to an ``extern "C"`` type + a marshal kind.

    Returns ``(extern_type, kind)``. ``kind`` drives how the caller marshals
    the argument across the FFI boundary (pointer + length, scalar, etc.).
    """
    compact = " ".join(rust_type.split())
    if compact in ("&[f64]", "&mut [f64]", "&mut[f64]", "&[ f64 ]"):
        return ("*const f64", "slice_f64")
    if compact.endswith("[f64]") and compact.startswith("&"):
        return ("*const f64", "slice_f64")
    if "Vec<f64>" in compact and ("Vec<Vec" in compact or compact.startswith("&[Vec")):
        return ("*const f64", "nested_f64")
    if compact == "f64":
        return ("f64", "scalar_f64")
    if compact in ("usize", "isize", "u32", "i32", "u64", "i64"):
        return (compact, "scalar_int")
    # Default: treat as an opaque caller-owned buffer.
    return ("*const f64", "slice_f64")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RustParam:
    name: str
    rust_type: str
    ffi_type: str
    kind: str


@dataclass
class RustFn:
    """A top-level Rust function isolated by its AST node boundaries."""
    name: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    signature: str                       # verbatim 'pub fn name(...) -> Ret '
    params: list[RustParam] = field(default_factory=list)
    return_type: str = "()"
    is_unsafe: bool = False
    body_text: str = ""

    def node_text(self, source: str) -> str:
        """Return the exact source bytes of this function node."""
        return source.encode("utf-8")[self.start_byte:self.end_byte].decode("utf-8")


def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8")


def extract_functions(source: str) -> list[RustFn]:
    """Extract every top-level ``function_item`` with byte-exact boundaries.

    Doc comments and attributes that precede a function are *separate* sibling
    nodes, so the returned ``start_byte``/``end_byte`` cover the function only —
    exactly the index coordinates needed for clamped deactivation.
    """
    src = source.encode("utf-8")
    tree = parse(source)
    fns: list[RustFn] = []

    for node in tree.root_node.children:
        if node.type != "function_item":
            continue

        name_n = node.child_by_field_name("name")
        body_n = node.child_by_field_name("body")
        if name_n is None or body_n is None:
            continue

        params_n = node.child_by_field_name("parameters")
        ret_n = node.child_by_field_name("return_type")

        signature = src[node.start_byte:body_n.start_byte].decode("utf-8")

        params: list[RustParam] = []
        if params_n is not None:
            for p in params_n.named_children:
                if p.type != "parameter":
                    continue  # skips self_parameter, variadic, etc.
                pat = p.child_by_field_name("pattern")
                typ = p.child_by_field_name("type")
                if pat is None or typ is None:
                    continue
                ptype = _text(src, typ)
                ffi_type, kind = map_ffi_type(ptype)
                params.append(RustParam(
                    name=_text(src, pat),
                    rust_type=ptype,
                    ffi_type=ffi_type,
                    kind=kind,
                ))

        return_type = _text(src, ret_n) if ret_n is not None else "()"

        fns.append(RustFn(
            name=_text(src, name_n),
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            signature=signature.rstrip() + " ",
            params=params,
            return_type=return_type,
            is_unsafe="unsafe" in signature.split(),
            body_text=_text(src, body_n),
        ))

    return fns


def last_use_end_byte(source: str) -> int:
    """Byte offset just past the final top-level ``use`` declaration."""
    src = source.encode("utf-8")
    tree = parse(source)
    last = 0
    for node in tree.root_node.children:
        if node.type == "use_declaration":
            last = node.end_byte
    return last


def module_anchor_byte(source: str) -> int:
    """A byte-exact anchor for injecting ``mod`` declarations.

    Prefers the position just past the last top-level ``use`` declaration. If
    the file has none, falls back to the end of the leading run of inner doc
    comments (``//!``) / inner attributes (``#![..]``), so injected modules
    never split a function from its preceding doc comments and never precede a
    mandatory crate-level inner attribute.
    """
    use_end = last_use_end_byte(source)
    if use_end > 0:
        return use_end

    src = source.encode("utf-8")
    tree = parse(source)
    anchor = 0
    for node in tree.root_node.children:
        head = src[node.start_byte:node.start_byte + 3]
        if node.type == "inner_attribute_item":
            anchor = node.end_byte
        elif node.type == "line_comment" and head == b"//!":
            anchor = node.end_byte
        elif node.type == "block_comment" and head == b"/*!":
            anchor = node.end_byte
        else:
            break
    return anchor


# ---------------------------------------------------------------------------
# Byte-exact splicing
# ---------------------------------------------------------------------------

@dataclass
class Edit:
    """A byte-range replacement. ``start == end`` is a pure insertion."""
    start: int
    end: int
    replacement: str


def apply_edits(source: str, edits: list[Edit]) -> str:
    """Apply non-overlapping byte edits back-to-front (offset-stable)."""
    src = source.encode("utf-8")
    ordered = sorted(edits, key=lambda e: e.start, reverse=True)

    last_start = len(src) + 1
    for e in ordered:
        if e.start < 0 or e.end > len(src) or e.start > e.end:
            raise ValueError(f"edit out of range: [{e.start}:{e.end}] (len {len(src)})")
        if e.end > last_start:
            raise ValueError("overlapping edits are not allowed")
        last_start = e.start
        src = src[:e.start] + e.replacement.encode("utf-8") + src[e.end:]

    return src.decode("utf-8")


def deactivation_block(original_text: str, node_id: str, hook: str) -> str:
    """Enclose a function node's exact text in deactivation comments.

    The comment boundaries are tied strictly to *original_text* (the AST node
    slice), so they cannot bleed into neighbouring items. ``/* ... */`` is used
    when safe; if the node text contains a stray ``*/`` (which would close the
    block early), we fall back to line-prefixed ``//`` comments — still clamped
    to the same node text.
    """
    banner = (f"[AERO-DEACTIVATED {node_id}] original hot path — execution "
              f"relocated to aero_ffi::{hook}")
    if "*/" not in original_text:
        return f"/* {banner}\n{original_text}\n*/"
    commented = "\n".join("// " + ln for ln in original_text.split("\n"))
    return f"// {banner}\n{commented}"
