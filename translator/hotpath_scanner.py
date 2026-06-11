"""Scan target directories, fingerprint files, and identify performance
hot-path candidates based on frequency and payload density."""

import os
import hashlib
import re
from dataclasses import dataclass, field


@dataclass
class ScannedFile:
    path: str
    size: int
    fingerprint: str
    fields: dict = field(default_factory=dict)


@dataclass
class HotPath:
    source_files: list
    pattern_id: str
    weight: int
    label: str


def fingerprint(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_kv_file(path: str) -> dict:
    """Extract KEY=VALUE pairs from a raw telemetry-style file."""
    fields = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.+)$", line)
            if m:
                fields[m.group(1)] = m.group(2)
    return fields


def scan_directory(target_dir: str) -> list:
    """Walk *target_dir* and return a list of ``ScannedFile`` records."""
    results = []
    for root, _dirs, files in os.walk(target_dir):
        for name in sorted(files):
            full = os.path.join(root, name)
            if not os.path.isfile(full):
                continue
            sf = ScannedFile(
                path=full,
                size=os.path.getsize(full),
                fingerprint=fingerprint(full),
                fields=parse_kv_file(full),
            )
            results.append(sf)
    return results


def identify_hotpaths(scanned: list, *, weight_threshold: int = 2) -> list:
    """Group scanned files by shared field schemas and promote groups that
    exceed *weight_threshold* into hot-path candidates."""
    schema_groups: dict[tuple, list] = {}
    for sf in scanned:
        key = tuple(sorted(sf.fields.keys()))
        schema_groups.setdefault(key, []).append(sf)

    hotpaths = []
    for idx, (schema, group) in enumerate(schema_groups.items()):
        if len(group) < weight_threshold:
            continue
        hotpaths.append(HotPath(
            source_files=[sf.path for sf in group],
            pattern_id=f"hp{idx}",
            weight=len(group),
            label="_".join(schema).lower() if schema else "raw",
        ))
    return hotpaths
