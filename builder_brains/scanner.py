# -*- coding: utf-8 -*-
"""
scanner.py — Concurrent Multi-File Signature Scanning & Anomaly Detection Engine

Implements:
  - Concurrent multi-file signature scanning (ThreadPoolExecutor) with
    thread-safe telemetry collection: workers are pure routines returning
    ``(scan_result, error_payload)`` tuples and the main thread consolidates
    all shared state as futures resolve
  - Regex-based token profiling with LRU-cached compiled patterns
  - Cryptographic file-delta diffing (SHA-256 fingerprinting)
  - Structural anomaly categorization maps (z-score based)

Pipeline entry point: evaluate(metadata, hyper_params)
"""

import hashlib
import json
import logging
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("builder_brains.scanner")

_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "build_manifest.json")


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def _load_manifest() -> Dict[str, Any]:
    try:
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load build_manifest.json: %s — using defaults", exc)
        return {}


def _get_scanner_params(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("hyperparameter_weights", {}).get("scanner", {})


def _get_thresholds(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("thresholds", {})


def _get_cost_ceilings(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("execution_cost_ceilings", {})


# ---------------------------------------------------------------------------
# Regex-based token profiler
# ---------------------------------------------------------------------------

_TOKEN_PATTERNS: Dict[str, str] = {
    "function_def": r"\bdef\s+[a-zA-Z_]\w*\s*\(",
    "class_def": r"\bclass\s+[a-zA-Z_]\w*\s*[\(:]",
    "import_stmt": r"^\s*(?:from\s+\S+\s+)?import\s+",
    "decorator": r"^\s*@[a-zA-Z_]\w*",
    "string_literal": r"(?:\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|\"[^\"\\]*(?:\\.[^\"\\]*)*\"|'[^'\\]*(?:\\.[^'\\]*)*')",
    "numeric_literal": r"\b(?:0[xXoObB])?[\d]+(?:\.[\d]+)?(?:[eE][+-]?\d+)?\b",
    "comment_line": r"#[^\n]*",
    "assignment": r"[a-zA-Z_]\w*\s*(?:[:+\-*/|&^]=|=(?!=))",
    "return_stmt": r"\breturn\b",
    "raise_stmt": r"\braise\b",
    "try_block": r"\btry\s*:",
    "except_block": r"\bexcept\b",
    "with_stmt": r"\bwith\b",
    "yield_expr": r"\byield\b",
    "lambda_expr": r"\blambda\b",
    "list_comp": r"\[.+\bfor\b.+\bin\b.+\]",
    "dict_comp": r"\{.+:\s*.+\bfor\b.+\bin\b.+\}",
    "assert_stmt": r"\bassert\b",
    "global_stmt": r"\bglobal\b",
    "nonlocal_stmt": r"\bnonlocal\b",
    "async_def": r"\basync\s+def\b",
    "await_expr": r"\bawait\b",
}


@lru_cache(maxsize=4096)
def _compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.MULTILINE)


class TokenProfiler:
    """Profile source code by counting regex-matched token categories."""

    def __init__(self, cache_size: int = 4096) -> None:
        self._cache_size: int = cache_size
        self._compiled: Dict[str, re.Pattern[str]] = {}
        for name, pat in _TOKEN_PATTERNS.items():
            self._compiled[name] = _compile_pattern(pat)

    def profile(self, source: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for name, pattern in self._compiled.items():
            matches = pattern.findall(source)
            counts[name] = len(matches)
        return counts

    def profile_with_positions(self, source: str) -> Dict[str, List[Tuple[int, int]]]:
        positions: Dict[str, List[Tuple[int, int]]] = {}
        for name, pattern in self._compiled.items():
            positions[name] = [(m.start(), m.end()) for m in pattern.finditer(source)]
        return positions


# ---------------------------------------------------------------------------
# N-gram token analysis
# ---------------------------------------------------------------------------

class TokenNgramAnalyzer:
    """Compute n-gram frequency distributions over token-category sequences."""

    def __init__(self, window: int = 4) -> None:
        self._window: int = max(1, window)

    def extract_ngrams(self, token_sequence: List[str]) -> Counter:
        ngrams: Counter = Counter()
        for i in range(len(token_sequence) - self._window + 1):
            gram = tuple(token_sequence[i : i + self._window])
            ngrams[gram] += 1
        return ngrams

    def build_sequence_from_positions(
        self,
        positions: Dict[str, List[Tuple[int, int]]],
    ) -> List[str]:
        events: List[Tuple[int, str]] = []
        for token_type, pos_list in positions.items():
            for start, _ in pos_list:
                events.append((start, token_type))
        events.sort(key=lambda x: x[0])
        return [token_type for _, token_type in events]

    def analyze(self, source: str, profiler: TokenProfiler) -> Dict[str, Any]:
        positions = profiler.profile_with_positions(source)
        sequence = self.build_sequence_from_positions(positions)
        ngrams = self.extract_ngrams(sequence)

        total_ngrams = sum(ngrams.values())
        top_ngrams = ngrams.most_common(20)

        return {
            "total_ngrams": total_ngrams,
            "unique_ngrams": len(ngrams),
            "top_ngrams": [
                {"ngram": list(ng), "count": ct} for ng, ct in top_ngrams
            ],
            "sequence_length": len(sequence),
        }


# ---------------------------------------------------------------------------
# Cryptographic file-delta diffing
# ---------------------------------------------------------------------------

class CryptoDeltaDiffer:
    """Compute SHA-256 fingerprints for files and detect content changes."""

    def __init__(self, algorithm: str = "sha256") -> None:
        self._algorithm: str = algorithm
        self._fingerprint_cache: Dict[str, str] = {}

    def fingerprint(self, content: str) -> str:
        h = hashlib.new(self._algorithm)
        h.update(content.encode("utf-8"))
        return h.hexdigest()

    def fingerprint_bytes(self, data: bytes) -> str:
        h = hashlib.new(self._algorithm)
        h.update(data)
        return h.hexdigest()

    def compute_delta(
        self,
        old_content: str,
        new_content: str,
    ) -> Dict[str, Any]:
        old_fp = self.fingerprint(old_content)
        new_fp = self.fingerprint(new_content)
        changed = old_fp != new_fp

        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()

        added_count = 0
        removed_count = 0
        old_line_set: Set[str] = set(old_lines)
        new_line_set: Set[str] = set(new_lines)
        added_count = len(new_line_set - old_line_set)
        removed_count = len(old_line_set - new_line_set)

        size_delta = len(new_content) - len(old_content)

        return {
            "changed": changed,
            "old_fingerprint": old_fp,
            "new_fingerprint": new_fp,
            "old_line_count": len(old_lines),
            "new_line_count": len(new_lines),
            "lines_added": added_count,
            "lines_removed": removed_count,
            "size_delta_bytes": size_delta,
        }

    def batch_fingerprint(
        self,
        file_contents: Dict[str, str],
    ) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for path, content in file_contents.items():
            result[path] = self.fingerprint(content)
        return result

    def detect_duplicates(
        self,
        file_fingerprints: Dict[str, str],
    ) -> Dict[str, List[str]]:
        fp_to_files: Dict[str, List[str]] = defaultdict(list)
        for path, fp in file_fingerprints.items():
            fp_to_files[fp].append(path)
        return {fp: paths for fp, paths in fp_to_files.items() if len(paths) > 1}


# ---------------------------------------------------------------------------
# Structural anomaly categorization
# ---------------------------------------------------------------------------

class AnomalyCategory:
    """A categorized anomaly with severity and context."""

    __slots__ = ("category", "severity", "zscore", "metric_name", "metric_value", "context")

    def __init__(
        self,
        category: str,
        severity: str,
        zscore: float,
        metric_name: str,
        metric_value: float,
        context: str = "",
    ) -> None:
        self.category: str = category
        self.severity: str = severity
        self.zscore: float = zscore
        self.metric_name: str = metric_name
        self.metric_value: float = metric_value
        self.context: str = context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "zscore": round(self.zscore, 4),
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 4),
            "context": self.context,
        }


class AnomalyCategorizer:
    """Detect structural anomalies in token profiles using z-score analysis.

    Anomaly categories:
      - complexity_spike: Unusually high function/class definition density
      - import_bloat: Excessive import statements relative to code body
      - comment_desert: Abnormally low comment-to-code ratio
      - error_handling_deficit: Low try/except density relative to complexity
      - magic_number_overload: High numeric literal density
    """

    CATEGORY_RULES: Dict[str, Dict[str, Any]] = {
        "complexity_spike": {
            "metrics": ["function_def", "class_def"],
            "description": "Unusually high definition density",
        },
        "import_bloat": {
            "metrics": ["import_stmt"],
            "description": "Excessive import statements",
        },
        "comment_desert": {
            "metrics": ["comment_line"],
            "description": "Abnormally low comment density",
            "invert": True,
        },
        "error_handling_deficit": {
            "metrics": ["try_block", "except_block"],
            "description": "Insufficient error handling",
            "invert": True,
        },
        "magic_number_overload": {
            "metrics": ["numeric_literal"],
            "description": "Excessive magic numbers",
        },
    }

    def __init__(self, zscore_threshold: float = 2.5) -> None:
        self._threshold: float = zscore_threshold
        self._baseline_stats: Dict[str, Tuple[float, float]] = {}

    def fit_baseline(self, profiles: List[Dict[str, int]]) -> None:
        if not profiles:
            return

        metric_values: Dict[str, List[float]] = defaultdict(list)
        for profile in profiles:
            for metric, count in profile.items():
                metric_values[metric].append(float(count))

        for metric, values in metric_values.items():
            if len(values) >= 2:
                mean = statistics.mean(values)
                stdev = statistics.stdev(values)
                self._baseline_stats[metric] = (mean, stdev)
            elif len(values) == 1:
                self._baseline_stats[metric] = (values[0], 1.0)

    def categorize(self, profile: Dict[str, int]) -> List[AnomalyCategory]:
        anomalies: List[AnomalyCategory] = []

        for category, rule in self.CATEGORY_RULES.items():
            metric_names: List[str] = rule["metrics"]
            is_inverted: bool = rule.get("invert", False)

            for metric_name in metric_names:
                value = float(profile.get(metric_name, 0))
                stats = self._baseline_stats.get(metric_name)

                if stats is None:
                    continue

                mean, stdev = stats
                if stdev < 1e-9:
                    continue

                zscore = (value - mean) / stdev

                is_anomalous = False
                if is_inverted and zscore < -self._threshold:
                    is_anomalous = True
                elif not is_inverted and zscore > self._threshold:
                    is_anomalous = True

                if is_anomalous:
                    severity = "critical" if abs(zscore) > self._threshold * 2 else "warning"
                    anomalies.append(
                        AnomalyCategory(
                            category=category,
                            severity=severity,
                            zscore=zscore,
                            metric_name=metric_name,
                            metric_value=value,
                            context=rule.get("description", ""),
                        )
                    )

        return anomalies


# ---------------------------------------------------------------------------
# Concurrent multi-file scanner
# ---------------------------------------------------------------------------

class FileScanner:
    """Scan multiple files concurrently, producing token profiles and
    cryptographic fingerprints for each."""

    def __init__(
        self,
        worker_count: int = 8,
        max_file_bytes: int = 10_485_760,
    ) -> None:
        self._worker_count: int = max(1, worker_count)
        self._max_file_bytes: int = max_file_bytes
        self._profiler: TokenProfiler = TokenProfiler()
        self._differ: CryptoDeltaDiffer = CryptoDeltaDiffer()
        self._scan_errors: List[Dict[str, str]] = []

    def _scan_single_file(
        self, file_path: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
        """Scan one file as a pure, side-effect-free worker routine.

        Runs inside pool threads, so it has zero write-access to the parent
        scanner's state space — no shared list appends, no attribute
        mutation. All telemetry travels back through the returned
        ``(scan_result, error_payload)`` tuple (``error_payload`` is ``None``
        on success) and is consolidated by the main thread in ``scan_files``.
        """
        result: Dict[str, Any] = {"path": file_path, "status": "pending"}
        error_payload: Optional[Dict[str, str]] = None
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self._max_file_bytes:
                result["status"] = "skipped_too_large"
                result["file_size"] = file_size
                return result, None

            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()

            result["file_size"] = len(content)
            result["fingerprint"] = self._differ.fingerprint(content)
            result["line_count"] = content.count("\n") + (1 if content else 0)
            result["token_profile"] = self._profiler.profile(content)
            result["status"] = "scanned"

        except PermissionError:
            result["status"] = "permission_denied"
            error_payload = {"path": file_path, "error": "permission_denied"}
        except OSError as exc:
            result["status"] = "os_error"
            result["error"] = str(exc)
            error_payload = {"path": file_path, "error": str(exc)}
        except Exception as exc:
            result["status"] = "unexpected_error"
            result["error"] = str(exc)
            error_payload = {"path": file_path, "error": str(exc)}

        return result, error_payload

    def scan_files(self, file_paths: List[str]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        self._scan_errors = []

        scan_start = time.monotonic()

        with ThreadPoolExecutor(max_workers=self._worker_count) as executor:
            future_to_path: Dict[
                Future[Tuple[Dict[str, Any], Optional[Dict[str, str]]]], str
            ] = {}
            for path in file_paths:
                future = executor.submit(self._scan_single_file, path)
                future_to_path[future] = path

            # Harvest the workers' (scan_result, error_payload) tuples inside
            # the main thread frame as futures resolve — only this thread ever
            # touches the shared telemetry lists.
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result, error_payload = future.result()
                    results.append(result)
                    if error_payload is not None:
                        errors.append(error_payload)
                except Exception as exc:
                    logger.error("Scanner thread failed for %s: %s", path, exc)
                    results.append({
                        "path": path,
                        "status": "thread_failure",
                        "error": str(exc),
                    })
                    errors.append({"path": path, "error": str(exc)})

        # Sequentially consolidate the errors list in the main thread before
        # returning (the attribute is retained for downstream observability).
        self._scan_errors = list(errors)

        scan_duration = round(time.monotonic() - scan_start, 6)

        scanned_count = sum(1 for r in results if r["status"] == "scanned")
        failed_count = len(results) - scanned_count

        return {
            "results": results,
            "scanned_count": scanned_count,
            "failed_count": failed_count,
            "total_files": len(file_paths),
            "scan_duration_seconds": scan_duration,
            "errors": errors,
        }

    def collect_profiles(
        self,
        scan_results: Dict[str, Any],
    ) -> Tuple[List[Dict[str, int]], Dict[str, str]]:
        profiles: List[Dict[str, int]] = []
        fingerprints: Dict[str, str] = {}
        for entry in scan_results.get("results", []):
            if entry.get("status") == "scanned":
                profiles.append(entry.get("token_profile", {}))
                fingerprints[entry["path"]] = entry.get("fingerprint", "")
        return profiles, fingerprints


# ---------------------------------------------------------------------------
# Similarity analysis
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: Dict[str, int], vec_b: Dict[str, int]) -> float:
    keys = set(vec_a.keys()) | set(vec_b.keys())
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in keys)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values())) or 1e-9
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values())) or 1e-9
    return dot / (mag_a * mag_b)


def find_structurally_similar_files(
    scan_results: Dict[str, Any],
    cutoff: float = 0.80,
) -> List[Dict[str, Any]]:
    entries = [r for r in scan_results.get("results", []) if r.get("status") == "scanned"]
    similar_pairs: List[Dict[str, Any]] = []

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            profile_a = entries[i].get("token_profile", {})
            profile_b = entries[j].get("token_profile", {})
            sim = _cosine_similarity(profile_a, profile_b)
            if sim >= cutoff:
                similar_pairs.append({
                    "file_a": entries[i]["path"],
                    "file_b": entries[j]["path"],
                    "similarity": round(sim, 6),
                })

    similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return similar_pairs


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def evaluate(metadata: Dict[str, Any], hyper_params: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full scanning pipeline on files referenced in metadata.

    Parameters
    ----------
    metadata : dict
        Must contain ``"scan_targets"`` (list of file paths) or
        ``"workspace_root"`` (str) to auto-discover Python files.
        Optionally ``"previous_fingerprints"`` (dict) for delta diffing.
    hyper_params : dict
        Runtime overrides merged on top of ``build_manifest.json`` defaults.

    Returns
    -------
    dict
        The enriched metadata dictionary with scan results.
    """
    start_time: float = time.monotonic()
    logger.info("Scanner pipeline started")

    manifest: Dict[str, Any] = _load_manifest()
    params: Dict[str, Any] = {**_get_scanner_params(manifest), **hyper_params}
    thresholds: Dict[str, Any] = _get_thresholds(manifest)
    ceilings: Dict[str, Any] = _get_cost_ceilings(manifest)

    wall_limit: float = ceilings.get("scanner_max_wall_seconds", 60.0)
    worker_count: int = int(params.get("concurrent_worker_pool_size", 8))
    max_file_bytes: int = int(params.get("max_file_scan_bytes", 10_485_760))
    zscore_threshold: float = float(params.get("anomaly_zscore_threshold", 2.5))
    ngram_window: int = int(params.get("token_ngram_window", 4))
    similarity_cutoff: float = float(params.get("structural_similarity_cutoff", 0.80))
    coverage_target: float = float(thresholds.get("scan_coverage_target", 0.95))
    anomaly_ceiling: int = int(thresholds.get("anomaly_alert_ceiling", 50))

    # --- Resolve scan targets ---
    scan_targets: List[str] = list(metadata.get("scan_targets", []))
    workspace_root: str = metadata.get("workspace_root", "")

    if not scan_targets and workspace_root and os.path.isdir(workspace_root):
        for dirpath, _dirnames, filenames in os.walk(workspace_root):
            for fname in filenames:
                if fname.endswith(".py"):
                    full_path = os.path.join(dirpath, fname)
                    scan_targets.append(full_path)
        logger.info("Auto-discovered %d Python files under %s", len(scan_targets), workspace_root)

    if not scan_targets:
        logger.warning("No scan targets provided — returning early")
        metadata["scanner_status"] = "skipped_no_targets"
        metadata["scanner_wall_seconds"] = 0.0
        return metadata

    metadata["scan_target_count"] = len(scan_targets)

    # --- Stage 1: Concurrent file scanning ---
    scanner = FileScanner(worker_count=worker_count, max_file_bytes=max_file_bytes)
    scan_output = scanner.scan_files(scan_targets)
    metadata["scan_summary"] = {
        "scanned": scan_output["scanned_count"],
        "failed": scan_output["failed_count"],
        "total": scan_output["total_files"],
        "duration_seconds": scan_output["scan_duration_seconds"],
    }
    metadata["scan_errors"] = scan_output["errors"]

    coverage = scan_output["scanned_count"] / max(scan_output["total_files"], 1)
    metadata["scan_coverage"] = round(coverage, 6)
    metadata["scan_coverage_passed"] = coverage >= coverage_target
    logger.info(
        "Scan complete: %d/%d files (%.1f%% coverage)",
        scan_output["scanned_count"],
        scan_output["total_files"],
        coverage * 100,
    )

    # --- Stage 2: Aggregate token profiles ---
    elapsed = time.monotonic() - start_time
    profiles, fingerprints = scanner.collect_profiles(scan_output)
    metadata["file_fingerprints"] = fingerprints

    aggregate_profile: Counter = Counter()
    for profile in profiles:
        aggregate_profile.update(profile)
    metadata["aggregate_token_profile"] = dict(aggregate_profile)

    # --- Stage 3: Delta diffing against previous fingerprints ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit:
        previous_fps: Dict[str, str] = metadata.get("previous_fingerprints", {})
        if previous_fps:
            differ = CryptoDeltaDiffer()
            changed_files: List[str] = []
            new_files: List[str] = []
            removed_files: List[str] = []

            for path, fp in fingerprints.items():
                old_fp = previous_fps.get(path)
                if old_fp is None:
                    new_files.append(path)
                elif old_fp != fp:
                    changed_files.append(path)

            for path in previous_fps:
                if path not in fingerprints:
                    removed_files.append(path)

            metadata["delta_summary"] = {
                "changed_files": changed_files,
                "new_files": new_files,
                "removed_files": removed_files,
                "changed_count": len(changed_files),
                "new_count": len(new_files),
                "removed_count": len(removed_files),
            }
            logger.info(
                "Delta diff: %d changed, %d new, %d removed",
                len(changed_files), len(new_files), len(removed_files),
            )

            # Detect duplicates
            duplicates = differ.detect_duplicates(fingerprints)
            if duplicates:
                metadata["duplicate_groups"] = {
                    fp: paths for fp, paths in duplicates.items()
                }
                logger.info("Found %d duplicate file groups", len(duplicates))
        else:
            metadata["delta_summary"] = {"status": "no_previous_fingerprints"}

    # --- Stage 4: Anomaly detection ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit and profiles:
        categorizer = AnomalyCategorizer(zscore_threshold=zscore_threshold)
        categorizer.fit_baseline(profiles)

        all_anomalies: List[Dict[str, Any]] = []
        for idx, scan_entry in enumerate(scan_output.get("results", [])):
            if scan_entry.get("status") != "scanned":
                continue
            file_profile = scan_entry.get("token_profile", {})
            file_anomalies = categorizer.categorize(file_profile)
            for anomaly in file_anomalies:
                entry = anomaly.to_dict()
                entry["file"] = scan_entry.get("path", f"file_{idx}")
                all_anomalies.append(entry)

        if len(all_anomalies) > anomaly_ceiling:
            logger.warning(
                "Anomaly count %d exceeds ceiling %d — truncating",
                len(all_anomalies), anomaly_ceiling,
            )
            all_anomalies = sorted(
                all_anomalies, key=lambda a: abs(a.get("zscore", 0)), reverse=True
            )[:anomaly_ceiling]

        metadata["anomalies"] = all_anomalies
        metadata["anomaly_count"] = len(all_anomalies)

        severity_counts: Dict[str, int] = defaultdict(int)
        for a in all_anomalies:
            severity_counts[a.get("severity", "unknown")] += 1
        metadata["anomaly_severity_summary"] = dict(severity_counts)

        logger.info(
            "Anomaly detection: %d anomalies found (%s)",
            len(all_anomalies),
            dict(severity_counts),
        )
    else:
        metadata["anomalies"] = []
        metadata["anomaly_count"] = 0

    # --- Stage 5: N-gram analysis ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit and profiles:
        ngram_analyzer = TokenNgramAnalyzer(window=ngram_window)
        profiler = TokenProfiler()

        ngram_summaries: List[Dict[str, Any]] = []
        for scan_entry in scan_output.get("results", []):
            if scan_entry.get("status") != "scanned":
                continue
            path = scan_entry.get("path", "")
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                ngram_result = ngram_analyzer.analyze(content, profiler)
                ngram_summaries.append({
                    "file": path,
                    "total_ngrams": ngram_result["total_ngrams"],
                    "unique_ngrams": ngram_result["unique_ngrams"],
                })
            except Exception as exc:
                logger.debug("N-gram analysis skipped for %s: %s", path, exc)

        metadata["ngram_analysis"] = ngram_summaries
        logger.info("N-gram analysis completed for %d files", len(ngram_summaries))

    # --- Stage 6: Structural similarity ---
    elapsed = time.monotonic() - start_time
    if elapsed < wall_limit:
        similar_pairs = find_structurally_similar_files(scan_output, cutoff=similarity_cutoff)
        metadata["structurally_similar_pairs"] = similar_pairs[:50]
        if similar_pairs:
            logger.info("Found %d structurally similar file pairs", len(similar_pairs))

    # --- Finalize ---
    metadata["scan_status"] = "complete"
    wall_seconds = round(time.monotonic() - start_time, 6)
    metadata["scanner_wall_seconds"] = wall_seconds
    metadata["scanner_status"] = (
        "budget_exceeded" if wall_seconds > wall_limit else "complete"
    )

    logger.info(
        "Scanner pipeline finished in %.3fs — %d files, %d anomalies",
        wall_seconds,
        scan_output["scanned_count"],
        metadata.get("anomaly_count", 0),
    )
    return metadata
