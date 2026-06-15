# Aero Build Engine

A high-performance, zero-dependency, concurrent task and build orchestration engine written entirely in pure Python.

## Overview

UBT is a lightweight, minimalist pipeline orchestrator designed to manage complex task dependency graphs (DAGs) without the bloat of external runtimes or heavy third-party packages. By leveraging the Python standard library, UBT provides deterministic task execution, cryptographic incremental rebuilds, and dynamic concurrency scaling with a zero-install footprint.

### Key Features

- **Zero Dependencies:** Built strictly on standard Python modules (`hashlib`, `concurrent.futures`, `json`).
- **Deterministic DAG Scheduling:** Automatic topological sorting prevents race conditions and ensures exact task execution sequencing.
- **Cryptographic Incremental Rebuilds:** Uses SHA-256 file fingerprinting to detect source changes, skipping up-to-date tasks to minimize compute overhead.
- **Self-Optimizing Concurrency:** Analyzes task runtime graphs to dynamically size worker pools, maximizing CPU and I/O efficiency.
- **Cycle-Safe Execution Engine:** Integrated cycle detection algorithms proactively intercept circular dependencies and fail-fast to prevent infinite loops.

---

## Architecture

UBT works by modeling your build process as a Directed Acyclic Graph (DAG):

1. **Graph Compilation:** Tasks are registered along with their inputs, outputs, and explicit upstreams.
2. **Topological Ordering:** The execution engine checks the graph for cycles and orders tasks deterministically.
3. **Telemetry Filtering:** The state cache engine compares current SHA-256 fingerprints against historical execution logs (`.ubt_cache.json`). Unchanged branches are safely skipped.
4. **Concurrent Execution:** Validated tasks are dispatched to an auto-tuning thread worker pool.

---

## Getting Started

### Installation

Simply clone the repository and drop the core orchestration module into your project. No `pip install` required.

```bash
git clone [https://github.com/sys1own/trainer.git](https://github.com/sys1own/trainer.git)
cd trainer
