# Aero Build Engine

A high-performance, zero-dependency, concurrent task and build orchestration engine written entirely in pure Python. Optimized for massive monorepos, ultra-complex dependency chains, and sub-second incremental execution.

## The Aero Philosophy: Zero Waste, Infinite Scale

Traditional build tools grow slow and bloated as codebases expand. **Aero Build Engine** solves this by treating your entire software ecosystem as a mathematical graph.

Aero achieves its hyper-fast execution speeds not through heavy AI models or opaque LLM layers, but through **deterministic cryptographic short-circuiting**. By tracking SHA-256 file fingerprints and evaluating dependencies using localized graph theory, Aero computes the exact delta of a codebase in milliseconds. If a file or its upstreams haven't changed, Aero bypasses it entirely. The result? **Large-scale builds that used to take minutes are safely executed in seconds.**

---

## What Can Aero Build?

Aero is designed without arbitrary scale limitations. If your pipeline can be modeled as a directed workflow, Aero can orchestrate it. It is uniquely suited for:

* **Massive Polyglot Monorepos:** Coordinate builds involving mixtures of C/C++, Rust, Go, Python, and Web assets simultaneously without cross-contamination.
* **Complex Codebase Asset Staging:** Automate code linting, structural testing, minification, and multi-tier compilation in non-blocking parallel tracks.
* **Deterministic Deployment Pipelines:** Package binaries, containerize environments, and generate system performance metrics with an ironclad guarantee of execution order.

---

## Core Technical Features

### 1. Zero External Dependencies

Built entirely on top of the native Python Standard Library (`hashlib`, `concurrent.futures`, `json`, `pathlib`). Aero introduces **zero security supply-chain risks**, requires no `pip install` overhead, and boasts a virtually nonexistent security attack surface.

### 2. Cycle-Safe Topological Scheduling

Using an advanced implementation of Graph Coloring and Kahn's Algorithm, Aero evaluates task trees before a single line of code runs. If a cyclic dependency (e.g., Task A requiring Task B, which requires Task A) is introduced, Aero catches it instantly and halts safely to protect system resources.

### 3. Cryptographic State Telemetry

Every task's inputs and outputs are hashed using SHA-256. This state data loops back into a local cache database (`.ubt_cache.json`). On subsequent runs, Aero performs a high-speed telemetry audit, automatically skipping branches that are already up-to-date.

### 4. Self-Optimizing Resource Allocation

Aero actively queries your host machine's hardware profile on launch. It dynamically auto-tunes its internal `ThreadPoolExecutor` worker boundary limits to perfectly balance maximum concurrent throughput against CPU kernel context-switching overhead.

---

## Architecture Blueprint

Aero processes work through a strict four-stage deterministic lifecycle:

```
[ User Definitions ] ➔ [ Graph Compilation ] ➔ [ Cycle Verification ] ➔ [ Cryptographic Filtering ] ➔ [ Thread-Pool Execution ]

```

1. **Registration:** Tasks declare explicit file dependencies, target outputs, and upstream prerequisites via decorators.
2. **Topological Sort:** The graph engine flattens the multi-dimensional task matrix into a linear, non-conflicting execution array.
3. **Cache Assessment:** Aero checks the disk against historical hashes. Up-to-date tracks are offloaded out of the active thread pool.
4. **Concurrent Processing:** Eligible tasks are securely executed across isolated workers.

---

## Getting Started & Configuration

### Installation

Simply clone this repository and drop the core engine file (`ubt.py`) right into your project directory.

```bash
git clone https://github.com/sys1own/trainer.git
cd trainer

```

### Creating an Advanced Configurable Blueprint

Aero allows you to build completely customizable pipelines using straightforward Python scripts. Create a file named `blueprint.aero` to map out your infrastructure:

```python
import os
import json
from pathlib import Path
from ubt import BuildEngine

# Initialize the Aero context
engine = BuildEngine()

# Establish a shared configuration matrix
CONFIG = {
    "ENV": "production",
    "TARGET_DIR": "dist_output",
    "SOURCE_FILE": "src/main.py"
}

# Ensure directories exist
Path(CONFIG["TARGET_DIR"]).mkdir(exist_ok=True)

@engine.task(inputs=[CONFIG["SOURCE_FILE"]], outputs=[f"{CONFIG['TARGET_DIR']}/build_log.txt"])
def compile_workspace_assets():
    print(f"[*] Aero Compiler processing: {CONFIG['SOURCE_FILE']}")
    # Your compilation, transformation, or bundling logic goes here
    with open(f"{CONFIG['TARGET_DIR']}/build_log.txt", "w") as f:
        f.write(f"Environment {CONFIG['ENV']} built cleanly.")

if __name__ == "__main__":
    print(f"=== Running Aero Build Engine for {CONFIG['ENV']} ===")
    engine.run()

```

Run your pipeline file from your terminal:

```bash
python blueprint.aero

```

---

## Troubleshooting & Verification Fail-Safes

Aero is built to fail predictably and gracefully rather than throwing generic Python tracebacks.

| Error Type | Triggering Condition | Resolution Strategy |
| --- | --- | --- |
| `CyclicDependencyError` | A circular loop is found in your task relationships. | Audit your task `dependencies=[]` parameters to break the closed loop. |
| `TaskExecutionError` | A custom task crashed or threw an error mid-execution. | Review the standard output logs printed directly below the task dispatch indicator. |
| `CacheError` | The telemetry tracker is blocked from writing to disk. | Ensure user permissions allow file writes inside your project root. |

---

## License

This project is open-source software licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
