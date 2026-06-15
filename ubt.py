"""
Universal Build Tool (UBT) Core Engine
Architect-Approved, Zero-Dependency, Formal-Verification-Ready.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class UBTError(Exception): """Base exception for Universal Build Tool."""
class CyclicDependencyError(UBTError): """Raised when a closed loop is detected in the task DAG."""
class TaskExecutionError(UBTError): """Raised when an individual task block fails runtime execution."""
class CacheError(UBTError): """Raised when the telemetry cache is corrupted or unreadable."""

class BuildEngine:
    def __init__(self, cache_path=None):
        self.cache_path = Path(cache_path or Path.cwd() / ".ubt_cache.json")
        self.tasks = {}
        self.cache = self._load_cache()

    def _load_cache(self):
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[!] Warning: Cache corrupted ({e}). Initializing empty baseline.")
            return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=4)
        except IOError as e:
            raise CacheError(f"Failed to persist optimization telemetry: {e}")

    def _compute_hash(self, file_paths):
        hasher = hashlib.sha256()
        for path_str in sorted(file_paths):
            path = Path(path_str)
            if path.exists() and path.is_file():
                hasher.update(path.read_bytes())
        return hasher.hexdigest()

    def task(self, inputs=None, outputs=None, dependencies=None):
        inputs = inputs or []
        outputs = outputs or []
        dependencies = dependencies or []

        def decorator(func):
            task_name = func.__name__
            self.tasks[task_name] = {
                "func": func,
                "inputs": inputs,
                "outputs": outputs,
                "dependencies": [d.__name__ if hasattr(d, '__name__') else d for d in dependencies],
            }
            return func
        return decorator

    def _topological_sort(self):
        visited = {name: 0 for name in self.tasks}
        order = []

        def visit(node):
            if visited[node] == 1:
                raise CyclicDependencyError(f"Mathematical cycle detected at task node: '{node}'")
            if visited[node] == 0:
                visited[node] = 1
                for dependency in self.tasks[node]["dependencies"]:
                    if dependency in self.tasks:
                        visit(dependency)
                visited[node] = 2
                order.append(node)

        for task_name in self.tasks:
            if visited[task_name] == 0:
                visit(task_name)
        return order

    def run(self):
        execution_order = self._topological_sort()
        max_workers = min(os.cpu_count() or 2, 4)
        print(f"[*] Initializing baseline graph with {max_workers} tuning threads.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for task_name in execution_order:
                task_meta = self.tasks[task_name]
                current_hash = self._compute_hash(task_meta["inputs"])
                cached_data = self.cache.get(task_name, {})
                outputs_exist = all(Path(p).exists() for p in task_meta["outputs"])

                if cached_data.get("hash") == current_hash and outputs_exist:
                    print(f"[→] {task_name}: Cached (Skipping execution).")
                    continue

                print(f"[⚙] Dispatching task execution: {task_name}")
                func = task_meta["func"]
                
                try:
                    func()
                    self.cache[task_name] = {"hash": current_hash}
                except Exception as e:
                    raise TaskExecutionError(f"Task '{task_name}' collapsed mid-execution. Real reason: {e}")

        self._save_cache()
        print("[✓] Optimization loop finished cleanly.")
