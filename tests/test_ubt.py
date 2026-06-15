import pytest
from pathlib import Path
from ubt import BuildEngine, CyclicDependencyError, TaskExecutionError

def test_clean_baseline_initialization():
    engine = BuildEngine(cache_path=".test_cache.json")
    assert len(engine.tasks) == 0

def test_infinite_loop_and_cyclic_protection():
    engine = BuildEngine(cache_path=".test_cache.json")
    
    @engine.task(dependencies=["task_b"])
    def task_a(): pass

    @engine.task(dependencies=["task_a"])
    def task_b(): pass

    with pytest.raises(CyclicDependencyError) as exc_info:
        engine.run()
    
    assert "cycle detected" in str(exc_info.value).lower()

def test_task_failure_propagation():
    engine = BuildEngine(cache_path=".test_cache.json")

    @engine.task()
    def broken_task():
        raise ZeroDivisionError("Core meltdown")

    with pytest.raises(TaskExecutionError):
        engine.run()

def teardown_module(module):
    p = Path(".test_cache.json")
    if p.exists():
        p.unlink()