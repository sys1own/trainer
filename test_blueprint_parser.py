# -*- coding: utf-8 -*-
"""
Unit tests for blueprint_parser.py.
"""

import os
import tempfile
import unittest

from blueprint_parser import BlueprintParseError, detect_cycles, parse_blueprint, parse_blueprint_content


class TestBlueprintParser(unittest.TestCase):

    def test_cycle_detection(self):
        acyclic = {
            "taskA": ["taskB", "taskC"],
            "taskB": ["taskD"],
            "taskC": [],
            "taskD": [],
        }
        self.assertEqual(detect_cycles(acyclic), [])

        cyclic_direct = {
            "taskA": ["taskB"],
            "taskB": ["taskA"],
        }
        cycle = detect_cycles(cyclic_direct)
        self.assertIn("taskA", cycle)
        self.assertIn("taskB", cycle)

        cyclic_long = {
            "taskA": ["taskB"],
            "taskB": ["taskC"],
            "taskC": ["taskD"],
            "taskD": ["taskB"],
        }
        cycle2 = detect_cycles(cyclic_long)
        self.assertTrue(len(cycle2) > 0)
        self.assertNotIn("taskA", cycle2)

    def test_valid_monolithic_schema_parsing(self):
        content = """
        [graph]
        entrypoint = orchestrator
        targets = ["scanner", "decision_tree", "parameter_tuner"]
        dependencies = {"scanner": [], "decision_tree": ["scanner"], "parameter_tuner": ["decision_tree"]}
        workspace_mode = incremental
        allow_partial_graph = false

        [compiler]
        profile_guided_optimization = enabled_strict
        tier_shifting_hotness_threshold = 100
        hotspot_loop_unroll_depth = 32
        aot_boundary_check_elimination = true
        vector_intrinsics_auto_generation = true
        pipeline_budget_seconds = 120.0
        max_memory_mb = 2048

        [cortex]
        consensus_protocol = raft_driven_mutation_lock
        mutation_entropy_clamp_threshold = 0.05
        total_cooperating_agents = 8
        heuristic_exploration_depth = 3
        execution_mode = lock_free_polling_wheel_realtime
        core_affinity_mask = 0xFFFF
        numa_node_locality_binding = true
        inter_core_ring_buffer_capacity = 262144
        """
        sections, deps = parse_blueprint_content(content)

        self.assertEqual(sections["graph"]["entrypoint"], "orchestrator")
        self.assertEqual(sections["compiler"]["pipeline_budget_seconds"], 120.0)
        self.assertEqual(sections["cortex"]["total_cooperating_agents"], 8)
        self.assertEqual(deps["decision_tree"], ["scanner"])
        self.assertEqual(deps["parameter_tuner"], ["decision_tree"])

    def test_missing_required_section_raises(self):
        bad_content = """
        [graph]
        targets = ["scanner"]
        dependencies = {"scanner": []}

        [compiler]
        profile_guided_optimization = enabled_strict
        """
        with self.assertRaises(BlueprintParseError) as exc:
            parse_blueprint_content(bad_content)
        self.assertIn("Missing required section", str(exc.exception))

    def test_fallback_reversion_gate(self):
        bad_content = """
        [graph]
        entrypoint = orchestrator
        targets = ["scanner", "decision_tree"]
        dependencies = {"scanner": ["decision_tree"], "decision_tree": ["scanner"]}

        [compiler]
        profile_guided_optimization = enabled_strict

        [cortex]
        consensus_protocol = raft_driven_mutation_lock
        """

        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False, encoding="utf-8") as handle:
            handle.write(bad_content)
            bad_blueprint_path = handle.name

        try:
            context = parse_blueprint(bad_blueprint_path)
            self.assertEqual(context["workspace_status"], "reverted_fallback")
            self.assertIn("Invalid instruction loop", context["fallback_reason"])
            self.assertIn("profile_guided_optimization", context["active_optimizer_flags"])
            self.assertEqual(context["graph"]["workspace_mode"], "fallback_manifest")
        finally:
            if os.path.exists(bad_blueprint_path):
                os.remove(bad_blueprint_path)

    def test_parse_blueprint_returns_build_context(self):
        blueprint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blueprint.aero")
        context = parse_blueprint(blueprint_path)

        self.assertEqual(context["workspace_status"], "stable_active")
        self.assertEqual(context["graph"]["entrypoint"], "orchestrator")
        self.assertEqual(context["compilation_targets"], ["scanner", "decision_tree", "parameter_tuner"])
        self.assertEqual(context["resource_metrics"]["max_memory_mb"], 2048)


if __name__ == "__main__":
    unittest.main()