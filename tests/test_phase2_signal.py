"""Unit tests for Phase 2 Simulated Adaptive Traffic Signal Control (Member 5).

Owner: Member 5 (phase2/m5-signal-paper).
Contract: reports/phase2_plan.md section 2, 5.5, 8.5.
All signal experiments tested here are purely simulated.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.phase2.signal_sim.simulator import (
    FixedSignalPolicy,
    GraphPredictivePolicy,
    IntersectionQueueSimulator,
    LocalPredictivePolicy,
    ReactiveSignalPolicy,
    SignalControlPolicy,
    SignalSimulationConfig,
    generate_synthetic_traffic_scenario,
    run_signal_simulation_benchmark,
)


class TestSignalPolicies(unittest.TestCase):
    """Test policy logic, green time allocation and bounds."""

    def setUp(self) -> None:
        self.config = SignalSimulationConfig(
            min_green_s=15.0,
            max_green_s=60.0,
            default_green_s=35.0,
        )

    def test_fixed_policy_always_returns_default(self) -> None:
        policy = FixedSignalPolicy(self.config)
        for speed in [10.0, 30.0, 65.0]:
            g = policy.compute_green_time(current_queue=20.0, current_speed=speed)
            self.assertEqual(g, 35.0)

    def test_reactive_policy_expands_green_under_congestion(self) -> None:
        policy = ReactiveSignalPolicy(self.config)
        g_free = policy.compute_green_time(current_queue=5.0, current_speed=65.0)
        g_cong = policy.compute_green_time(current_queue=25.0, current_speed=15.0)
        self.assertGreater(g_cong, g_free)
        self.assertLessEqual(g_cong, self.config.max_green_s)
        self.assertGreaterEqual(g_free, self.config.min_green_s)

    def test_predictive_local_uses_forecast(self) -> None:
        policy = LocalPredictivePolicy(self.config)
        # Sensed speed is free flow, but forecast is severe congestion
        g = policy.compute_green_time(
            current_queue=5.0,
            current_speed=65.0,
            forecast_speed_15m=15.0,
        )
        self.assertGreater(g, self.config.default_green_s)

    def test_graph_predictive_responds_to_upstream_shockwave(self) -> None:
        policy = GraphPredictivePolicy(self.config)
        # High queue and low upstream speed
        g = policy.compute_green_time(
            current_queue=35.0,
            current_speed=50.0,
            forecast_speed_15m=30.0,
            forecast_speed_30m=20.0,
            upstream_neighbor_speed=15.0,
        )
        self.assertGreater(g, 45.0)
        self.assertLessEqual(g, self.config.max_green_s)

    def test_strict_clipping_bounds(self) -> None:
        policy = ReactiveSignalPolicy(self.config)
        # Test extreme inputs
        self.assertEqual(policy.clip_green(-100.0), self.config.min_green_s)
        self.assertEqual(policy.clip_green(500.0), self.config.max_green_s)


class TestIntersectionSimulator(unittest.TestCase):
    """Test discrete queue dynamics, vehicle conservation, and delay calculations."""

    def setUp(self) -> None:
        self.config = SignalSimulationConfig(seed=123)
        self.sim = IntersectionQueueSimulator(self.config)

    def test_greenshields_flow_bounds(self) -> None:
        flow_low = self.sim.calculate_arrival_flow(5.0)
        flow_free = self.sim.calculate_arrival_flow(65.0)
        flow_crit = self.sim.calculate_arrival_flow(32.5)

        self.assertGreater(flow_crit, flow_low)
        self.assertGreaterEqual(flow_crit, flow_free)
        self.assertGreaterEqual(flow_low, 400.0)

    def test_vehicle_conservation_in_simulation(self) -> None:
        policy = FixedSignalPolicy(self.config)
        speeds = [45.0] * 20
        res = self.sim.run_simulation(policy, speeds, scenario_name="test_regime")

        self.assertEqual(res.num_cycles, 20)
        self.assertGreater(res.total_vehicles_served, 0)
        self.assertGreaterEqual(res.mean_delay_s, 0.0)
        self.assertGreaterEqual(res.spillback_rate_pct, 0.0)
        self.assertLessEqual(res.spillback_rate_pct, 100.0)

    def test_determinism_with_same_seed(self) -> None:
        cfg1 = SignalSimulationConfig(seed=999)
        cfg2 = SignalSimulationConfig(seed=999)
        sim1 = IntersectionQueueSimulator(cfg1)
        sim2 = IntersectionQueueSimulator(cfg2)
        policy1 = ReactiveSignalPolicy(cfg1)
        policy2 = ReactiveSignalPolicy(cfg2)

        speeds = [55.0, 40.0, 25.0, 20.0, 30.0, 50.0] * 5
        res1 = sim1.run_simulation(policy1, speeds)
        res2 = sim2.run_simulation(policy2, speeds)

        self.assertEqual(res1.mean_delay_s, res2.mean_delay_s)
        self.assertEqual(res1.max_queue_veh, res2.max_queue_veh)
        self.assertEqual(res1.total_vehicles_served, res2.total_vehicles_served)


class TestBenchmarkRunner(unittest.TestCase):
    """Test full scenario generator and benchmark runner."""

    def test_scenario_generation(self) -> None:
        for scen in ["morning_peak", "evening_peak", "shockwave_incident", "off_peak"]:
            obs, f_loc, f15, f30, up = generate_synthetic_traffic_scenario(scen, num_cycles=50, seed=42)
            self.assertEqual(len(obs), 50)
            self.assertEqual(len(f_loc), 50)
            self.assertEqual(len(f15), 50)
            self.assertEqual(len(f30), 50)
            self.assertEqual(len(up), 50)
            self.assertTrue(np.all(np.isfinite(obs)))

    def test_benchmark_runs_end_to_end(self) -> None:
        df, summary = run_signal_simulation_benchmark(
            seeds=(42,),
            scenarios=("off_peak", "shockwave_incident"),
        )
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 8)  # 2 scenarios * 1 seed * 4 policies
        self.assertIn("mean_delay_s", df.columns)
        self.assertIn("spillback_rate_pct", df.columns)
        self.assertIn("scenarios", summary)
        self.assertIn("shockwave_incident", summary["scenarios"])


if __name__ == "__main__":
    unittest.main()
