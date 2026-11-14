"""Simulated Adaptive Traffic Signal Control (Member 5).

Contract: reports/phase2_plan.md section 2, 5.5, 8.5.
All signal experiments in this module are purely simulated.
No claim of live or real-world traffic actuation is made.
"""

from src.phase2.signal_sim.simulator import (
    SignalSimulationConfig,
    SignalControlPolicy,
    FixedSignalPolicy,
    ReactiveSignalPolicy,
    LocalPredictivePolicy,
    GraphPredictivePolicy,
    IntersectionQueueSimulator,
    SimulationResult,
    run_signal_simulation_benchmark,
)

__all__ = [
    "SignalSimulationConfig",
    "SignalControlPolicy",
    "FixedSignalPolicy",
    "ReactiveSignalPolicy",
    "LocalPredictivePolicy",
    "GraphPredictivePolicy",
    "IntersectionQueueSimulator",
    "SimulationResult",
    "run_signal_simulation_benchmark",
]

