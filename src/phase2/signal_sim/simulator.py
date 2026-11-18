"""Simulated adaptive traffic signal control (Member 5).

Contract: reports/phase2_plan.md sections 2, 5.5, 8.5.

Everything here is a SOFTWARE SIMULATION of one signalised approach (plus one
conflicting cross approach) on a 90 s cycle. No live or real-world signal is
actuated, and no result here is a field measurement.

Model, per cycle ``t``:

* main-approach arrivals ~ Poisson(q(v_t) * C / 3600), with the Greenshields-shaped
  arrival rate ``q(v) = base_arrival_vph + k_jam * v * (1 - v / v_free)``;
* main discharge capacity ``lanes * s * g / 3600``, reduced while the observed speed is
  below critical (``0.4 + 0.6 * v / v_crit``, a stand-in for downstream congestion);
* the cross approach gets ``C - lost_time - g`` seconds of green and constant demand,
  so extra main green has a real cost;
* delay = Webster uniform term per arriving vehicle + one full cycle for every vehicle
  left in queue at the end of the cycle (overflow);
* spillback = main queue above ``storage_capacity_veh`` at the end of a cycle;
* fuel/CO2 are proxies from idle time and stops only.

Forecasts are SYNTHETIC: the generator returns "true future speed + Gaussian noise",
with the noise sized so its MAE equals a forecast MAE (see ``ForecastErrorSpec``).
That ignores the lag that real models show at abrupt breakdowns, and the neighbour
speed series is the local series shifted earlier by an assumed number of cycles.
Results therefore show what a given forecast *accuracy* is worth under these
assumptions, not what the trained GAT/DCRNN models achieve in the field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

SCENARIOS = ("morning_peak", "evening_peak", "shockwave_incident", "off_peak")
POLICY_NAMES = ("fixed_pretimed", "reactive_actuated", "predictive_local_p1", "predictive_graph_p2")
RESULT_COLUMNS = (
    "policy_name",
    "scenario_name",
    "seed",
    "num_cycles",
    "mean_delay_s",
    "p95_delay_s",
    "max_queue_veh",
    "mean_queue_veh",
    "spillback_rate_pct",
    "total_vehicles_served",
    "throughput_vph",
    "mean_stops_per_veh",
    "fuel_consumption_proxy",
    "co2_emission_proxy_kg",
)


@dataclass
class ForecastErrorSpec:
    """MAE (source speed units) used to size the synthetic forecast noise."""

    local_15m_mae: float = 2.970   # XGBoost 32, test, all: data/metadata/phase1_metrics.csv
    graph_15m_mae: float = 2.762   # placeholder; scripts/phase2_run_signal_sim.py replaces it with
    graph_30m_mae: float = 3.148   # GAT's measured test MAE once metrics_summary.csv exists
    source: str = (
        "local: XGBoost 32 test MAE from data/metadata/phase1_metrics.csv; "
        "graph: placeholder MAE (no measured GAT result supplied)"
    )


@dataclass
class SignalSimulationConfig:
    """Parameters of the simulated intersection. All values are assumptions, not calibrated."""

    seed: int = 42
    cycle_length_s: float = 90.0
    lost_time_s: float = 10.0
    min_green_s: float = 15.0
    max_green_s: float = 60.0
    default_green_s: float = 35.0
    saturation_flow_vphpl: float = 1800.0
    main_lanes: int = 2
    cross_lanes: int = 1
    cross_demand_vph: float = 500.0
    free_flow_speed: float = 65.0
    jam_coefficient: float = 80.0
    base_arrival_vph: float = 400.0
    storage_capacity_veh: float = 60.0
    min_discharge_factor: float = 0.4
    idle_fuel_l_per_h: float = 1.0
    fuel_per_stop_l: float = 0.01
    co2_kg_per_l: float = 2.31
    neighbor_lead_cycles: int = 2
    forecast_errors: ForecastErrorSpec = field(default_factory=ForecastErrorSpec)

    @property
    def critical_speed(self) -> float:
        return self.free_flow_speed / 2.0


def arrival_flow_vph(cfg: SignalSimulationConfig, speed: float) -> float:
    """Main-approach arrival rate (veh/h) at a given speed (Greenshields shape plus a floor)."""
    v = min(max(float(speed), 0.0), cfg.free_flow_speed)
    return cfg.base_arrival_vph + cfg.jam_coefficient * v * (1.0 - v / cfg.free_flow_speed)


def discharge_factor(cfg: SignalSimulationConfig, speed: float) -> float:
    """Share of saturation flow that can discharge; drops below critical speed."""
    ratio = max(float(speed), 0.0) / cfg.critical_speed
    return min(1.0, cfg.min_discharge_factor + (1.0 - cfg.min_discharge_factor) * ratio)


class SignalControlPolicy:
    """Base policy: give the main approach enough green to serve its queue plus expected arrivals.

    For each speed signal ``v`` the policy can see, the green it would need is
    ``(queue + q(v) * C / 3600) / (lanes * s / 3600 * discharge(v))`` with ``q`` and
    ``discharge`` as in the simulator; the policy takes the largest of these (it prepares
    for the worst condition it sees). Subclasses only choose which speed signals they see.
    """

    name = "base"

    def __init__(self, config: SignalSimulationConfig) -> None:
        self.config = config

    def clip_green(self, green_s: float) -> float:
        return float(min(max(green_s, self.config.min_green_s), self.config.max_green_s))

    def speed_signals(
        self,
        current_speed: float,
        forecast_speed_15m: Optional[float],
        forecast_speed_30m: Optional[float],
        upstream_neighbor_speed: Optional[float],
    ) -> list[Optional[float]]:
        return [current_speed]

    def compute_green_time(
        self,
        current_queue: float,
        current_speed: float,
        forecast_speed_15m: Optional[float] = None,
        forecast_speed_30m: Optional[float] = None,
        upstream_neighbor_speed: Optional[float] = None,
    ) -> float:
        cfg = self.config
        signals = self.speed_signals(current_speed, forecast_speed_15m, forecast_speed_30m, upstream_neighbor_speed)
        rate_per_s = cfg.main_lanes * cfg.saturation_flow_vphpl / 3600.0
        needed = 0.0
        for v in signals:
            if v is None or not np.isfinite(v):
                continue
            expected_arrivals = arrival_flow_vph(cfg, v) * cfg.cycle_length_s / 3600.0
            needed = max(needed, (max(current_queue, 0.0) + expected_arrivals) / (rate_per_s * discharge_factor(cfg, v)))
        return self.clip_green(needed)


class FixedSignalPolicy(SignalControlPolicy):
    """Pre-timed: always the default green."""

    name = "fixed_pretimed"

    def compute_green_time(self, current_queue: float, current_speed: float, *args, **kwargs) -> float:
        return self.clip_green(self.config.default_green_s)


class ReactiveSignalPolicy(SignalControlPolicy):
    """Actuated: current queue and current detector speed only."""

    name = "reactive_actuated"


class LocalPredictivePolicy(SignalControlPolicy):
    """Adds the local 15 min forecast (Phase 1 style, sensor's own history)."""

    name = "predictive_local_p1"

    def speed_signals(self, current_speed, forecast_speed_15m, forecast_speed_30m, upstream_neighbor_speed):
        return [current_speed, forecast_speed_15m]


class GraphPredictivePolicy(SignalControlPolicy):
    """Adds the graph 15/30 min forecasts and the neighbouring sensor's current speed."""

    name = "predictive_graph_p2"

    def speed_signals(self, current_speed, forecast_speed_15m, forecast_speed_30m, upstream_neighbor_speed):
        return [current_speed, forecast_speed_15m, forecast_speed_30m, upstream_neighbor_speed]


@dataclass
class SimulationResult:
    policy_name: str
    scenario_name: str
    seed: int
    num_cycles: int
    mean_delay_s: float
    p95_delay_s: float
    max_queue_veh: float
    mean_queue_veh: float
    spillback_rate_pct: float
    total_vehicles_served: int
    throughput_vph: float
    mean_stops_per_veh: float
    fuel_consumption_proxy: float
    co2_emission_proxy_kg: float
    queue_history: list[float] = field(default_factory=list, repr=False)
    green_history: list[float] = field(default_factory=list, repr=False)

    def to_row(self) -> dict:
        row = asdict(self)
        return {k: row[k] for k in RESULT_COLUMNS}


def _webster_uniform(cycle_s: float, green_s: float, x: float) -> tuple[float, float]:
    """Webster's first (uniform) delay term and the matching stop fraction."""
    lam = green_s / cycle_s
    x = min(max(x, 0.0), 1.0)
    denom = max(1.0 - lam * x, 1e-9)
    delay = cycle_s * (1.0 - lam) ** 2 / (2.0 * denom)
    stops = min((1.0 - lam) / denom, 1.0)
    return delay, stops


class IntersectionQueueSimulator:
    """Cycle-by-cycle queue simulation of one main approach and one cross approach."""

    def __init__(self, config: SignalSimulationConfig) -> None:
        self.config = config

    def calculate_arrival_flow(self, speed: float) -> float:
        """Arrival rate (veh/h) on the main approach for a given speed (Greenshields shape)."""
        return arrival_flow_vph(self.config, speed)

    def discharge_factor(self, speed: float) -> float:
        return discharge_factor(self.config, speed)

    def run_simulation(
        self,
        policy: SignalControlPolicy,
        speed_series: Sequence[float],
        forecast_15m_series: Optional[Sequence[float]] = None,
        forecast_30m_series: Optional[Sequence[float]] = None,
        upstream_speed_series: Optional[Sequence[float]] = None,
        scenario_name: str = "custom",
    ) -> SimulationResult:
        cfg = self.config
        speeds = np.asarray(speed_series, dtype=float)
        n = len(speeds)
        C = cfg.cycle_length_s

        # Arrivals are drawn up front from the seed, so every policy sees the same demand.
        rng = np.random.default_rng(cfg.seed)
        main_arrivals = rng.poisson([self.calculate_arrival_flow(v) * C / 3600.0 for v in speeds])
        cross_arrivals = rng.poisson(cfg.cross_demand_vph * C / 3600.0, size=n)

        def at(series: Optional[Sequence[float]], t: int) -> Optional[float]:
            return None if series is None else float(series[t])

        queues = {"main": 0.0, "cross": 0.0}
        total_delay = total_stops = total_arrivals = total_served = 0.0
        spill_cycles = 0
        cycle_delays: list[float] = []
        queue_hist: list[float] = []
        green_hist: list[float] = []

        for t in range(n):
            g = policy.compute_green_time(
                current_queue=queues["main"],
                current_speed=float(speeds[t]),
                forecast_speed_15m=at(forecast_15m_series, t),
                forecast_speed_30m=at(forecast_30m_series, t),
                upstream_neighbor_speed=at(upstream_speed_series, t),
            )
            g_cross = max(C - cfg.lost_time_s - g, 0.0)
            approaches = (
                ("main", main_arrivals[t], g,
                 cfg.main_lanes * cfg.saturation_flow_vphpl * g / 3600.0 * self.discharge_factor(speeds[t])),
                ("cross", cross_arrivals[t], g_cross,
                 cfg.cross_lanes * cfg.saturation_flow_vphpl * g_cross / 3600.0),
            )

            cycle_delay = cycle_stops = cycle_arrivals = 0.0
            for key, arrivals, green, cap in approaches:
                served = min(queues[key] + arrivals, cap)
                queues[key] = queues[key] + arrivals - served
                d_uniform, stop_frac = _webster_uniform(C, green, arrivals / cap if cap > 0 else 1.0)
                cycle_delay += arrivals * d_uniform + queues[key] * C
                cycle_stops += arrivals * stop_frac + queues[key]
                cycle_arrivals += arrivals
                total_served += served

            total_delay += cycle_delay
            total_stops += cycle_stops
            total_arrivals += cycle_arrivals
            cycle_delays.append(cycle_delay / cycle_arrivals if cycle_arrivals > 0 else 0.0)
            queue_hist.append(queues["main"])
            green_hist.append(g)
            if queues["main"] > cfg.storage_capacity_veh:
                spill_cycles += 1

        hours = n * C / 3600.0
        fuel = total_delay / 3600.0 * cfg.idle_fuel_l_per_h + total_stops * cfg.fuel_per_stop_l
        safe_arrivals = max(total_arrivals, 1.0)
        return SimulationResult(
            policy_name=policy.name,
            scenario_name=scenario_name,
            seed=cfg.seed,
            num_cycles=n,
            mean_delay_s=round(total_delay / safe_arrivals, 2),
            p95_delay_s=round(float(np.percentile(cycle_delays, 95)) if n else 0.0, 2),
            max_queue_veh=round(max(queue_hist, default=0.0), 1),
            mean_queue_veh=round(float(np.mean(queue_hist)) if n else 0.0, 1),
            spillback_rate_pct=round(100.0 * spill_cycles / n, 2) if n else 0.0,
            total_vehicles_served=int(round(total_served)),
            throughput_vph=round(total_served / hours, 1) if n else 0.0,
            mean_stops_per_veh=round(total_stops / safe_arrivals, 3),
            fuel_consumption_proxy=round(fuel, 2),
            co2_emission_proxy_kg=round(fuel * cfg.co2_kg_per_l, 2),
            queue_history=queue_hist,
            green_history=green_hist,
        )


def _bump(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-(((x - center) / width) ** 2))


def _shift_earlier(series: np.ndarray, steps: int) -> np.ndarray:
    """``out[t] = series[t + steps]``, holding the last value at the end."""
    if steps <= 0:
        return series.copy()
    if steps >= len(series):
        return np.full_like(series, series[-1])
    return np.concatenate([series[steps:], np.full(steps, series[-1])])


def generate_synthetic_traffic_scenario(
    scenario_type: str,
    num_cycles: int = 120,
    seed: int = 42,
    cycle_length_s: float = 90.0,
    errors: Optional[ForecastErrorSpec] = None,
    neighbor_lead_cycles: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic speed series and synthetic forecasts for one scenario.

    Returns ``(observed, local_15m, graph_15m, graph_30m, neighbor)``, each of length
    ``num_cycles``. Forecasts are the true speed 15/30 min ahead plus Gaussian noise whose
    MAE equals the configured forecast MAE (sigma = MAE * sqrt(pi / 2)).
    """
    if scenario_type not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario_type!r}; expected one of {SCENARIOS}")
    errors = errors or ForecastErrorSpec()
    rng = np.random.default_rng([seed, SCENARIOS.index(scenario_type)])
    n = num_cycles
    x = np.linspace(0.0, 1.0, n)

    if scenario_type == "off_peak":
        base = np.full(n, 60.0)
    elif scenario_type == "morning_peak":
        base = 62.0 - 32.0 * _bump(x, 0.45, 0.20)
    elif scenario_type == "evening_peak":
        base = 62.0 - 38.0 * _bump(x, 0.55, 0.28)
    else:  # shockwave_incident: abrupt breakdown for the middle 35 % of the run
        base = np.full(n, 63.0)
        start, end = int(0.35 * n), int(0.70 * n)
        base[start:end] = 12.0
        ramp_down = min(2, start)
        if ramp_down:
            base[start - ramp_down:start] = np.linspace(63.0, 12.0, ramp_down + 2)[1:-1]
        recovery = min(6, n - end)
        if recovery:
            base[end:end + recovery] = np.linspace(12.0, 63.0, recovery + 2)[1:-1]

    observed = np.clip(base + rng.normal(0.0, 1.5, n), 3.0, 70.0)
    neighbor = np.clip(_shift_earlier(base, neighbor_lead_cycles) + rng.normal(0.0, 1.5, n), 3.0, 70.0)

    steps_15 = int(round(15 * 60 / cycle_length_s))
    steps_30 = int(round(30 * 60 / cycle_length_s))
    to_sigma = math.sqrt(math.pi / 2.0)

    def forecast(steps: int, mae: float) -> np.ndarray:
        return np.clip(_shift_earlier(observed, steps) + rng.normal(0.0, mae * to_sigma, n), 3.0, 70.0)

    local_15 = forecast(steps_15, errors.local_15m_mae)
    graph_15 = forecast(steps_15, errors.graph_15m_mae)
    graph_30 = forecast(steps_30, errors.graph_30m_mae)
    return observed, local_15, graph_15, graph_30, neighbor


def _policy_inputs(policy: SignalControlPolicy, f_local, f_graph_15, f_graph_30, neighbor):
    if policy.name == "predictive_local_p1":
        return f_local, None, None
    if policy.name == "predictive_graph_p2":
        return f_graph_15, f_graph_30, neighbor
    return None, None, None


def run_signal_simulation_benchmark(
    seeds: Iterable[int] = (42, 43, 44, 45, 46),
    scenarios: Iterable[str] = SCENARIOS,
    num_cycles: int = 120,
    config_overrides: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """Run every policy on every scenario and seed; return per-run rows and a summary."""
    seeds, scenarios = list(seeds), list(scenarios)
    overrides = config_overrides or {}
    base_cfg = SignalSimulationConfig(**overrides)
    rows = []
    for scen in scenarios:
        for seed in seeds:
            cfg = SignalSimulationConfig(**{**overrides, "seed": seed})
            obs, f_local, f_g15, f_g30, neighbor = generate_synthetic_traffic_scenario(
                scen,
                num_cycles=num_cycles,
                seed=seed,
                cycle_length_s=cfg.cycle_length_s,
                errors=cfg.forecast_errors,
                neighbor_lead_cycles=cfg.neighbor_lead_cycles,
            )
            sim = IntersectionQueueSimulator(cfg)
            for policy_cls in (FixedSignalPolicy, ReactiveSignalPolicy, LocalPredictivePolicy, GraphPredictivePolicy):
                policy = policy_cls(cfg)
                f15, f30, up = _policy_inputs(policy, f_local, f_g15, f_g30, neighbor)
                rows.append(sim.run_simulation(policy, obs, f15, f30, up, scenario_name=scen).to_row())

    df = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
    summary: dict = {
        "metadata": {
            "evaluation_type": "simulated_signal_control",
            "disclaimer": (
                "Software simulation of one signalised approach with synthetic speeds and synthetic "
                "forecasts; not a field result and not driven by trained model predictions."
            ),
            "forecast_source": "synthetic: true future speed + Gaussian noise sized to a forecast MAE",
            "forecast_mae": asdict(base_cfg.forecast_errors),
            "neighbor_lead_cycles_assumed": base_cfg.neighbor_lead_cycles,
            "cycle_length_s": base_cfg.cycle_length_s,
            "num_cycles": num_cycles,
            "num_scenarios": len(scenarios),
            "num_seeds": len(seeds),
            "total_runs": len(df),
        },
        "scenarios": {},
    }
    for scen in scenarios:
        summary["scenarios"][scen] = {}
        for pol in POLICY_NAMES:
            sub = df[(df["scenario_name"] == scen) & (df["policy_name"] == pol)]
            if sub.empty:
                continue
            summary["scenarios"][scen][pol] = {
                "mean_delay_s": round(float(sub["mean_delay_s"].mean()), 2),
                "std_delay_s": round(float(sub["mean_delay_s"].std(ddof=1)) if len(sub) > 1 else 0.0, 2),
                "spillback_rate_pct": round(float(sub["spillback_rate_pct"].mean()), 2),
                "throughput_vph": round(float(sub["throughput_vph"].mean()), 1),
                "fuel_consumption_proxy": round(float(sub["fuel_consumption_proxy"].mean()), 2),
            }
    return df, summary
