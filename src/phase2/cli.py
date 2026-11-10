"""Command line for Phase 2 runs (Member 1).

    python -m src.phase2.cli <cmd> --runs configs/phase2/runs_<name>.yaml
                             [--only model_id[:seed] ...] [--device auto|cpu|mps|cuda]
                             [--subset 0.05] [--out DIR]

====== ==============================================================================
cmd    does
====== ==============================================================================
search trains each candidate of every group (seed 42) on train, early-stops on
       validation, writes ``<out>/search/<name>.csv``. No test predictions.
pick   per group, selects the candidate with the lowest mean validation MAE and writes
       ``<out>/final/<name>.yaml`` with concrete params.
final  trains every final run and seed; writes validation and test predictions,
       ``seed<S>_meta.json``, best checkpoint (and attention if the model has it).
all    search, pick, final in order; skips finished work and resumes interrupted work.
       If ``<out>/final/<name>.yaml`` already exists for this run file, search and
       pick are skipped (so helpers can start extra seeds with ``--only``).
smoke  2 training steps and a prediction pass for every model in the file on a tiny
       subset (default 0.01); checks shape, finiteness and gradients. Writes nothing.
====== ==============================================================================

``<out>`` is ``--out``, else ``$PHASE2_OUT``, else ``data/processed/phase2``. Runs with
``--subset < 1`` go to ``<out>/subset_<frac>`` and never mix with real results.

Run file schema: see reports/phase2_plan.md section 4.6. ``--runs`` may also point at a
file written by ``pick`` (key ``runs``), e.g. ``configs/phase2/final/gat.yaml``; then
only ``final`` / ``smoke`` apply.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import yaml

from src.phase2.common import (
    HORIZON_NAMES,
    ROOT,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    config_hash,
    output_root,
    resolve_device,
    set_seed,
)
from src.phase2.data import Phase2Data, load_data
from src.phase2.graph import VARIANTS, load_graph
from src.phase2.registry import build_model, get_builder
from src.phase2.train import (
    TrainConfig,
    count_parameters,
    export_attention,
    export_meta,
    export_predictions,
    fit,
    model_cfg,
    predict,
    smoke_check,
)

SEARCH_SEED = 42
FROM_SEARCH = "from_search"


def log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- run files
def load_runs(path: str | Path) -> dict[str, Any]:
    """Read and validate a run file. Returns the parsed dict with ``kind`` = ``spec`` or ``final``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"run file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    runs = copy.deepcopy(raw)
    runs.setdefault("name", path.stem.removeprefix("runs_"))
    runs["train"] = dict(runs.get("train") or {})
    TrainConfig.from_dict(runs["train"])  # validates keys early
    runs["source_hash"] = config_hash(raw)

    if "groups" in raw:
        runs["kind"] = "spec"
        if not isinstance(raw["groups"], list) or not raw["groups"]:
            raise ValueError(f"{path}: 'groups' must be a non-empty list")
        for gi, group in enumerate(runs["groups"]):
            if "final" not in group or not group["final"]:
                raise ValueError(f"{path}: group {gi} needs a non-empty 'final' list")
            search = group.get("search")
            if search is not None:
                if not search.get("candidates"):
                    raise ValueError(f"{path}: group {gi} search needs 'candidates'")
                search.setdefault("graph", "bidirectional")
                _check_names(search["model"], search["graph"], f"group {gi} search")
            for run in group["final"]:
                if run.get("params") == FROM_SEARCH and search is None:
                    raise ValueError(f"{path}: run {run.get('model_id')!r} uses from_search but group {gi} has no search")
        entries = [run for group in runs["groups"] for run in group["final"]]
    elif "runs" in raw:
        runs["kind"] = "final"
        entries = runs["runs"]
    else:
        raise ValueError(f"{path}: expected a 'groups' (spec) or 'runs' (picked) key")

    seen: set[str] = set()
    for run in entries:
        for key in ("model_id", "model", "graph", "seeds"):
            if key not in run:
                raise ValueError(f"{path}: every final run needs '{key}': {run}")
        if run["model_id"] in seen:
            raise ValueError(f"{path}: duplicate model_id {run['model_id']!r}")
        seen.add(run["model_id"])
        _check_names(run["model"], run["graph"], run["model_id"])
        run.setdefault("graph_seed", 0)
        run.setdefault("graph_base", "bidirectional")
        run["seeds"] = [int(s) for s in run["seeds"]]
        if run.get("params") not in (FROM_SEARCH, None) and not isinstance(run["params"], dict):
            raise ValueError(f"{path}: params of {run['model_id']!r} must be a mapping or 'from_search'")
    return runs


def _check_names(model: str, graph: str, where: str) -> None:
    get_builder(model)  # raises KeyError listing registered models
    if graph not in VARIANTS:
        raise ValueError(f"{where}: unknown graph variant {graph!r}; choose from {VARIANTS}")


def merged_train(base: dict[str, Any], *overrides: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    for extra in overrides:
        out.update(extra or {})
    TrainConfig.from_dict(out)
    return out


def parse_only(values: list[str] | None) -> list[tuple[str, int | None]]:
    parsed: list[tuple[str, int | None]] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            name, _, seed = item.partition(":")
            parsed.append((name, int(seed) if seed else None))
    return parsed


def wanted(model_id: str, seed: int, only: list[tuple[str, int | None]]) -> bool:
    return not only or any(name == model_id and (s is None or s == seed) for name, s in only)


# --------------------------------------------------------------------------- context
class Context:
    """Process-wide state: device, subset, output folder, lazily loaded data and graphs."""

    def __init__(self, device: str, subset: float, out: str | None) -> None:
        self.device = resolve_device(device)
        self.subset = subset
        self.out = output_root(out, subset)
        self._data: Phase2Data | None = None
        self._graphs: dict[tuple, torch.Tensor] = {}

    @property
    def data(self) -> Phase2Data:
        if self._data is None:
            t0 = time.perf_counter()
            self._data = load_data({"device": str(self.device), "subset": self.subset})
            sizes = {k: len(v) for k, v in self._data.origins.items()}
            log(f"data ready on {self.device} in {time.perf_counter() - t0:.1f}s; origins {sizes}")
        return self._data

    def graph(self, variant: str, seed: int = 0, base: str = "bidirectional") -> torch.Tensor:
        key = (variant, seed, base)
        if key not in self._graphs:
            adj = load_graph(variant, seed=seed, base=base)
            if adj.shape[0] != self.data.num_nodes:
                raise ValueError(f"graph has {adj.shape[0]} nodes but data has {self.data.num_nodes}")
            self._graphs[key] = adj
        return self._graphs[key]


def train_one(
    ctx: Context,
    model_name: str,
    graph: str,
    graph_seed: int,
    graph_base: str,
    params: dict[str, Any],
    train: dict[str, Any],
    seed: int,
    run_dir: Path,
    label: str,
):
    """Seed, build, fit one model. Returns ``(model, adj, fit_result)`` with best weights loaded."""
    set_seed(seed)
    adj = ctx.graph(graph, graph_seed, graph_base)
    model = build_model(model_name, model_cfg(ctx.data, params))
    result = fit(model, ctx.data.datasets(), adj, train, ctx.device, run_dir, seed, label=label, log=log)
    return model, adj, result


# --------------------------------------------------------------------------- search / pick
def _candidate_key(group: dict[str, Any], base_train: dict[str, Any], ci: int) -> tuple[dict[str, Any], str]:
    search = group["search"]
    train = merged_train(base_train, search.get("train"), {"max_minutes": search["max_minutes"]} if "max_minutes" in search else None)
    spec = {
        "model": search["model"],
        "graph": search["graph"],
        "graph_seed": search.get("graph_seed", 0),
        "graph_base": search.get("graph_base", "bidirectional"),
        "params": search["candidates"][ci],
        "train": train,
        "seed": search.get("seed", SEARCH_SEED),
    }
    return spec, config_hash(spec)


def search_dir(ctx: Context, name: str, gi: int, ci: int, key: str) -> Path:
    return ctx.out / "search" / name / f"g{gi}_c{ci}_{key[:8]}"


def run_search(runs: dict[str, Any], ctx: Context) -> None:
    if runs["kind"] != "spec":
        raise SystemExit("search/pick need a run file with 'groups' (not a picked 'runs' file)")
    name = runs["name"]
    rows: list[dict[str, Any]] = []
    for gi, group in enumerate(runs["groups"]):
        if "search" not in group:
            continue
        for ci in range(len(group["search"]["candidates"])):
            spec, key = _candidate_key(group, runs["train"], ci)
            run_dir = search_dir(ctx, name, gi, ci, key)
            label = f"search {name} g{gi} c{ci}"
            log(f"== {label}: {json.dumps(spec['params'])}")
            _, _, result = train_one(
                ctx, spec["model"], spec["graph"], spec["graph_seed"], spec["graph_base"],
                spec["params"], spec["train"], spec["seed"], run_dir, label,
            )
            rows.append(_search_row(gi, ci, spec, result))
    write_search_csv(ctx, name, rows)


def _search_row(gi: int, ci: int, spec: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "group": gi,
        "candidate": ci,
        "model": spec["model"],
        "graph": spec["graph"],
        "params": json.dumps(spec["params"], sort_keys=True),
    }
    for h in HORIZON_NAMES:
        row[f"val_mae_{h}"] = round(result["best_val_mae_by_horizon"][h], 6)
    row.update(
        val_mae_mean=round(result["best_val_mae"], 6),
        epochs_run=result["epochs_run"],
        best_epoch=result["best_epoch"],
        stop_reason=result["stop_reason"],
        train_seconds=round(result["train_seconds"], 1),
        param_count=result["params"],
    )
    return row


def write_search_csv(ctx: Context, name: str, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    path = ctx.out / "search" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, buffer.getvalue().encode())
    log(f"search results -> {path}")
    return path


def completed_search_rows(runs: dict[str, Any], ctx: Context) -> list[dict[str, Any]]:
    """Rows rebuilt from ``done.json`` files; raises if any candidate is unfinished."""
    rows, missing = [], []
    for gi, group in enumerate(runs["groups"]):
        if "search" not in group:
            continue
        for ci in range(len(group["search"]["candidates"])):
            spec, key = _candidate_key(group, runs["train"], ci)
            done = search_dir(ctx, runs["name"], gi, ci, key) / "done.json"
            if not done.is_file():
                missing.append(f"group {gi} candidate {ci}")
                continue
            rows.append(_search_row(gi, ci, spec, json.loads(done.read_text())))
    if missing:
        raise SystemExit(f"search is not finished for: {', '.join(missing)}. Run `search` (or `all`) first.")
    return rows


def run_pick(runs: dict[str, Any], ctx: Context) -> Path:
    if runs["kind"] != "spec":
        raise SystemExit("search/pick need a run file with 'groups' (not a picked 'runs' file)")
    rows = completed_search_rows(runs, ctx)
    write_search_csv(ctx, runs["name"], rows)
    picked, final_runs = [], []
    for gi, group in enumerate(runs["groups"]):
        best = None
        if "search" in group:
            candidates = [r for r in rows if r["group"] == gi]
            best = min(candidates, key=lambda r: (r["val_mae_mean"], r["candidate"]))
            picked.append(
                {
                    "group": gi,
                    "candidate": best["candidate"],
                    "params": json.loads(best["params"]),
                    "val_mae_mean": best["val_mae_mean"],
                }
            )
        for run in group["final"]:
            concrete = copy.deepcopy(run)
            if concrete.get("params") == FROM_SEARCH:
                concrete["params"] = json.loads(best["params"])
            concrete["params"] = dict(concrete.get("params") or {})
            concrete["train"] = merged_train(runs["train"], run.get("train"))
            concrete["group"] = gi
            final_runs.append(concrete)
    out = {
        "name": runs["name"],
        "source_hash": runs["source_hash"],
        "picked": picked,
        "runs": final_runs,
    }
    path = ctx.out / "final" / f"{runs['name']}.yaml"
    atomic_write_bytes(path, yaml.safe_dump(out, sort_keys=False).encode())
    log(f"picked configuration -> {path}")
    for p in picked:
        log(f"   group {p['group']}: candidate {p['candidate']} (val MAE {p['val_mae_mean']}) {p['params']}")
    return path


# --------------------------------------------------------------------------- final
def final_run_spec(run: dict[str, Any], base_train: dict[str, Any], seed: int, subset: float) -> dict[str, Any]:
    return {
        "model": run["model"],
        "graph": run["graph"],
        "graph_seed": run.get("graph_seed", 0),
        "graph_base": run.get("graph_base", "bidirectional"),
        "params": dict(run.get("params") or {}),
        "train": merged_train(base_train, run.get("train")),
        "seed": seed,
        "subset": subset,
    }


def run_final(runs: dict[str, Any], ctx: Context, only: list[tuple[str, int | None]]) -> None:
    if runs["kind"] != "final":
        raise SystemExit("internal error: final needs a picked run file")
    ctx_data = ctx.data
    datasets = ctx_data.datasets()
    done_count = 0
    for run in runs["runs"]:
        for seed in run["seeds"]:
            if not wanted(run["model_id"], seed, only):
                continue
            model_id = run["model_id"]
            spec = final_run_spec(run, runs["train"], seed, ctx.subset)
            key = config_hash(spec)
            meta_path = ctx.out / "predictions" / model_id / f"seed{seed}_meta.json"
            val_path = ctx.out / "predictions" / model_id / f"seed{seed}_val.npz"
            test_path = ctx.out / "predictions" / model_id / f"seed{seed}_test.npz"
            if meta_path.is_file() and val_path.is_file() and test_path.is_file():
                if json.loads(meta_path.read_text()).get("config_hash") == key:
                    log(f"== final {model_id} seed {seed}: already complete, skipping")
                    continue
                log(f"== final {model_id} seed {seed}: config changed since the last run; retraining")
            label = f"final {model_id} s{seed}"
            log(f"== {label}")
            run_dir = ctx.out / "checkpoints" / model_id / f"seed{seed}_{key[:8]}"
            model, adj, result = train_one(
                ctx, spec["model"], spec["graph"], spec["graph_seed"], spec["graph_base"],
                spec["params"], spec["train"], seed, run_dir, label,
            )
            export_predictions(predict(model, datasets["val"], adj, ctx.device, label=label), datasets["val"], model_id, seed, "val", ctx.out)
            export_predictions(predict(model, datasets["test"], adj, ctx.device, label=label), datasets["test"], model_id, seed, "test", ctx.out)
            attention = export_attention(model, datasets["test"], adj, ctx.device, model_id, seed, ctx.out)
            if attention is not None:
                log(f"   attention -> {attention}")
            atomic_torch_save(
                {"state_dict": model.state_dict(), "config": spec, "scaler": vars(ctx_data.scaler)},
                ctx.out / "best" / f"{model_id}_seed{seed}.pt",
            )
            export_meta(ctx.out, model_id, seed, result, spec, ctx.device, ctx.subset)
            done_count += 1
    log(f"final stage finished; trained {done_count} run(s)")


def load_final_file(runs: dict[str, Any], ctx: Context) -> dict[str, Any]:
    path = ctx.out / "final" / f"{runs['name']}.yaml"
    if not path.is_file():
        raise SystemExit(f"{path} not found. Run `all` (or `search` then `pick`) first, without --only.")
    return load_runs(path)


def run_all(runs: dict[str, Any], ctx: Context, only: list[tuple[str, int | None]]) -> None:
    if runs["kind"] == "final":
        return run_final(runs, ctx, only)
    final_path = ctx.out / "final" / f"{runs['name']}.yaml"
    existing = yaml.safe_load(final_path.read_text()) if final_path.is_file() else None
    up_to_date = existing is not None and existing.get("source_hash") == runs["source_hash"]
    if only and existing is None:
        raise SystemExit("--only needs a finished search+pick first: run `all` once without --only.")
    if up_to_date or (only and existing is not None):
        if not up_to_date:
            log("warning: run file changed since pick; using the existing picked configuration because --only was given")
        log(f"using existing picked configuration {final_path}")
    else:
        run_search(runs, ctx)
        run_pick(runs, ctx)
    run_final(load_final_file(runs, ctx), ctx, only)


# --------------------------------------------------------------------------- smoke
def smoke_targets(runs: dict[str, Any]) -> list[dict[str, Any]]:
    """Distinct (model, graph, params) combinations that appear in a run file."""
    items: dict[str, dict[str, Any]] = {}

    def add(label, model, graph, gseed, gbase, params):
        spec = {"model": model, "graph": graph, "graph_seed": gseed, "graph_base": gbase, "params": params}
        items.setdefault(config_hash(spec), {**spec, "label": label})

    if runs["kind"] == "spec":
        for gi, group in enumerate(runs["groups"]):
            search = group.get("search")
            if search:
                for ci, params in enumerate(search["candidates"]):
                    add(f"{search['model']}[g{gi} c{ci}] graph={search['graph']}", search["model"], search["graph"], 0, "bidirectional", params)
            for run in group["final"]:
                if run.get("params") == FROM_SEARCH:
                    for ci, params in enumerate(search["candidates"]):
                        add(f"{run['model_id']}[c{ci}]", run["model"], run["graph"], run["graph_seed"], run["graph_base"], params)
                else:
                    add(run["model_id"], run["model"], run["graph"], run["graph_seed"], run["graph_base"], dict(run.get("params") or {}))
    else:
        for run in runs["runs"]:
            add(run["model_id"], run["model"], run["graph"], run["graph_seed"], run["graph_base"], dict(run.get("params") or {}))
    return list(items.values())


def run_smoke(runs: dict[str, Any], ctx: Context) -> int:
    datasets = ctx.data.datasets()
    failures = 0
    log(f"{'model':<46}{'params':>10}{'s/step':>9}  status")
    for target in smoke_targets(runs):
        set_seed(0)
        try:
            adj = ctx.graph(target["graph"], target["graph_seed"], target["graph_base"])
            model = build_model(target["model"], model_cfg(ctx.data, target["params"]))
            info = smoke_check(model, datasets, adj, ctx.device, label=target["label"])
        except Exception as exc:  # noqa: BLE001
            info = {"params": 0, "step_seconds": float("nan"), "ok": False, "issues": [f"{type(exc).__name__}: {exc}"]}
        status = "PASS" if info["ok"] else "FAIL: " + "; ".join(info["issues"])
        failures += 0 if info["ok"] else 1
        log(f"{target['label']:<46}{info['params']:>10}{info.get('step_seconds', float('nan')):>9.3f}  {status}")
    log("SMOKE OK" if not failures else f"SMOKE FAILED ({failures} model(s))")
    return 1 if failures else 0


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.phase2.cli", description=__doc__.split("\n\n")[0])
    parser.add_argument("cmd", choices=["search", "pick", "final", "all", "smoke"])
    parser.add_argument("--runs", required=True, help="run file, e.g. configs/phase2/runs_gat.yaml")
    parser.add_argument("--only", action="append", help="model_id or model_id:seed (comma separated or repeated); final stage only")
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--subset", type=float, default=None, help="fraction of origins per split (smoke/dev runs)")
    parser.add_argument("--out", default=None, help="output folder (default $PHASE2_OUT or data/processed/phase2)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_path = Path(args.runs)
    if not runs_path.is_absolute() and not runs_path.exists():
        runs_path = ROOT / runs_path
    subset = args.subset if args.subset is not None else (0.01 if args.cmd == "smoke" else 1.0)
    runs = load_runs(runs_path)
    ctx = Context(args.device, subset, args.out)
    only = parse_only(args.only)
    log(f"run file {runs_path.name} ({runs['kind']}), cmd={args.cmd}, device={ctx.device}, subset={subset}, out={ctx.out}")
    if args.cmd == "smoke":
        return run_smoke(runs, ctx)
    if args.cmd == "search":
        if only:
            log("note: --only is ignored by `search`; groups are searched as a whole")
        run_search(runs, ctx)
    elif args.cmd == "pick":
        run_pick(runs, ctx)
    elif args.cmd == "final":
        run_final(runs if runs["kind"] == "final" else load_final_file(runs, ctx), ctx, only)
    else:
        run_all(runs, ctx, only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
