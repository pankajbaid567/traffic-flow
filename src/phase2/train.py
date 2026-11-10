"""Trainer, predictor and exporters for Phase 2 (Member 1).

Contract (reports/phase2_plan.md sections 4.3, 4.6, 4.7, 4.9):

* a model maps ``x [B,12,N,7]`` and ``adj [N,N]`` to ``[B,4,N]`` **z-scored** speed;
  the trainer inverts the scaler and minimises masked MAE in source speed units
  on valid targets only;
* early stopping and ``pick`` use the validation MAE averaged over the four horizons
  (each horizon pooled over all valid targets first);
* every epoch is checkpointed atomically (works with Colab/Drive disconnects) and a
  re-run resumes automatically; a wall-clock cap (``max_minutes``, cumulative over
  resumes) stops a run and keeps the best validation checkpoint;
* fp32 only. Only validation metrics are computed here. Test predictions are written by
  the CLI ``final`` stage and scored later by the evaluation harness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import io
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
import torch

from src.phase2.common import (
    HISTORY_STEPS,
    HORIZON_NAMES,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    config_hash,
    git_commit,
    set_seed,
    sync_device,
)
from src.phase2.data import StdScaler, WindowDataset


class InterruptedForTest(RuntimeError):
    """Raised by the ``interrupt_after`` test hook to simulate a disconnect."""


# --------------------------------------------------------------------------- config
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 60
    patience: int = 8
    grad_clip: float = 5.0
    max_minutes: float | None = None
    scheduler: str = "plateau"  # plateau | cosine | none
    optimizer: str = "adam"  # adam | adamw
    lr_factor: float = 0.5
    lr_patience: int = 3
    min_delta: float = 0.0
    eval_batch_size: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TrainConfig":
        raw = dict(raw or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown train config key(s) {unknown}; allowed: {sorted(known)}")
        cfg = cls(**raw)
        if cfg.scheduler not in {"plateau", "cosine", "none"}:
            raise ValueError(f"scheduler must be plateau|cosine|none, got {cfg.scheduler!r}")
        if cfg.optimizer not in {"adam", "adamw"}:
            raise ValueError(f"optimizer must be adam|adamw, got {cfg.optimizer!r}")
        if cfg.batch_size < 1 or cfg.epochs < 1 or cfg.patience < 1:
            raise ValueError("batch_size, epochs and patience must be >= 1")
        return cfg


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_cfg(dataset_or_data, params: dict[str, Any] | None) -> dict[str, Any]:
    """The ``cfg`` dict handed to a model builder."""
    data = getattr(dataset_or_data, "data", dataset_or_data)
    base = {
        "num_nodes": int(data.num_nodes),
        "in_channels": int(data.features.shape[-1]),
        "history": HISTORY_STEPS,
        "num_horizons": len(HORIZON_NAMES),
    }
    base.update(params or {})
    return base


# --------------------------------------------------------------------------- core math
def _check_output(out: torch.Tensor, batch: dict[str, torch.Tensor], label: str) -> None:
    expected = (batch["x"].shape[0], 4, batch["x"].shape[2])
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != expected:
        shape = tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
        raise ValueError(
            f"model {label!r} returned {shape} but the contract is [B, 4, N] = {expected} "
            "(z-scored speed for the 5/15/30/60 minute horizons)"
        )


def masked_mae_sums(pred_raw: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-horizon ``(sum |error|, valid count)`` shaped ``[4]`` each."""
    valid = batch["y_valid"].to(pred_raw.dtype)
    err = (pred_raw - batch["y"]).abs() * valid
    return err.sum(dim=(0, 2)), valid.sum(dim=(0, 2))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: WindowDataset,
    adj: torch.Tensor,
    batch_size: int = 64,
    label: str = "model",
) -> list[float]:
    """Pooled masked MAE per horizon in source speed units."""
    was_training = model.training
    model.eval()
    err_sum = torch.zeros(4, dtype=torch.float64)
    count = torch.zeros(4, dtype=torch.float64)
    for batch in dataset.batches(batch_size):
        out = model(batch["x"], adj)
        _check_output(out, batch, label)
        e, c = masked_mae_sums(dataset.scaler.inverse(out), batch)
        err_sum += e.cpu().double()  # cpu() first: MPS has no float64
        count += c.cpu().double()
    model.train(was_training)
    if (count == 0).any():
        raise ValueError(f"no valid targets for horizon(s) {np.flatnonzero(count.numpy() == 0).tolist()}")
    return (err_sum / count).tolist()


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataset: WindowDataset,
    adj: torch.Tensor,
    device: torch.device | str | None = None,
    batch_size: int = 64,
    label: str = "model",
) -> np.ndarray:
    """Forecasts in source speed units, ``float32 [n_origins, 4, N]``, in dataset order."""
    device = torch.device(device) if device is not None else dataset.device
    model = model.to(device)
    adj = adj.to(device=device, dtype=torch.float32)
    was_training = model.training
    model.eval()
    chunks = []
    for batch in dataset.batches(batch_size):
        out = model(batch["x"], adj)
        _check_output(out, batch, label)
        chunks.append(dataset.scaler.inverse(out).float().cpu().numpy())
    model.train(was_training)
    pred = np.concatenate(chunks) if chunks else np.zeros((0, 4, dataset.num_nodes), np.float32)
    bad = ~np.isfinite(pred)
    if bad.any():
        raise FloatingPointError(
            f"model {label!r} produced {int(bad.sum())} non-finite prediction(s) on split {dataset.split!r}"
        )
    return pred.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- fit
def _rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        try:
            state["mps"] = torch.mps.get_rng_state()
        except Exception:  # noqa: BLE001 - older torch
            pass
    return state


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and device.type == "mps":
        try:
            torch.mps.set_rng_state(state["mps"])
        except Exception:  # noqa: BLE001
            pass


def _make_optimizer(params, cfg: TrainConfig):
    cls = torch.optim.Adam if cfg.optimizer == "adam" else torch.optim.AdamW
    return cls(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _make_scheduler(optimizer, cfg: TrainConfig):
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience
        )
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    return None


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def fit(
    model: torch.nn.Module,
    datasets: dict[str, WindowDataset],
    adj: torch.Tensor,
    train_cfg: TrainConfig | dict[str, Any],
    device: torch.device | str,
    run_dir: str | Path,
    seed: int,
    label: str = "model",
    resume: bool = True,
    interrupt_after: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train ``model`` on ``datasets['train']``, early-stopping on ``datasets['val']``.

    Call ``set_seed(seed)`` *before* building ``model`` (the CLI does). Writes in ``run_dir``:
    ``last.pt`` (resume state, every epoch), ``best.pt`` (best validation weights),
    ``epochs.jsonl`` (log) and ``done.json`` (final summary). A finished run is not retrained.
    ``interrupt_after`` is a test hook that raises after that many epochs are checkpointed.
    """
    cfg = train_cfg if isinstance(train_cfg, TrainConfig) else TrainConfig.from_dict(train_cfg)
    device = torch.device(device)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path, best_path, last_path, log_path = (
        run_dir / "done.json",
        run_dir / "best.pt",
        run_dir / "last.pt",
        run_dir / "epochs.jsonl",
    )
    model = model.to(device)
    adj = adj.to(device=device, dtype=torch.float32)
    n_params = count_parameters(model)
    eval_bs = cfg.eval_batch_size or cfg.batch_size
    train_ds, val_ds = datasets["train"], datasets["val"]

    if resume and done_path.is_file() and best_path.is_file():
        model.load_state_dict(_load_checkpoint(best_path)["state_dict"])
        log(f"[{label}] already finished, loaded best checkpoint")
        return json.loads(done_path.read_text())

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:  # e.g. last_value: evaluate once, nothing to train
        t0 = time.perf_counter()
        per_h = evaluate(model, val_ds, adj, eval_bs, label)
        result = _result(per_h, 0, 0, "no_trainable_params", time.perf_counter() - t0, 0.0, n_params)
        atomic_torch_save({"state_dict": model.state_dict(), "epoch": 0, "val_mae": result["best_val_mae"]}, best_path)
        atomic_write_json(done_path, result)
        log(f"[{label}] no trainable parameters; validation MAE {result['best_val_mae']:.4f}")
        return result

    optimizer = _make_optimizer(trainable, cfg)
    scheduler = _make_scheduler(optimizer, cfg)

    state: dict[str, Any] = {
        "epoch": 0,  # epochs completed
        "best": float("inf"),
        "best_epoch": 0,
        "best_by_h": None,
        "bad": 0,
        "elapsed": 0.0,
        "history": [],
        "stop_reason": None,
    }
    if resume and last_path.is_file():
        ckpt = _load_checkpoint(last_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        state.update(ckpt["state"])
        _restore_rng(ckpt["rng"], device)
        atomic_write_bytes(log_path, "".join(json.dumps(h) + "\n" for h in state["history"]).encode())
        log(f"[{label}] resumed after epoch {state['epoch']} ({state['elapsed']/60:.1f} min used)")

    cap_seconds = None if cfg.max_minutes is None else float(cfg.max_minutes) * 60.0
    first_batch_checked = False

    while state["stop_reason"] is None:
        epoch = state["epoch"]
        if epoch >= cfg.epochs:
            state["stop_reason"] = "max_epochs"
            break
        if cap_seconds is not None and epoch > 0 and state["elapsed"] >= cap_seconds:
            state["stop_reason"] = "time_cap"
            break

        model.train()
        sync_device(device)
        t_start = time.perf_counter()
        loss_acc = torch.zeros((), device=device)
        n_batches = 0
        time_capped = False
        for batch in train_ds.batches(cfg.batch_size, shuffle=True, seed=seed, epoch=epoch):
            out = model(batch["x"], adj)
            if not first_batch_checked:
                _check_output(out, batch, label)
            err, cnt = masked_mae_sums(train_ds.scaler.inverse(out), batch)
            loss = err.sum() / cnt.sum().clamp_min(1.0)
            if not first_batch_checked:
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"[{label}] non-finite loss on the first batch; check inputs or lower lr")
                first_batch_checked = True
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            loss_acc += loss.detach()
            n_batches += 1
            if cap_seconds is not None and state["elapsed"] + (time.perf_counter() - t_start) >= cap_seconds:
                time_capped = True
                break
        sync_device(device)
        train_loss = float(loss_acc.item() / max(n_batches, 1))
        if not np.isfinite(train_loss):
            raise FloatingPointError(
                f"[{label}] training loss became non-finite in epoch {epoch + 1}; lower the learning rate or grad_clip"
            )

        per_h = evaluate(model, val_ds, adj, eval_bs, label)
        val_mean = float(np.mean(per_h))
        if scheduler is not None:
            scheduler.step(val_mean) if cfg.scheduler == "plateau" else scheduler.step()
        epoch_seconds = time.perf_counter() - t_start
        state["elapsed"] += epoch_seconds
        state["epoch"] = epoch + 1

        improved = val_mean < state["best"] - cfg.min_delta
        if improved:
            state.update(best=val_mean, best_epoch=state["epoch"], best_by_h=per_h, bad=0)
            atomic_torch_save({"state_dict": model.state_dict(), "epoch": state["epoch"], "val_mae": val_mean}, best_path)
        else:
            state["bad"] += 1

        if time_capped:
            state["stop_reason"] = "time_cap"
        elif state["bad"] >= cfg.patience:
            state["stop_reason"] = "early_stop"
        elif state["epoch"] >= cfg.epochs:
            state["stop_reason"] = "max_epochs"
        elif cap_seconds is not None and state["elapsed"] >= cap_seconds:
            state["stop_reason"] = "time_cap"

        record = {
            "epoch": state["epoch"],
            "train_loss": round(train_loss, 6),
            "val_mae_mean": round(val_mean, 6),
            "val_mae_by_horizon": {h: round(v, 6) for h, v in zip(HORIZON_NAMES, per_h)},
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(epoch_seconds, 2),
            "elapsed_seconds": round(state["elapsed"], 2),
            "best_val_mae": round(state["best"], 6),
            "improved": bool(improved),
            "batches": n_batches,
        }
        state["history"].append(record)
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "state": state,
                "rng": _rng_state(device),
                "config_hash": config_hash(asdict(cfg)),
            },
            last_path,
        )
        atomic_write_bytes(log_path, "".join(json.dumps(h) + "\n" for h in state["history"]).encode())
        mean_epoch = state["elapsed"] / state["epoch"]
        left = max(cfg.epochs - state["epoch"], 0)
        if cap_seconds is not None:
            left = min(left, int(max(cap_seconds - state["elapsed"], 0) // max(mean_epoch, 1e-9)))
        log(
            f"[{label}] epoch {state['epoch']:>3} loss {train_loss:.4f} val {val_mean:.4f} "
            f"(best {state['best']:.4f}@{state['best_epoch']}) {epoch_seconds:.1f}s/epoch, "
            f"<= {left * mean_epoch / 60:.1f} min left"
        )
        if interrupt_after is not None and state["epoch"] == interrupt_after and state["stop_reason"] is None:
            raise InterruptedForTest(f"test interruption after epoch {state['epoch']}")

    if not best_path.is_file():  # stopped before any epoch finished
        raise RuntimeError(f"[{label}] stopped without a checkpoint; raise max_minutes")
    model.load_state_dict(_load_checkpoint(best_path)["state_dict"])
    result = _result(
        state["best_by_h"],
        state["best_epoch"],
        state["epoch"],
        state["stop_reason"],
        state["elapsed"],
        state["elapsed"] / max(state["epoch"], 1),
        n_params,
    )
    atomic_write_json(done_path, result)
    log(f"[{label}] done: {result['stop_reason']}, best val MAE {result['best_val_mae']:.4f} @ epoch {result['best_epoch']}")
    return result


def _result(per_h, best_epoch, epochs_run, stop_reason, seconds, sec_per_epoch, n_params) -> dict[str, Any]:
    return {
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "best_val_mae": float(np.mean(per_h)),
        "best_val_mae_by_horizon": {h: float(v) for h, v in zip(HORIZON_NAMES, per_h)},
        "stop_reason": stop_reason,
        "train_seconds": float(seconds),
        "seconds_per_epoch": float(sec_per_epoch),
        "params": int(n_params),
    }


# --------------------------------------------------------------------------- export
def export_predictions(
    pred: np.ndarray,
    dataset: WindowDataset,
    model_id: str,
    seed: int,
    split: str,
    out_root: str | Path,
) -> Path:
    """Write ``<out_root>/predictions/<model_id>/seed<S>_<split>.npz`` (plan section 4.7)."""
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    expected = (len(dataset), 4, dataset.num_nodes)
    if tuple(pred.shape) != expected:
        raise ValueError(f"predictions have shape {tuple(pred.shape)}, expected {expected}")
    if not np.isfinite(pred).all():
        raise FloatingPointError("refusing to export non-finite predictions")
    buffer = io.BytesIO()
    np.savez(
        buffer,
        pred=pred.astype(np.float32, copy=False),
        origin_ns=dataset.origin_ns,
        sensor_ids=np.array(dataset.sensor_ids, dtype=str),
    )
    path = Path(out_root) / "predictions" / model_id / f"seed{seed}_{split}.npz"
    atomic_write_bytes(path, buffer.getvalue())
    return path


def export_meta(
    out_root: str | Path,
    model_id: str,
    seed: int,
    fit_result: dict[str, Any],
    run_cfg: dict[str, Any],
    device: torch.device | str,
    subset: float = 1.0,
) -> Path:
    """Write ``seed<S>_meta.json`` next to the predictions."""
    meta = {
        "model_id": model_id,
        "seed": int(seed),
        "git_commit": git_commit(),
        "config_hash": config_hash(run_cfg),
        "config": run_cfg,
        "device": str(device),
        "torch_version": torch.__version__,
        "subset": float(subset),
        "params": fit_result["params"],
        "epochs_run": fit_result["epochs_run"],
        "best_epoch": fit_result["best_epoch"],
        "best_val_mae": fit_result["best_val_mae"],
        "best_val_mae_by_horizon": fit_result["best_val_mae_by_horizon"],
        "train_seconds": fit_result["train_seconds"],
        "seconds_per_epoch": fit_result["seconds_per_epoch"],
        "stop_reason": fit_result["stop_reason"],
    }
    path = Path(out_root) / "predictions" / model_id / f"seed{seed}_meta.json"
    atomic_write_json(path, meta)
    return path


ATTENTION_CLOCKS = ("08:00", "13:00", "17:30")


def attention_origin_positions(dataset: WindowDataset) -> list[int]:
    """Dataset positions of the 12 attention-export origins (plan section 4.7).

    For each of the first four weekdays of the split: the origin nearest to 08:00,
    13:00 and 17:30 on the stored clock. Ascending in time.
    """
    import pandas as pd

    ts = pd.DatetimeIndex(dataset.timestamps)
    days = [d for d in sorted(ts.normalize().unique()) if d.day_of_week < 5][:4]
    positions: list[int] = []
    for day in days:
        in_day = np.flatnonzero(ts.normalize() == day)
        if len(in_day) == 0:
            continue
        for clock in ATTENTION_CLOCKS:
            target = day + pd.Timedelta(hours=int(clock[:2]), minutes=int(clock[3:]))
            nearest = in_day[np.argmin(np.abs((ts[in_day] - target).asi8))]
            positions.append(int(nearest))
    return sorted(set(positions))


@torch.no_grad()
def export_attention(
    model: torch.nn.Module,
    dataset: WindowDataset,
    adj: torch.Tensor,
    device: torch.device | str,
    model_id: str,
    seed: int,
    out_root: str | Path,
) -> Path | None:
    """Save ``attn float16 [n_origins, layers, heads, N, N]`` if the model has ``attention_maps``."""
    if not hasattr(model, "attention_maps"):
        return None
    device = torch.device(device)
    model = model.to(device).eval()
    positions = attention_origin_positions(dataset)
    batch = dataset.get_batch(positions)
    maps = model.attention_maps(batch["x"], adj.to(device=device, dtype=torch.float32))
    if isinstance(maps, torch.Tensor):
        maps = [maps]
    stacked = torch.stack([m.float() for m in maps], dim=1)  # [B, layers, heads, N, N]
    if not torch.isfinite(stacked).all():
        raise FloatingPointError("attention maps contain non-finite values")
    buffer = io.BytesIO()
    np.savez(
        buffer,
        attn=stacked.half().cpu().numpy(),
        origin_ns=dataset.origin_ns[positions],
        sensor_ids=np.array(dataset.sensor_ids, dtype=str),
    )
    path = Path(out_root) / "attention" / f"{model_id}_seed{seed}.npz"
    atomic_write_bytes(path, buffer.getvalue())
    return path


# --------------------------------------------------------------------------- smoke
def smoke_check(
    model: torch.nn.Module,
    datasets: dict[str, WindowDataset],
    adj: torch.Tensor,
    device: torch.device | str,
    label: str = "model",
    steps: int = 2,
) -> dict[str, Any]:
    """2 training steps (grads finite, not all zero) plus one prediction pass; never raises on model bugs."""
    device = torch.device(device)
    issues: list[str] = []
    info: dict[str, Any] = {"label": label, "params": count_parameters(model), "ok": False}
    try:
        model = model.to(device)
        adj = adj.to(device=device, dtype=torch.float32)
        train_ds = datasets["train"]
        trainable = [p for p in model.parameters() if p.requires_grad]
        t0 = time.perf_counter()
        if trainable:
            optimizer = torch.optim.Adam(trainable, lr=1e-3)
            model.train()
            batches = train_ds.batches(min(8, len(train_ds)), shuffle=True, seed=0)
            nonzero_seen = False
            for _, batch in zip(range(steps), batches):
                out = model(batch["x"], adj)
                _check_output(out, batch, label)
                err, cnt = masked_mae_sums(train_ds.scaler.inverse(out), batch)
                loss = err.sum() / cnt.sum().clamp_min(1.0)
                if not torch.isfinite(loss):
                    issues.append("non-finite loss")
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grads = [p.grad for p in trainable if p.grad is not None]
                if len(grads) < len(trainable):
                    issues.append(f"{len(trainable) - len(grads)} parameter tensor(s) received no gradient")
                if any(not torch.isfinite(g).all() for g in grads):
                    issues.append("non-finite gradient")
                nonzero_seen |= any(bool((g != 0).any()) for g in grads)
                optimizer.step()
            if not nonzero_seen:
                issues.append("all gradients are zero")
        info["step_seconds"] = round((time.perf_counter() - t0) / max(steps, 1), 4)
        pred = predict(model, datasets["val"], adj, device, batch_size=8, label=label)
        info["pred_shape"] = list(pred.shape)
    except Exception as exc:  # noqa: BLE001 - report instead of crashing the whole smoke table
        issues.append(f"{type(exc).__name__}: {exc}")
    info["issues"] = issues
    info["ok"] = not issues
    return info
