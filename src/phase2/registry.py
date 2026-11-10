"""Model registry: models register with a decorator, the CLI looks them up by name.

Usage in a model file under ``src/phase2/models``::

    from src.phase2.registry import register_model

    @register_model("gat")
    def build(cfg: dict) -> torch.nn.Module:
        # cfg = {num_nodes: 207, in_channels: 7, history: 12, num_horizons: 4, **run params}
        ...

Model contract (see reports/phase2_plan.md section 4.4)::

    class Model(nn.Module):
        needs_graph: bool                       # False for graph-free models
        def forward(self, x, adj): ...          # [B,12,N,C], [N,N] -> [B,4,N] z-scored speed
        def attention_maps(self, x, adj): ...   # OPTIONAL: list per layer of [B, heads, N, N]

Every module in ``src/phase2/models`` is imported on first lookup, so adding a
file is enough to register its models; no shared file is edited.
"""

from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable[[dict], object]] = {}


def register_model(name: str) -> Callable:
    """Decorator registering a ``build(cfg) -> nn.Module`` function under ``name``."""
    if not name or not isinstance(name, str):
        raise ValueError("model name must be a non-empty string")

    def decorator(builder: Callable[[dict], object]) -> Callable[[dict], object]:
        existing = _REGISTRY.get(name)
        if existing is not None and (
            existing.__module__ != builder.__module__ or existing.__qualname__ != builder.__qualname__
        ):
            raise ValueError(
                f"model name {name!r} is already registered by "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        _REGISTRY[name] = builder
        return builder

    return decorator


def _ensure_loaded() -> None:
    import src.phase2.models  # noqa: F401  (auto-imports the model modules)


def list_models() -> list[str]:
    _ensure_loaded()
    return sorted(_REGISTRY)


def get_builder(name: str) -> Callable[[dict], object]:
    """Return the builder for ``name`` or raise a ``KeyError`` that lists what exists."""
    _ensure_loaded()
    if name not in _REGISTRY:
        from src.phase2 import models as models_pkg

        errors = getattr(models_pkg, "IMPORT_ERRORS", {})
        hint = ""
        if errors:
            hint = " Modules that failed to import: " + "; ".join(f"{m}: {e}" for m, e in errors.items())
        raise KeyError(f"unknown model {name!r}. Registered: {sorted(_REGISTRY)}.{hint}")
    return _REGISTRY[name]


def build_model(name: str, cfg: dict):
    """Build a registered model from ``cfg`` (see module docstring for the keys)."""
    return get_builder(name)(cfg)
