"""Auto-import every module in this folder so ``@register_model`` decorators run.

A module that fails to import is recorded in ``IMPORT_ERRORS`` instead of
breaking everyone else's models; the registry reports it when a model name is
missing.
"""

from __future__ import annotations

import importlib
import pkgutil

IMPORT_ERRORS: dict[str, str] = {}

for _info in pkgutil.iter_modules(__path__):
    if _info.name.startswith("_"):
        continue
    _module_name = f"{__name__}.{_info.name}"
    try:
        importlib.import_module(_module_name)
    except Exception as exc:  # noqa: BLE001 - keep other models usable
        IMPORT_ERRORS[_module_name] = f"{type(exc).__name__}: {exc}"
