"""Phase 2 (deep learning) package: graph-aware traffic-speed forecasting on METR-LA.

Shared core owned by Member 1 (data, graph, registry, train, cli). Model code
lives in ``src/phase2/models`` and registers itself with ``register_model``.
Phase 1 code under ``src/data``, ``src/models`` and ``src/visualization`` is
imported, never edited.
"""

import os
import warnings

# Frozen Phase 1 code uses DatetimeIndex.dayofweek, which pandas 3 deprecates in favour of
# day_of_week. The warning is harmless noise for Phase 2 runs; hide only that message.
warnings.filterwarnings("ignore", message=r"DatetimeIndex\.dayofweek is deprecated")

# Must be set before torch runs its first MPS op. Harmless on cpu/cuda.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
