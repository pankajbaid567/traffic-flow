"""Dataset tools.

Person A: quality, descriptive, eda, run_person_a.
Person B: load_data, split, features, dataset_builder (import those modules directly).
"""

from .quality import analyze_frame

__all__ = ["analyze_frame"]
