"""Compatibility exports for effective-observation asset readers."""
from mgb_ops.assets.effective_observations import (
    InsufficientObservationSourceError,
    load_effective_observations,
    load_effective_rainfall_observations,
    load_preferred_rainfall_observations,
)

__all__ = [
    "InsufficientObservationSourceError", "load_effective_observations",
    "load_effective_rainfall_observations", "load_preferred_rainfall_observations",
]
