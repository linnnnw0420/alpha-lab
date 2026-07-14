from __future__ import annotations

import random

try:  # optional
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def set_global_seed(seed: int) -> None:
    """
    Set global RNG seeds for reproducibility.

    Affects:
    - Python stdlib random
    - numpy (if installed)
    - torch (if installed; also seeds CUDA)
    """
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)


def new_numpy_random_generator(seed: int | None = None):
    """
    Create a new numpy Generator with its own seed.

    Returns:
        numpy.random.Generator

    Raises:
        ImportError: if numpy is not installed
    """
    if np is None:
        raise ImportError("numpy is required for new_numpy_random_generator()")
    return np.random.default_rng(seed)


__all__ = ["set_global_seed", "new_numpy_random_generator"]
