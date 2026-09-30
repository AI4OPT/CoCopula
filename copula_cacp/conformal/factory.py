"""Conformalizer construction and the hyperparameter grids searched per window."""
from __future__ import annotations

from typing import Any, Dict

# Grids searched on the validation week (Appendix B.1); the winner by mean
# Winkler score is refit on calibration+validation.
SEARCH_SPACES: Dict[str, Dict[str, list]] = {
    "none": {},                                   # pass-through, nothing to tune
    "cqr": {},                                    # no hyperparameters
    "kernel": {"kernel": ["rbf", "laplacian"], "gamma": [0.5, 1.0, 2.0]},
    "knn": {"nneighbors": [50, 100, 200, 500, 1000]},
    "kmeans": {"n_clusters": [3, 5, 8, 12]},
}

# Methods that consume context vectors.
METHODS_REQUIRING_COVS = {"kernel", "knn", "kmeans"}


def requires_covariates(method_name: str) -> bool:
    return method_name.lower() in METHODS_REQUIRING_COVS


def make_conformalizer(name: str, **kw: Any):
    key = name.lower()
    if key == "cqr":
        from .cqr import CQRConformalizer
        return CQRConformalizer(**kw)
    if key == "kernel":
        from .kernel import KernelConformalizer
        return KernelConformalizer(**kw)
    if key == "knn":
        from .knn import KNNConformalizer
        return KNNConformalizer(**kw)
    if key == "kmeans":
        from .kmeans import KMeansConformalizer
        return KMeansConformalizer(**kw)
    if key == "none":
        raise ValueError(
            "Method 'none' is the uncalibrated pass-through and is handled in "
            "calibrate.export_passthrough(), not via make_conformalizer()."
        )
    raise ValueError(
        f"Unknown conformalizer: {name!r}. Available: cqr, kernel, knn, kmeans, none"
    )


__all__ = [
    "SEARCH_SPACES",
    "METHODS_REQUIRING_COVS",
    "requires_covariates",
    "make_conformalizer",
]
