"""Extreme Forecast Index (ECMWF-style) from an ensemble vs. a climatology."""
import numpy as np


def efi(ensemble: np.ndarray, climate: np.ndarray, n: int = 100) -> np.ndarray:
    """ensemble: (members, ...) forecasts; climate: (years, ...) same variable.
    Returns EFI in [-1, 1] per grid point (near +1 = extreme high)."""
    p = (np.arange(n) + 0.5) / n
    q = np.quantile(climate, p, axis=0)                       # climate quantiles
    F = np.stack([(ensemble <= qi).mean(axis=0) for qi in q])  # forecast CDF at q
    w = (1.0 / np.sqrt(p * (1 - p))).reshape(-1, *[1] * (F.ndim - 1))
    return (2 / np.pi) * ((p.reshape(w.shape) - F) * w).sum(axis=0) / n
