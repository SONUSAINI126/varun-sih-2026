"""Physics validation gate: rejects unrealistic downscaled members."""
import numpy as np


def passes(coarse: np.ndarray, fine: np.ndarray, tol: float = 0.1) -> bool:
    if np.any(fine < 0):                                    # no negative rainfall
        return False
    return abs(fine.mean() - coarse.mean()) <= tol * (coarse.mean() + 1e-6)  # mass consistency


def filter_members(coarse, members):
    kept = [m for m in members if passes(coarse, m)]
    return np.stack(kept) if kept else members[:0]
