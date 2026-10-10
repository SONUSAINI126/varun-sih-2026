"""End-to-end: coarse forecast -> downscale -> physics gate -> probabilistic hazard."""
import numpy as np
from .downscale import downscale
from .physics_gate import filter_members


def run(coarse: np.ndarray, threshold_mm: float = 100.0) -> dict:
    members = filter_members(coarse, downscale(coarse))
    if len(members) == 0:
        return {"hazard": None, "members_kept": 0, "peak_prob": 0.0}
    prob = (members > threshold_mm).mean(axis=0)            # P(rain > threshold) per 5 km cell
    return {"hazard": prob, "members_kept": int(len(members)), "peak_prob": float(prob.max())}


if __name__ == "__main__":
    demo = np.random.default_rng(1).gamma(2, 25, (20, 30))  # fake 12 km rainfall field
    out = run(demo)
    print(out["members_kept"], "members kept | peak prob", round(out["peak_prob"], 2))
