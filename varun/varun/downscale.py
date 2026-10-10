"""Stage 2: conditional diffusion downscaling 12 km -> 5 km (TODO: diffusers model)."""
import numpy as np


def downscale(coarse: np.ndarray, n_members: int = 10, seed: int = 0) -> np.ndarray:
    """Baseline stand-in: upsample 2.4x (12 km -> 5 km) + noise. Returns (members, H', W').
    Replace with a diffusion model conditioned on GNN output + topography."""
    rng = np.random.default_rng(seed)
    up = np.kron(coarse, np.ones((12, 12)))[::5, ::5]
    return np.clip(up + rng.normal(0, 0.05 * up.std() + 1e-6, (n_members, *up.shape)), 0, None)
