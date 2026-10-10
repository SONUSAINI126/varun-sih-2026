"""Stage 1: spherical-mesh GNN for anomaly detection + tracking (TODO: DGL model)."""
import numpy as np


def detect_and_track(efi_series: np.ndarray, threshold: float = 0.8):
    """efi_series: (time, lat, lon). Placeholder tracker: centroid of EFI > threshold.
    Replace with a GraphCast-style encoder-processor-decoder on an icosahedral mesh."""
    tracks = []
    for t, field in enumerate(efi_series):
        ys, xs = np.where(field > threshold)
        if len(ys):
            tracks.append((t, float(ys.mean()), float(xs.mean()), float(field.max())))
    return tracks
