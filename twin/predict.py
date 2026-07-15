"""Run the trained IOH predictor on a MAP trajectory.

The persisted MAP-only model uses features (map_last, map_slope) computed over the
observation window. Both are recomputable from the mechanistic twin's projected
MAP, so the same VitalDB-trained predictor scores the what-if projection live.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import twin.config as c
from twin.models.baselines import MAP_FEATURES

MODEL_PATH = c.RESULTS_DIR / "predictor_map_only.joblib"
META_PATH = c.RESULTS_DIR / "predictor_meta.json"


def load_predictor():
    """Load the persisted model + metadata, or (None, None) if not trained yet."""
    if not MODEL_PATH.exists():
        return None, None
    import joblib
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return model, meta


def _rolling_slope(y: np.ndarray, win: int) -> np.ndarray:
    """Least-squares slope over a trailing window of `win` samples, per index."""
    n = len(y)
    slope = np.zeros(n)
    for t in range(n):
        lo = max(0, t - win + 1)
        seg = y[lo:t + 1]
        if len(seg) >= 2:
            x = np.arange(len(seg))
            slope[t] = np.polyfit(x, seg, 1)[0]
    return slope


def ioh_probability(map_traj: np.ndarray, dt: float, model, stride: int = 10) -> np.ndarray:
    """Probability of IOH in the next horizon at each timestep of the projection.

    Features are evaluated on a coarse stride for speed, then interpolated to the
    full-length trajectory. `map_slope` is per-second (matches training features).
    """
    map_traj = np.asarray(map_traj, dtype=float)
    n = len(map_traj)
    win = int(c.OBS_WINDOW_SECONDS / dt)
    idx = np.arange(0, n, stride)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)

    slope_full = _rolling_slope(map_traj, win)
    feats = pd.DataFrame({
        "map_last": map_traj[idx],
        "map_slope": slope_full[idx],
    })[MAP_FEATURES]
    p_coarse = model.predict_proba(feats)
    return np.interp(np.arange(n), idx, p_coarse)
