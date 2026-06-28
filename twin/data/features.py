"""Extract static + numeric-trend features over an observation window."""
import numpy as np
import pandas as pd
import twin.config as c

# Numeric tracks summarized as trend features (MAP handled explicitly).
_TREND_TRACKS = [
    "Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/ETCO2",
    "Solar8000/RR", "Solar8000/BT", "Solar8000/ART_SBP", "Solar8000/ART_DBP",
]


def _slope(values: np.ndarray) -> float:
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return 0.0
    x = np.arange(len(values))[valid]
    y = values[valid]
    return float(np.polyfit(x, y, 1)[0])


def _summ(prefix, values, out):
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        out[f"{prefix}_mean"] = np.nan
        out[f"{prefix}_std"] = np.nan
        out[f"{prefix}_min"] = np.nan
        out[f"{prefix}_max"] = np.nan
        out[f"{prefix}_last"] = np.nan
        out[f"{prefix}_slope"] = 0.0
        return
    out[f"{prefix}_mean"] = float(valid.mean())
    out[f"{prefix}_std"] = float(valid.std())
    out[f"{prefix}_min"] = float(valid.min())
    out[f"{prefix}_max"] = float(valid.max())
    out[f"{prefix}_last"] = float(valid[-1])
    out[f"{prefix}_slope"] = _slope(values)


def extract_features(frame: pd.DataFrame, t_end: int, static_row: dict) -> dict:
    """Build a feature dict for the window ending at t_end."""
    start = t_end - c.OBS_WINDOW_SECONDS
    out = dict(static_row)  # static covariates pass through unchanged

    map_track = c.MAP_TRACK if (c.MAP_TRACK in frame and frame[c.MAP_TRACK].notna().any()) else c.MAP_TRACK_FALLBACK
    map_win = frame[map_track].to_numpy(dtype=float)[start:t_end + 1]
    _summ("map", map_win, out)

    for track in _TREND_TRACKS:
        if track in frame:
            win = frame[track].to_numpy(dtype=float)[start:t_end + 1]
            _summ(track.split("/")[-1].lower(), win, out)
    return out
