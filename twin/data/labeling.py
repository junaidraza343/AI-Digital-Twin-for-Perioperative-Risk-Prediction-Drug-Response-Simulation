"""Turn a 1 Hz MAP series into labeled prediction windows."""
import numpy as np
import pandas as pd
import twin.config as c


def make_map_series(frame: pd.DataFrame) -> np.ndarray:
    """Extract a 1 Hz MAP array, preferring the arterial track, else NIBP."""
    if c.MAP_TRACK in frame and frame[c.MAP_TRACK].notna().any():
        s = frame[c.MAP_TRACK].astype(float)
    elif c.MAP_TRACK_FALLBACK in frame:
        s = frame[c.MAP_TRACK_FALLBACK].astype(float)
    else:
        raise ValueError("No MAP track available in frame")
    # Physiologically impossible values -> NaN
    arr = s.to_numpy(dtype=float)
    arr[(arr < 10) | (arr > 250)] = np.nan
    return arr


def label_windows(map_series: np.ndarray) -> pd.DataFrame:
    """Roll windows over a 1 Hz MAP array and assign label + category.

    A window ends at t_end (seconds). It is a prediction point only if the full
    observation window precedes it and the full horizon follows it. The current
    MAP at t_end must be valid and >= threshold (otherwise the event is already
    underway -> detection, not prediction, so we drop it).

    y = 1 iff MAP < threshold for >= EVENT_MIN_SECONDS cumulative seconds in the
    horizon (t_end, t_end + HORIZON]. category: overt (y==1), else gray if the
    minimum valid horizon MAP < GRAY_HIGH, else stable.
    """
    n = len(map_series)
    rows = []
    last_end = n - c.HORIZON_SECONDS - 1
    for t_end in range(c.OBS_WINDOW_SECONDS, last_end + 1, c.STRIDE_SECONDS):
        cur = map_series[t_end]
        if np.isnan(cur) or cur < c.MAP_THRESHOLD:
            continue
        horizon = map_series[t_end + 1: t_end + 1 + c.HORIZON_SECONDS]
        valid = horizon[~np.isnan(horizon)]
        if len(valid) < c.HORIZON_MIN_VALID * c.HORIZON_SECONDS:
            continue
        seconds_below = int(np.sum(valid < c.MAP_THRESHOLD))
        y = 1 if seconds_below >= c.EVENT_MIN_SECONDS else 0
        if y == 1:
            category = "overt"
        elif valid.min() < c.GRAY_HIGH:
            category = "gray"
        else:
            category = "stable"
        rows.append((t_end, y, category))
    return pd.DataFrame(rows, columns=["t_end", "y", "category"])
