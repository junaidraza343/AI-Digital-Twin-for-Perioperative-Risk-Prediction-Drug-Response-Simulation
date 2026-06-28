import numpy as np
import pandas as pd
import twin.config as c
from twin.data.features import extract_features


def _frame(n=2000):
    # MAP ramps linearly from 100 down to 80 over the whole frame
    idx = np.arange(n)
    return pd.DataFrame({
        c.MAP_TRACK: np.linspace(100.0, 80.0, n),
        "Solar8000/HR": np.full(n, 70.0),
    }, index=idx)


def test_returns_static_and_trend_features():
    frame = _frame()
    static = {"age": 50, "sex": 1, "bmi": 24.0}
    t_end = c.OBS_WINDOW_SECONDS + 100
    feats = extract_features(frame, t_end, static)
    assert feats["age"] == 50
    assert "map_mean" in feats
    assert "map_slope" in feats
    assert "map_last" in feats


def test_map_last_matches_value_at_t_end():
    frame = _frame()
    t_end = c.OBS_WINDOW_SECONDS + 100
    feats = extract_features(frame, t_end, {})
    assert abs(feats["map_last"] - frame[c.MAP_TRACK].iloc[t_end]) < 1e-6


def test_map_slope_is_negative_for_downward_ramp():
    frame = _frame()
    t_end = c.OBS_WINDOW_SECONDS + 100
    feats = extract_features(frame, t_end, {})
    assert feats["map_slope"] < 0


def test_map_extremes_clipped_to_nan():
    # Place an impossible MAP value (300) inside the observation window and
    # verify it does not appear in map_max (i.e. it is clipped to NaN).
    frame = _frame()
    t_end = c.OBS_WINDOW_SECONDS + 50
    # Index OBS_WINDOW_SECONDS falls inside [t_end - OBS_WINDOW_SECONDS, t_end]
    arr = frame[c.MAP_TRACK].to_numpy(dtype=float)
    arr[c.OBS_WINDOW_SECONDS] = 300.0
    frame2 = frame.copy()
    frame2[c.MAP_TRACK] = arr
    feats = extract_features(frame2, t_end, {})
    assert feats["map_max"] <= 250.0
