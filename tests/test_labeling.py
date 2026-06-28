import numpy as np
import twin.config as c
from twin.data.labeling import label_windows


def _flat(value, n=3000):
    return np.full(n, value, dtype=float)


def test_stable_window_all_high():
    m = _flat(80.0)
    df = label_windows(m)
    assert (df.y == 0).all()
    assert (df.category == "stable").all()


def test_overt_event_90s_dip():
    m = _flat(80.0)
    # 90s sustained dip below 65 inside the horizon of the first valid window
    start = c.OBS_WINDOW_SECONDS + 30
    m[start:start + 90] = 60.0
    df = label_windows(m)
    pos = df[df.t_end < start]  # windows whose horizon covers the dip
    assert (pos.y == 1).any()
    assert (df.loc[df.y == 1, "category"] == "overt").all()


def test_short_dip_40s_is_negative_graylike():
    m = _flat(80.0)
    start = c.OBS_WINDOW_SECONDS + 30
    m[start:start + 40] = 60.0  # only 40s < 65, fails the 60s rule
    df = label_windows(m)
    covering = df[(df.t_end < start) & (df.t_end + c.HORIZON_SECONDS >= start + 40)]
    assert (covering.y == 0).all()
    assert (covering.category == "gray").all()  # min horizon MAP < 75


def test_gray_band_70_is_gray_negative():
    m = _flat(80.0)
    start = c.OBS_WINDOW_SECONDS + 30
    m[start:start + 120] = 70.0  # in [65,75), never below 65
    df = label_windows(m)
    covering = df[(df.t_end < start) & (df.t_end + c.HORIZON_SECONDS >= start + 120)]
    assert (covering.y == 0).all()
    assert (covering.category == "gray").all()


def test_already_hypotensive_window_dropped():
    m = _flat(80.0)
    t = c.OBS_WINDOW_SECONDS + 300
    m[t] = 60.0  # current MAP at t_end already < 65 -> that window excluded
    df = label_windows(m)
    assert (df.t_end != t).all()


def test_no_windows_before_obs_or_after_horizon():
    m = _flat(80.0, n=c.OBS_WINDOW_SECONDS + c.HORIZON_SECONDS + 5)
    df = label_windows(m)
    assert df.t_end.min() >= c.OBS_WINDOW_SECONDS
    assert (df.t_end + c.HORIZON_SECONDS <= len(m) - 1).all()
