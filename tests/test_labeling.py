import numpy as np
import pandas as pd
import pytest
import twin.config as c
from twin.data.labeling import label_windows, make_map_series


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


# ---------------------------------------------------------------------------
# make_map_series tests
# ---------------------------------------------------------------------------

def _make_frame(n=200, **columns):
    """Build a DataFrame with the given constant-value columns of length n."""
    return pd.DataFrame({col: np.full(n, val, dtype=float) for col, val in columns.items()})


def test_make_map_series_prefers_arterial():
    """When arterial track is valid, its values should be returned."""
    frame = _make_frame(
        **{c.MAP_TRACK: 80.0, c.MAP_TRACK_FALLBACK: 50.0}
    )
    arr = make_map_series(frame)
    np.testing.assert_array_equal(arr, np.full(200, 80.0))


def test_make_map_series_fallback_nibp_when_art_nan():
    """When the arterial track is all-NaN, NIBP values should be returned."""
    frame = pd.DataFrame({
        c.MAP_TRACK: np.full(200, np.nan),
        c.MAP_TRACK_FALLBACK: np.full(200, 70.0),
    })
    arr = make_map_series(frame)
    np.testing.assert_array_equal(arr, np.full(200, 70.0))


def test_make_map_series_clips_impossible_values():
    """Values outside [10, 250] should be replaced with NaN."""
    values = np.full(200, 80.0)
    values[10] = 300.0   # above physiological max (250)
    values[20] = 5.0     # below physiological min (10)
    frame = pd.DataFrame({c.MAP_TRACK: values})
    arr = make_map_series(frame)
    assert np.isnan(arr[10]), "300.0 should be clipped to NaN"
    assert np.isnan(arr[20]), "5.0 should be clipped to NaN"
    # Remaining values should be untouched
    mask = np.ones(200, dtype=bool)
    mask[10] = False
    mask[20] = False
    np.testing.assert_array_equal(arr[mask], np.full(198, 80.0))


def test_make_map_series_raises_when_no_map_track():
    """ValueError raised when neither MAP track is present in the frame."""
    frame = pd.DataFrame({"Solar8000/HR": np.full(50, 70.0)})
    with pytest.raises(ValueError, match="No MAP track available"):
        make_map_series(frame)


# ---------------------------------------------------------------------------
# NaN-horizon-drop test for label_windows
# ---------------------------------------------------------------------------

def test_horizon_nan_drop_excludes_window():
    """A window whose horizon is >50% NaN must be excluded by HORIZON_MIN_VALID."""
    # Use a long flat-80 array so there are many valid windows outside the NaN region.
    n = 3000
    m = np.full(n, 80.0, dtype=float)

    # Choose a t_end whose full 300-second horizon falls entirely within the NaN span.
    # We NaN out indices [1500, 1800) so a t_end of 1499 has horizon [1500, 1800).
    nan_start = 1500
    nan_end = nan_start + c.HORIZON_SECONDS  # 1800
    m[nan_start:nan_end] = np.nan

    df = label_windows(m)

    # t_end=1499 is valid (MAP=80, >= threshold) but its horizon is entirely NaN,
    # so it must be dropped.
    assert 1499 not in df.t_end.values, (
        "t_end=1499 should be dropped because its full horizon is NaN"
    )

    # Sanity: other windows well before the NaN region should still appear.
    early_windows = df[df.t_end <= c.OBS_WINDOW_SECONDS + 60]
    assert len(early_windows) > 0, "Expected valid windows well before the NaN region"
