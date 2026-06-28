import numpy as np
import pandas as pd
import twin.config as c
from scripts.build_dataset import build_windows_for_case


def test_build_windows_for_case_returns_labeled_features(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "DATA_CACHE", tmp_path)
    n = c.OBS_WINDOW_SECONDS + c.HORIZON_SECONDS + 200
    # Build a frame: stable MAP at 85 with one 90s dip near the end.
    arr = np.full((n, len(c.NUMERIC_TRACKS)), 85.0)
    map_col = c.NUMERIC_TRACKS.index(c.MAP_TRACK)
    dip = c.OBS_WINDOW_SECONDS + 60
    arr[dip:dip + 90, map_col] = 60.0

    def fake_loader(caseid, tracks, interval):
        return arr

    static = {"caseid": 1, "age": 50, "sex": 1, "bmi": 24.0}
    out = build_windows_for_case(1, static, loader_fn=fake_loader)
    assert {"caseid", "t_end", "y", "category", "map_last"} <= set(out.columns)
    assert (out.caseid == 1).all()
    assert out.y.isin([0, 1]).all()
