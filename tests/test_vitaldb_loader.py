import numpy as np
import pandas as pd
import twin.config as c
from twin.data.vitaldb_loader import load_numeric_frame


def test_loader_called_once_then_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "DATA_CACHE", tmp_path)
    calls = {"n": 0}

    def fake_loader(caseid, tracks, interval):
        calls["n"] += 1
        # vitaldb returns a 2D array: rows = time, cols = tracks
        return np.tile(np.arange(5.0), (len(tracks), 1)).T

    f1 = load_numeric_frame(1, loader_fn=fake_loader)
    f2 = load_numeric_frame(1, loader_fn=fake_loader)
    assert calls["n"] == 1           # second call served from cache
    assert isinstance(f1, pd.DataFrame)
    assert list(f1.columns) == c.NUMERIC_TRACKS
    pd.testing.assert_frame_equal(f1, f2)


def test_returns_one_column_per_track(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "DATA_CACHE", tmp_path)

    def fake_loader(caseid, tracks, interval):
        return np.zeros((10, len(tracks)))

    frame = load_numeric_frame(7, loader_fn=fake_loader)
    assert frame.shape == (10, len(c.NUMERIC_TRACKS))
