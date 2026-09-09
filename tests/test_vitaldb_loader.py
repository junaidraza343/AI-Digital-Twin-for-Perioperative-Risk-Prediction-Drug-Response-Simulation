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


def test_cached_caseids_lists_only_files_for_the_current_track_set(tmp_path, monkeypatch):
    """Cache files are tagged by a hash of NUMERIC_TRACKS; only the current tag counts.

    A file left over from an earlier track set can never be served, so reporting
    it as cached would send callers down a path that silently hits the network.
    """
    import twin.config as c
    from twin.data.vitaldb_loader import cached_caseids, _tracks_tag

    monkeypatch.setattr(c, "DATA_CACHE", tmp_path)
    tag = _tracks_tag()
    (tmp_path / f"case_11_{tag}.parquet").write_bytes(b"")
    (tmp_path / f"case_12_{tag}.parquet").write_bytes(b"")
    (tmp_path / "case_13_deadbeef.parquet").write_bytes(b"")   # stale track set
    (tmp_path / "notes.txt").write_text("ignore me")

    assert cached_caseids() == {11, 12}
