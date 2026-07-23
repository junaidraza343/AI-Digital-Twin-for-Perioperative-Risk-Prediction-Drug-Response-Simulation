import numpy as np
import pandas as pd
import twin.config as c
from scripts.measure_drug_coverage import case_qualifies, measure_coverage


def _arr(n, propofol=0.0, art=True):
    """Build a (n, len(NUMERIC_TRACKS)) array in NUMERIC_TRACKS column order."""
    cols = c.NUMERIC_TRACKS
    a = np.full((n, len(cols)), np.nan)
    a[:, cols.index(c.PROPOFOL_TRACK)] = propofol
    if art:
        a[:, cols.index(c.MAP_TRACK)] = 80.0
    return a


def test_case_qualifies_true_with_propofol_and_art():
    frame = pd.DataFrame(_arr(600, propofol=60.0, art=True), columns=c.NUMERIC_TRACKS)
    ok, reason = case_qualifies(frame)
    assert ok is True


def test_case_rejected_without_propofol():
    frame = pd.DataFrame(_arr(600, propofol=0.0, art=True), columns=c.NUMERIC_TRACKS)
    ok, reason = case_qualifies(frame)
    assert ok is False and "propofol" in reason


def test_measure_coverage_counts_qualifying_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "DATA_CACHE", tmp_path)
    eligible = pd.DataFrame({"caseid": [1, 2]})

    def fake_loader(caseid, tracks, interval):
        return _arr(600, propofol=60.0 if caseid == 1 else 0.0, art=True)

    summary = measure_coverage(eligible, loader_fn=fake_loader)
    assert summary["n_qualifying"] == 1
    assert 1 in summary["caseids"]
