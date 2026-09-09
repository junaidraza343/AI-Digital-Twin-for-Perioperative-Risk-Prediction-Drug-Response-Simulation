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


def _arr_cadence(n, propofol=60.0, cadence=2, value=80.0, lead_in=0):
    """Frame whose ART_MBP is sampled every `cadence` seconds (Solar8000 is 2s).

    On the 1 Hz grid this caps raw non-NaN coverage at 1/cadence. `lead_in`
    leaves the head and tail of the record without ART, mirroring an arterial
    line sited after induction and removed before emergence.
    """
    cols = c.NUMERIC_TRACKS
    a = np.full((n, len(cols)), np.nan)
    a[:, cols.index(c.PROPOFOL_TRACK)] = propofol
    span = np.arange(lead_in, n - lead_in, cadence)
    a[span, cols.index(c.MAP_TRACK)] = value
    return pd.DataFrame(a, columns=cols)


def test_case_qualifies_with_real_2s_art_cadence():
    """Solar8000 samples ART_MBP every 2s with a post-induction lead-in.

    Raw coverage lands just under 0.5 (real case 16 measured 0.493), which the
    original >=0.5 gate rejected -- emptying the whole SP4 cohort.
    """
    frame = _arr_cadence(12868, cadence=2, lead_in=300)
    raw = frame[c.MAP_TRACK].notna().mean()
    assert raw < 0.5  # the trap: never reachable at 2s cadence
    ok, reason = case_qualifies(frame)
    assert ok is True, reason


def test_case_rejected_when_art_gaps_are_long():
    """A genuinely sparse line (sampled every 30s) is not continuous ART."""
    frame = _arr_cadence(600, cadence=30)
    ok, reason = case_qualifies(frame)
    assert ok is False and "ART" in reason


def test_case_rejected_when_art_is_all_artifact():
    """Out-of-range values (e.g. -53 or 277 mmHg) are artifacts, not coverage."""
    frame = _arr_cadence(600, cadence=2, value=-53.0)
    ok, reason = case_qualifies(frame)
    assert ok is False and "ART" in reason


def _fake_index():
    """Minimal stand-in for VitalDB's /trks index (caseid, tname)."""
    return pd.DataFrame({
        "caseid": [1, 1, 2, 3, 3],
        "tname": [c.PROPOFOL_TRACK, c.MAP_TRACK,   # case 1: both -> candidate
                  c.PROPOFOL_TRACK,                # case 2: propofol only
                  c.MAP_TRACK, "Solar8000/HR"],    # case 3: no propofol
    })


def test_candidates_with_tracks_keeps_only_cases_having_both():
    from scripts.measure_drug_coverage import candidates_with_tracks
    eligible = pd.DataFrame({"caseid": [1, 2, 3, 4]})
    assert candidates_with_tracks(eligible, _fake_index()) == [1]


def test_candidates_with_tracks_respects_eligibility():
    """A case with both tracks is still dropped if it is not in the cohort."""
    from scripts.measure_drug_coverage import candidates_with_tracks
    eligible = pd.DataFrame({"caseid": [2, 3]})
    assert candidates_with_tracks(eligible, _fake_index()) == []
