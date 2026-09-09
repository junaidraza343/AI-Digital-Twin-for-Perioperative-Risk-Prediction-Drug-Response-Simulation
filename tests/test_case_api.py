"""Guarded tests for the real-case loader (skipped when no cached data present)."""
import pytest

from twin.data.case_api import available_cases, load_case_bundle
from twin.predict import load_predictor

_CASES = available_cases()
pytestmark = pytest.mark.skipif(not _CASES, reason="no cached VitalDB cases available")


def test_bundle_has_real_map_and_patient():
    b = load_case_bundle(_CASES[0])
    assert len(b["t_min"]) == len(b["map_real"]) > 0
    assert set(b["patient"]) == {"age", "weight", "height", "sex", "map0"}
    assert 40 <= b["patient"]["map0"] <= 130


def test_bundle_with_predictor_returns_probability():
    model, _ = load_predictor()
    if model is None:
        pytest.skip("predictor not trained")
    b = load_case_bundle(_CASES[0], model=model)
    assert len(b["prob_real"]) == len(b["map_real"])
    assert 0.0 <= b["peak_prob"] <= 1.0


def test_scan_demo_cases_never_downloads_uncached_cases(tmp_path, monkeypatch):
    """scan_demo_cases is documented as an OFFLINE scan and must behave like one.

    It previously walked every eligible case and loaded each one, so on a
    deployment that bundles only a handful of cases it fired thousands of VitalDB
    downloads and pinned the request worker.
    """
    import twin.config as c
    import twin.data.case_api as case_api
    from twin.data.vitaldb_loader import _tracks_tag

    monkeypatch.setattr(c, "DATA_CACHE", tmp_path)
    monkeypatch.setattr(c, "RESULTS_DIR", tmp_path)
    (tmp_path / f"case_7_{_tracks_tag()}.parquet").write_bytes(b"")
    monkeypatch.setattr(case_api, "_all_cases", lambda: [7, 8, 9])

    seen = []

    def spy_loader(caseid):
        seen.append(caseid)
        raise RuntimeError("unreadable stub")   # content does not matter here

    monkeypatch.setattr(case_api, "load_numeric_frame", spy_loader)
    case_api.available_cases.cache_clear()
    case_api.scan_demo_cases()

    assert seen == [7], f"attempted uncached cases: {seen}"
