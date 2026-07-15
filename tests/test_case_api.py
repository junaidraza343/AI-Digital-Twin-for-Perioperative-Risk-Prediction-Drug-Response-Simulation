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
