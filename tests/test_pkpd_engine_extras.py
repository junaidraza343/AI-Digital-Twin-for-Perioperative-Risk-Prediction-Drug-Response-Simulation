import numpy as np
from twin.pkpd.engine import project_map, project_band, ioh_risk, Patient


def _patient(map0=90.0):
    return Patient(age=60, weight=70, height=170, sex="M", map0=map0)


# --- ioh_risk ---

def test_risk_low_when_no_drug():
    proj = project_map(_patient(), [], [], duration=300.0)
    assert ioh_risk(proj) == "Low"


def test_risk_high_when_map_breaches_65():
    frail = Patient(age=80, weight=55, height=165, sex="F", map0=85.0)
    proj = project_map(frail, [(0.0, 900.0, 50.0)], [], duration=900.0)
    assert proj.map.min() < 65.0
    assert ioh_risk(proj) == "High"


def test_risk_medium_in_gray_zone():
    proj = project_map(_patient(), [(0.0, 600.0, 40.0)], [], duration=600.0)
    assert 65.0 <= proj.map.min() < 75.0
    assert ioh_risk(proj) == "Medium"


# --- project_band ---

def test_band_brackets_the_central_trajectory():
    p = _patient()
    lo, hi, central = project_band(p, [(0.0, 600.0, 30.0)], [], duration=600.0)
    assert lo.shape == hi.shape == central.shape
    assert np.all(hi >= central - 1e-9)
    assert np.all(central >= lo - 1e-9)


def test_band_has_width_under_drug():
    p = _patient()
    lo, hi, _ = project_band(p, [(0.0, 600.0, 40.0)], [], duration=600.0)
    assert np.max(hi - lo) > 0.5  # non-trivial uncertainty when drug is on


def test_band_wider_with_larger_spread():
    p = _patient()
    lo1, hi1, _ = project_band(p, [(0.0, 600.0, 40.0)], [], duration=600.0, spread=0.1)
    lo2, hi2, _ = project_band(p, [(0.0, 600.0, 40.0)], [], duration=600.0, spread=0.3)
    assert np.max(hi2 - lo2) > np.max(hi1 - lo1)
