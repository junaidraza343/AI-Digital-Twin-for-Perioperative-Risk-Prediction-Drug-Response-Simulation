import numpy as np
from twin.pkpd.engine import project_map, Patient, MapProjection


def _patient():
    return Patient(age=60, weight=70, height=170, sex="M", map0=90.0)


def test_returns_projection_with_aligned_axes():
    proj = project_map(_patient(), prop_schedule=[], norepi_schedule=[], duration=300.0)
    assert isinstance(proj, MapProjection)
    assert proj.t.shape == proj.map.shape
    assert proj.t[0] == 0.0 and proj.t[-1] == 300.0


def test_no_drug_stays_at_baseline():
    proj = project_map(_patient(), prop_schedule=[], norepi_schedule=[], duration=300.0)
    assert np.allclose(proj.map, 90.0)


def test_propofol_lowers_map():
    proj = project_map(_patient(), prop_schedule=[(0.0, 600.0, 30.0)],
                       norepi_schedule=[], duration=600.0)
    assert proj.map.min() < 90.0


def test_heavier_dose_lowers_min_map_more():
    light = project_map(_patient(), [(0.0, 600.0, 15.0)], [], duration=600.0)
    heavy = project_map(_patient(), [(0.0, 600.0, 40.0)], [], duration=600.0)
    assert heavy.map.min() < light.map.min()


def test_norepi_raises_map_relative_to_none():
    base = project_map(_patient(), [(0.0, 600.0, 30.0)], [], duration=600.0)
    rescued = project_map(_patient(), [(0.0, 600.0, 30.0)], [(120.0, 600.0, 10.0)], duration=600.0)
    assert rescued.map.min() > base.map.min()


def test_deltas_change_trajectory():
    pop = project_map(_patient(), [(0.0, 600.0, 30.0)], [], duration=600.0)
    pers = project_map(_patient(), [(0.0, 600.0, 30.0)], [], duration=600.0,
                       deltas={"EC50": -0.5})  # more sensitive -> lower MAP
    assert not np.allclose(pop.map, pers.map)
    assert pers.map.min() < pop.map.min()


def test_deterministic():
    a = project_map(_patient(), [(0.0, 300.0, 20.0)], [], duration=300.0)
    b = project_map(_patient(), [(0.0, 300.0, 20.0)], [], duration=300.0)
    assert np.array_equal(a.map, b.map)


def test_minutes_below_65_reported():
    proj = project_map(_patient(), [(0.0, 600.0, 60.0)], [], duration=600.0)
    assert proj.minutes_below_65 >= 0.0
