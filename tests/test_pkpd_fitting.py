import numpy as np

from twin.pkpd.engine import Patient, project_map
from twin.pkpd.fitting import fit_deltas


def _patient():
    return Patient(age=65, weight=70, height=170, sex="M", map0=90.0)


def test_recovers_known_deltas():
    p = _patient()
    prop = [(0.0, 900.0, 35.0)]
    true = {"EC50": -0.3, "ke0": 0.2}
    observed = project_map(p, prop, [], duration=900.0, deltas=true).map
    fitted, rmse = fit_deltas(p, prop, [], observed, duration=900.0)
    assert abs(fitted["EC50"] - true["EC50"]) < 0.15
    assert abs(fitted["ke0"] - true["ke0"]) < 0.2
    assert rmse < 1.0  # mmHg


def test_population_observed_fits_near_zero():
    p = _patient()
    prop = [(0.0, 900.0, 30.0)]
    observed = project_map(p, prop, [], duration=900.0).map
    fitted, rmse = fit_deltas(p, prop, [], observed, duration=900.0)
    assert abs(fitted["EC50"]) < 0.2 and abs(fitted["ke0"]) < 0.25
    assert rmse < 0.5


def test_fit_reduces_error_vs_population():
    p = _patient()
    prop = [(0.0, 900.0, 40.0)]
    true = {"EC50": -0.45, "ke0": -0.25}
    observed = project_map(p, prop, [], duration=900.0, deltas=true).map
    pop = project_map(p, prop, [], duration=900.0).map
    pop_rmse = float(np.sqrt(np.mean((pop - observed) ** 2)))
    _, fit_rmse = fit_deltas(p, prop, [], observed, duration=900.0)
    assert fit_rmse < pop_rmse
