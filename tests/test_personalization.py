import numpy as np
import pytest
pytest.importorskip("torch")
import torch
from twin.pkpd.engine import Patient
from twin.pkpd.schedule import rate_vector
from twin.pkpd import torch_engine as te
from twin.eval.personalization import (
    per_patient_map_rmse, delta_identifiability, whatif_monotonic)


def _case(delta_ec50=0.3, T=300):
    patient = Patient(50, 70, 170, "M", 90.0)
    prop = torch.tensor(rate_vector([(0, 60, 200.0), (60, T, 20.0)], 1.0,
                        float(T - 1))[None, :], dtype=torch.float64)
    norepi = torch.tensor(rate_vector([(120, 121, 8.0)], 1.0,
                          float(T - 1))[None, :], dtype=torch.float64)
    d = torch.zeros(1, 6, dtype=torch.float64); d[0, 4] = delta_ec50
    observed = te.project_map_torch([patient], prop, norepi, d).detach()
    return patient, prop, norepi, observed, d


def test_personalized_beats_population_rmse():
    patient, prop, norepi, observed, d = _case()
    rmse_pop = per_patient_map_rmse([patient], prop, norepi, observed,
                                    torch.zeros(1, 6, dtype=torch.float64))
    rmse_pers = per_patient_map_rmse([patient], prop, norepi, observed, d)
    assert rmse_pers[0] < rmse_pop[0]
    assert rmse_pers[0] < 1e-3


def test_delta_identifiability_returns_six_values():
    deltas = torch.zeros(5, 6, dtype=torch.float64)
    deltas[:, 4] = torch.linspace(-0.3, 0.3, 5)  # EC50 varies, others 0
    spread = delta_identifiability(deltas)
    assert spread.shape == (6,)
    assert spread[4] > spread[2]  # EC50 moves more than V3


def test_whatif_monotonic_more_propofol_more_risk():
    patient = Patient(50, 70, 170, "M", 90.0)
    assert whatif_monotonic(patient) is True
