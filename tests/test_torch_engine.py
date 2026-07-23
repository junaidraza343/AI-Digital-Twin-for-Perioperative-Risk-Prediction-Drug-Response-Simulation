import numpy as np
import pytest
pytest.importorskip("torch")
import torch

from twin.pkpd.params import POP_PROPOFOL, POP_PD, POP_NOREPI
from twin.pkpd.covariates import scale_propofol
from twin.pkpd.schedule import rate_vector
from twin.pkpd.engine import Patient, project_map
from twin.pkpd import torch_engine as te


def _sample_case(duration=600.0, dt=1.0):
    patient = Patient(age=50, weight=70, height=170, sex="M", map0=90.0)
    prop = [(0, 60, 200.0), (60, int(duration), 20.0)]   # induction bolus + maintenance
    norepi = [(120, 121, 8.0)]
    prop_rate = rate_vector(prop, dt, duration)
    norepi_rate = rate_vector(norepi, dt, duration)
    return patient, prop_rate, norepi_rate, dt, duration


def test_matches_numpy_engine_at_delta_zero():
    patient, prop_rate, norepi_rate, dt, duration = _sample_case()
    ref = project_map(patient, [(0, 60, 200.0), (60, int(duration), 20.0)],
                      [(120, 121, 8.0)], duration=duration, dt=dt).map
    out = te.project_map_torch(
        patient_batch=[patient],
        prop_rate=torch.tensor(prop_rate[None, :], dtype=torch.float64),
        norepi_rate=torch.tensor(norepi_rate[None, :], dtype=torch.float64),
        deltas=torch.zeros(1, 6, dtype=torch.float64),
        dt=dt,
    )
    got = out.detach().numpy()[0]
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-4)


def test_gradient_flows_to_deltas():
    patient, prop_rate, norepi_rate, dt, duration = _sample_case()
    deltas = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
    out = te.project_map_torch(
        patient_batch=[patient],
        prop_rate=torch.tensor(prop_rate[None, :], dtype=torch.float64),
        norepi_rate=torch.tensor(norepi_rate[None, :], dtype=torch.float64),
        deltas=deltas, dt=dt,
    )
    out.sum().backward()
    # EC50 (index 4) and ke0 (index 3) must have nonzero gradient
    g = deltas.grad[0]
    assert abs(g[4].item()) > 1e-6
    assert abs(g[3].item()) > 1e-6
