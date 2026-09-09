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
    # All six deltas (V1, V2, V3, ke0, EC50, gamma) must have nonzero gradient
    assert torch.all(deltas.grad.abs() > 1e-6)


def test_matches_numpy_engine_with_nonzero_deltas():
    patient, prop_rate, norepi_rate, dt, duration = _sample_case()
    deltas = {"V1": 0.1, "ke0": -0.2, "EC50": 0.3}   # subset; others 0
    ref = project_map(patient, [(0, 60, 200.0), (60, int(duration), 20.0)],
                      [(120, 121, 8.0)], duration=duration, dt=dt, deltas=deltas).map
    # order: (V1, V2, V3, ke0, EC50, gamma)
    dvec = torch.tensor([[0.1, 0.0, 0.0, -0.2, 0.3, 0.0]], dtype=torch.float64)
    out = te.project_map_torch([patient],
        torch.tensor(prop_rate[None, :], dtype=torch.float64),
        torch.tensor(norepi_rate[None, :], dtype=torch.float64),
        dvec, dt=dt).detach().numpy()[0]
    assert np.allclose(out, ref, atol=1e-4)


def test_batched_matches_per_patient():
    dt, duration = 1.0, 600.0
    p1 = Patient(age=40, weight=60, height=165, sex="F", map0=88.0)
    p2 = Patient(age=65, weight=90, height=180, sex="M", map0=95.0)
    prop = rate_vector([(0, 60, 200.0), (60, int(duration), 20.0)], dt, duration)
    norepi = rate_vector([(120, 121, 8.0)], dt, duration)
    prop_t = torch.tensor(np.stack([prop, prop]), dtype=torch.float64)
    norepi_t = torch.tensor(np.stack([norepi, norepi]), dtype=torch.float64)
    out = te.project_map_torch([p1, p2], prop_t, norepi_t,
                               torch.zeros(2, 6, dtype=torch.float64), dt=dt).detach().numpy()
    ref1 = project_map(p1, [(0, 60, 200.0), (60, int(duration), 20.0)], [(120, 121, 8.0)],
                       duration=duration, dt=dt).map
    ref2 = project_map(p2, [(0, 60, 200.0), (60, int(duration), 20.0)], [(120, 121, 8.0)],
                       duration=duration, dt=dt).map
    assert np.allclose(out[0], ref1, atol=1e-4)
    assert np.allclose(out[1], ref2, atol=1e-4)
    # distinct patients -> distinct trajectories
    assert not np.allclose(out[0], out[1])


def test_minutes_below_threshold_counts_seconds():
    traj = torch.tensor([[70.0, 60.0, 60.0, 80.0]], dtype=torch.float64)  # 2 s < 65
    out = te.minutes_below_threshold(traj, dt=1.0, thresh=65.0)
    assert np.allclose(out.numpy(), [2.0 / 60.0])


def test_norepi_rise_still_applied_when_infusion_is_present():
    """Guard for the all-zero norepi fast path: a real infusion must still raise MAP.

    Most SP4 cases carry no vasopressor, so the cascade is skipped when the input
    is identically zero. That shortcut must not suppress a genuine pressor effect.
    """
    pats = [Patient(age=50, weight=70, height=170, sex="M", map0=80.0)]
    T = 600
    prop = torch.zeros(1, T, dtype=torch.float64)
    zero = torch.zeros(1, T, dtype=torch.float64)
    dosed = torch.full((1, T), 10.0, dtype=torch.float64)
    d = torch.zeros(1, 6, dtype=torch.float64)

    without = te.project_map_torch(pats, prop, zero, d)
    with_pressor = te.project_map_torch(pats, prop, dosed, d)
    assert with_pressor[0, -1] > without[0, -1] + 1.0
