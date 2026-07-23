import pytest
pytest.importorskip("torch")
import torch
from twin.pkpd.engine import Patient
from twin.pkpd.schedule import rate_vector
from twin.pkpd import torch_engine as te
from twin.models.coupled import CoupledTwin
from scripts.train_calibration_stage2 import joint_step


def _batch(B=4, T=300):
    patients = [Patient(age=50, weight=70, height=170, sex="M", map0=90.0)] * B
    prop = torch.tensor(
        rate_vector([(0, 60, 200.0), (60, T, 20.0)], 1.0, float(T - 1))[None, :],
        dtype=torch.float64).repeat(B, 1)
    norepi = torch.tensor(
        rate_vector([(120, 121, 8.0)], 1.0, float(T - 1))[None, :],
        dtype=torch.float64).repeat(B, 1)
    # observed MAP generated with a known EC50 delta so recon is learnable
    d = torch.zeros(B, 6, dtype=torch.float64); d[:, 4] = 0.3
    observed = te.project_map_torch(patients, prop, norepi, d).detach()
    x = torch.randn(B, 6, dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    return patients, prop, norepi, observed, x, y


def test_joint_step_reduces_recon_loss():
    torch.manual_seed(0)
    patients, prop, norepi, observed, x, y = _batch()
    model = CoupledTwin(n_features=6, latent_dim=16).double()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    first = joint_step(model, opt, x, y, patients, prop, norepi, observed,
                       lam_recon=1.0, lam_prior=1e-3)
    for _ in range(150):
        last = joint_step(model, opt, x, y, patients, prop, norepi, observed,
                          lam_recon=1.0, lam_prior=1e-3)
    assert last["recon"] < first["recon"]
