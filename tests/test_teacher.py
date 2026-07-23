"""Tests for the SP4 distillation teacher.

The teacher's contract is trajectory RECONSTRUCTION: optimize per-case deltas
through the differentiable twin so the projected MAP matches the observed MAP.
Exact recovery of a specific parameter (e.g. EC50) is deliberately NOT asserted --
with all 6 deltas free the model is degenerate (other deltas compensate), which is
the project's documented identifiability limitation. Distillation only needs the
teacher to reconstruct the observed trajectory; the head learns whatever deltas the
teacher emits and Stage 2 fine-tunes on reconstruction anyway.
"""
import pytest
pytest.importorskip("torch")
import torch

from twin.pkpd.engine import Patient
from twin.pkpd.schedule import rate_vector
from twin.pkpd import torch_engine as te
from twin.pkpd.teacher import fit_deltas_batch


def _case(dur=300.0):
    patient = Patient(age=50, weight=70, height=170, sex="M", map0=90.0)
    prop = rate_vector([(0, 60, 200.0), (60, int(dur), 20.0)], 1.0, dur)
    norepi = rate_vector([(120, 121, 8.0)], 1.0, dur)
    prop_t = torch.tensor(prop[None, :], dtype=torch.float64)
    norepi_t = torch.tensor(norepi[None, :], dtype=torch.float64)
    return patient, prop_t, norepi_t


def test_teacher_reconstructs_and_improves_over_population():
    """Fitted deltas reconstruct an observed MAP far better than the population twin."""
    patient, prop, norepi = _case()
    true_delta = torch.zeros(1, 6, dtype=torch.float64)
    true_delta[0, 4] = 0.4  # a real (EC50-driven) personalization to reconstruct
    observed = te.project_map_torch([patient], prop, norepi, true_delta).detach()

    # population (zero-delta) reconstruction error, for reference
    zero = torch.zeros(1, 6, dtype=torch.float64)
    base_mse = float(((te.project_map_torch([patient], prop, norepi, zero) - observed) ** 2).mean())

    fitted, loss = fit_deltas_batch([patient], prop, norepi, observed,
                                    n_iters=200, lr=0.05, prior_lambda=1e-3)
    fit_mse = float(((te.project_map_torch([patient], prop, norepi, fitted) - observed) ** 2).mean())

    assert fit_mse < 1.0                 # reconstructs to well under 1 mmHg RMSE
    assert fit_mse < 0.1 * base_mse      # and is a large improvement over population
    assert loss == pytest.approx(fit_mse, abs=0.05)  # returned loss is the recon MSE


def test_prior_keeps_deltas_small_when_no_personalization_needed():
    """When the observed MAP already equals the population twin, the prior keeps
    every delta near zero (nothing to personalize)."""
    patient, prop, norepi = _case()
    observed = te.project_map_torch([patient], prop, norepi,
                                    torch.zeros(1, 6, dtype=torch.float64)).detach()
    fitted, _ = fit_deltas_batch([patient], prop, norepi, observed,
                                 n_iters=100, lr=0.05, prior_lambda=1e-2)
    assert float(fitted.abs().max()) < 0.1
