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
                                    n_iters=100, lr=0.05, prior_lambda=1e-3)
    fit_mse = float(((te.project_map_torch([patient], prop, norepi, fitted) - observed) ** 2).mean())

    assert fit_mse < 1.0                 # reconstructs to well under 1 mmHg RMSE
    assert fit_mse < 0.1 * base_mse      # and is a large improvement over population
    assert loss == pytest.approx(fit_mse, abs=1e-6)  # returned loss = recon at final deltas


def test_masked_observed_ignores_nan_samples():
    """NaN samples in observed_map are masked out: the fit still reconstructs the
    valid portion and never leaks NaN into the returned deltas or loss."""
    patient, prop, norepi = _case()
    true_delta = torch.zeros(1, 6, dtype=torch.float64)
    true_delta[0, 4] = 0.4
    observed = te.project_map_torch([patient], prop, norepi, true_delta).detach().clone()
    observed[0, 150:] = float("nan")   # second half unmeasured

    fitted, loss = fit_deltas_batch([patient], prop, norepi, observed,
                                    n_iters=100, lr=0.05, prior_lambda=1e-3)
    assert not torch.isnan(fitted).any()
    assert loss == loss and loss < 1.0             # loss is finite (not NaN) and small
    # reconstruction is scored only on the observed (non-NaN) portion
    pred = te.project_map_torch([patient], prop, norepi, fitted).detach()
    valid_mse = float(((pred[0, :150] - observed[0, :150]) ** 2).mean())
    assert valid_mse < 1.0


def test_prior_keeps_deltas_small_when_no_personalization_needed():
    """When the observed MAP already equals the population twin, the prior keeps
    every delta near zero (nothing to personalize)."""
    patient, prop, norepi = _case()
    observed = te.project_map_torch([patient], prop, norepi,
                                    torch.zeros(1, 6, dtype=torch.float64)).detach()
    fitted, _ = fit_deltas_batch([patient], prop, norepi, observed,
                                 n_iters=100, lr=0.05, prior_lambda=1e-2)
    assert float(fitted.abs().max()) < 0.1


def _ragged_case(T, prop_level, map0, seed):
    """One synthetic case of length T with its own propofol level and baseline."""
    g = torch.Generator().manual_seed(seed)
    patient = Patient(age=50, weight=70, height=170, sex="M", map0=map0)
    prop = torch.full((1, T), float(prop_level), dtype=torch.float64)
    nore = torch.zeros((1, T), dtype=torch.float64)
    true_d = torch.zeros(1, 6, dtype=torch.float64)
    true_d[0, 4] = 0.3   # perturb EC50 so there is something to recover
    obs = te.project_map_torch([patient], prop, nore, true_d)
    obs = obs + 0.5 * torch.randn(obs.shape, generator=g, dtype=torch.float64)
    return {"patient": patient, "prop": prop, "norepi": nore, "observed": obs}


def test_cohort_fit_matches_per_case_fit_on_ragged_lengths():
    """Padding shorter cases into one batch must not change their fitted deltas.

    Cases have different durations, so a cohort fit has to pad and mask. If the
    padding leaked into the loss, the batched deltas would drift from the
    per-case ones.
    """
    from twin.pkpd.teacher import fit_deltas_cohort

    cases = [_ragged_case(400, 40.0, 92.0, 0),
             _ragged_case(900, 70.0, 85.0, 1),
             _ragged_case(650, 55.0, 88.0, 2)]

    solo = torch.cat([fit_deltas_batch([c["patient"]], c["prop"], c["norepi"],
                                       c["observed"], n_iters=120)[0]
                      for c in cases], dim=0)
    batched, _ = fit_deltas_cohort(cases, n_iters=120)

    assert batched.shape == (3, 6)
    assert torch.allclose(batched, solo, atol=5e-3), (batched - solo).abs().max()


def test_cohort_fit_respects_batch_size_chunking():
    """Chunking the cohort must not change the result for any case."""
    from twin.pkpd.teacher import fit_deltas_cohort

    cases = [_ragged_case(300, 40.0, 92.0, 3),
             _ragged_case(500, 60.0, 86.0, 4),
             _ragged_case(400, 50.0, 90.0, 5)]
    one_go, _ = fit_deltas_cohort(cases, n_iters=100, batch_size=8)
    chunked, _ = fit_deltas_cohort(cases, n_iters=100, batch_size=2)
    assert torch.allclose(one_go, chunked, atol=5e-3)
