"""Distillation teacher: optimize per-case deltas directly through the
differentiable twin to match observed MAP, regularized by an identifiability
prior. Returns the delta that Stage 1 trains the calibration head to predict.

This is the non-amortized oracle; the head is its amortized approximation."""
import torch

from twin.pkpd import torch_engine as te


def fit_deltas_batch(patient_batch, prop_rate, norepi_rate, observed_map,
                     n_iters=300, lr=0.05, prior_lambda=1e-3, delta_bound=0.7):
    """Adam-optimize deltas [B,6] to minimize masked MSE(MAP_hat, observed) +
    prior_lambda*||delta||^2. observed_map may contain NaN (unmeasured) -> masked.
    Returns (deltas detached [B,6], final_loss float)."""
    dtype, device = prop_rate.dtype, prop_rate.device
    B = prop_rate.shape[0]
    raw = torch.zeros(B, 6, dtype=dtype, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    obs = observed_map.to(dtype=dtype, device=device)
    mask = ~torch.isnan(obs)
    obs_filled = torch.nan_to_num(obs, nan=0.0)

    loss_val = float("inf")
    for _ in range(n_iters):
        opt.zero_grad()
        deltas = delta_bound * torch.tanh(raw)          # keep in (-bound, bound)
        pred = te.project_map_torch(patient_batch, prop_rate, norepi_rate, deltas)
        m = min(pred.shape[1], obs_filled.shape[1])
        err = (pred[:, :m] - obs_filled[:, :m]) * mask[:, :m]
        denom = mask[:, :m].sum().clamp(min=1.0)
        recon = (err ** 2).sum() / denom
        prior = prior_lambda * (deltas ** 2).sum()
        loss = recon + prior
        loss.backward()
        opt.step()
        loss_val = float(recon.detach())
    with torch.no_grad():
        deltas = delta_bound * torch.tanh(raw)
    return deltas.detach(), loss_val
