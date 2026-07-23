"""SP4 Stage 2: end-to-end fine-tune of the coupled twin.

Loss = BCE(IOH logit, y) + lam_recon * masked_MSE(MAP_hat, observed)
       + lam_prior * ||delta||^2 , backprop through the differentiable twin."""
import torch
import torch.nn as nn

from twin.models.deepnet import pick_device
from twin.pkpd import torch_engine as te


def joint_step(model, opt, x, y, patient_batch, prop_rate, norepi_rate,
               observed_map, lam_recon=1.0, lam_prior=1e-3):
    """One joint optimization step. Returns dict of loss components (floats)."""
    model.train()
    opt.zero_grad()
    logit, delta, _ = model(x)

    bce = nn.functional.binary_cross_entropy_with_logits(logit, y.to(logit.dtype))

    pred = te.project_map_torch(patient_batch, prop_rate, norepi_rate,
                                delta.to(prop_rate.dtype))
    obs = observed_map.to(pred.dtype)
    mask = ~torch.isnan(obs)
    obs = torch.nan_to_num(obs, nan=0.0)
    m = min(pred.shape[1], obs.shape[1])
    err = (pred[:, :m] - obs[:, :m]) * mask[:, :m]
    recon = (err ** 2).sum() / mask[:, :m].sum().clamp(min=1.0)

    prior = (delta ** 2).sum(dim=1).mean()
    loss = bce + lam_recon * recon + lam_prior * prior
    loss.backward(); opt.step()
    return {"loss": float(loss.detach()), "bce": float(bce.detach()),
            "recon": float(recon.detach()), "prior": float(prior.detach())}


def run_stage2(smoke=False):
    device = pick_device()
    if smoke:
        from twin.pkpd.engine import Patient
        from twin.pkpd.schedule import rate_vector
        from twin.models.coupled import CoupledTwin
        B, T = 4, 200
        patients = [Patient(50, 70, 170, "M", 90.0)] * B
        prop = torch.tensor(rate_vector([(0, 60, 200.0), (60, T, 20.0)], 1.0,
                            float(T - 1))[None, :], dtype=torch.float64).repeat(B, 1)
        norepi = torch.tensor(rate_vector([(120, 121, 8.0)], 1.0,
                              float(T - 1))[None, :], dtype=torch.float64).repeat(B, 1)
        d = torch.zeros(B, 6, dtype=torch.float64); d[:, 4] = 0.3
        observed = te.project_map_torch(patients, prop, norepi, d).detach()
        x = torch.randn(B, 6, dtype=torch.float64)
        y = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        model = CoupledTwin(n_features=6, latent_dim=16).double()
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        for _ in range(50):
            out = joint_step(model, opt, x, y, patients, prop, norepi, observed)
        print(f"[stage2-smoke] {out}")
        return model
    raise NotImplementedError(
        "Full Stage 2 needs the SP4 cohort + Stage 1 init; run on the GPU harness.")
