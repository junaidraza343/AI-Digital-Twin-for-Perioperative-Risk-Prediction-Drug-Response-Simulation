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

    Returns (deltas detached [B,6], recon_mse float). The returned loss is the
    reconstruction MSE ONLY (not recon+prior), evaluated at the FINAL deltas -- a
    batch-size/lambda-agnostic fit-quality metric a caller can use to filter cases.
    """
    dtype, device = prop_rate.dtype, prop_rate.device
    B = prop_rate.shape[0]
    raw = torch.zeros(B, 6, dtype=dtype, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    obs = observed_map.to(dtype=dtype, device=device)
    mask = ~torch.isnan(obs)
    obs_filled = torch.nan_to_num(obs, nan=0.0)

    def recon_mse(deltas):
        pred = te.project_map_torch(patient_batch, prop_rate, norepi_rate, deltas)
        # Score the overlap only; a shorter observed vector is truncated, not an error.
        m = min(pred.shape[1], obs_filled.shape[1])
        err = (pred[:, :m] - obs_filled[:, :m]) * mask[:, :m]
        denom = mask[:, :m].sum().clamp(min=1.0)
        return (err ** 2).sum() / denom

    for _ in range(n_iters):
        opt.zero_grad()
        deltas = delta_bound * torch.tanh(raw)          # keep in (-bound, bound)
        # prior sums over the batch while recon is count-normalized; the asymmetry is
        # harmless because Adam's per-parameter step self-normalizes the magnitude.
        loss = recon_mse(deltas) + prior_lambda * (deltas ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        deltas = delta_bound * torch.tanh(raw)
        loss_val = float(recon_mse(deltas))   # recon at the FINAL deltas (see docstring)
    return deltas.detach(), loss_val


def _pad_cohort(cases):
    """Right-pad a ragged cohort into [B, T_max] tensors.

    Drug rates pad with 0 (no infusion) and observed MAP pads with NaN, which the
    masked loss already ignores -- so padding cannot contribute to any case's fit.
    """
    dtype = cases[0]["observed"].dtype
    device = cases[0]["observed"].device
    T = max(c["observed"].shape[1] for c in cases)

    def stack(key, fill):
        rows = []
        for c in cases:
            v = c[key].to(dtype=dtype, device=device)
            pad = T - v.shape[1]
            if pad:
                v = torch.cat([v, torch.full((1, pad), fill, dtype=dtype,
                                             device=device)], dim=1)
            rows.append(v)
        return torch.cat(rows, dim=0)

    return (stack("prop", 0.0), stack("norepi", 0.0),
            stack("observed", float("nan")))


def fit_deltas_cohort(cases, n_iters=300, lr=0.05, prior_lambda=1e-3,
                      delta_bound=0.7, batch_size=64):
    """Fit per-case deltas for a whole cohort, many cases at a time.

    The twin's time loop is a Python loop over T, so its cost is per-timestep and
    barely moves with batch width: fitting B cases together is ~B times the
    throughput of fitting them one by one. Cases are chunked into groups of
    `batch_size` and padded to the longest member of each chunk.

    Unlike fit_deltas_batch, the reconstruction term is normalized PER CASE and
    averaged, so each case's recon/prior balance is independent of how many cases
    share its chunk -- fitting in chunks gives the same deltas as fitting alone.

    Returns (deltas [N,6], per-case recon MSE [N]).
    """
    out_d, out_l = [], []
    for start in range(0, len(cases), batch_size):
        chunk = cases[start:start + batch_size]
        prop, norepi, obs = _pad_cohort(chunk)
        patients = [c["patient"] for c in chunk]
        dtype, device = prop.dtype, prop.device
        B = prop.shape[0]

        raw = torch.zeros(B, 6, dtype=dtype, device=device, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=lr)
        mask = ~torch.isnan(obs)
        obs_filled = torch.nan_to_num(obs, nan=0.0)

        def per_case_mse(deltas):
            pred = te.project_map_torch(patients, prop, norepi, deltas)
            m = min(pred.shape[1], obs_filled.shape[1])
            err = (pred[:, :m] - obs_filled[:, :m]) * mask[:, :m]
            denom = mask[:, :m].sum(dim=1).clamp(min=1.0)
            return (err ** 2).sum(dim=1) / denom

        for _ in range(n_iters):
            opt.zero_grad()
            deltas = delta_bound * torch.tanh(raw)
            loss = (per_case_mse(deltas).mean()
                    + prior_lambda * (deltas ** 2).sum(dim=1).mean())
            loss.backward()
            opt.step()

        with torch.no_grad():
            deltas = delta_bound * torch.tanh(raw)
            out_d.append(deltas.detach())
            out_l.append(per_case_mse(deltas))
    return torch.cat(out_d, dim=0), torch.cat(out_l, dim=0)
