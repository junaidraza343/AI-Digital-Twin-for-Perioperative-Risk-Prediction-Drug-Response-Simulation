"""SP4 evaluation: does personalization improve the twin, and is it identifiable?

- per_patient_map_rmse: masked RMSE of projected vs observed MAP, per patient.
- delta_identifiability: per-delta spread across the cohort (which delta move).
- whatif_monotonic: sanity that more propofol -> more projected hypotension."""
import numpy as np
import torch

from twin.pkpd import torch_engine as te
from twin.pkpd.schedule import rate_vector


def per_patient_map_rmse(patient_batch, prop_rate, norepi_rate, observed_map, deltas):
    """Masked RMSE (mmHg) of projected vs observed MAP, one value per patient.

    .cpu() before .numpy() so this works when training/eval ran on CUDA (Task 8).
    """
    pred = te.project_map_torch(patient_batch, prop_rate, norepi_rate,
                                deltas.to(prop_rate.dtype))
    obs = observed_map.to(pred.dtype)
    mask = ~torch.isnan(obs)
    obs = torch.nan_to_num(obs, nan=0.0)
    m = min(pred.shape[1], obs.shape[1])
    err = (pred[:, :m] - obs[:, :m]) * mask[:, :m]
    mse = (err ** 2).sum(dim=1) / mask[:, :m].sum(dim=1).clamp(min=1.0)
    return torch.sqrt(mse).detach().cpu().numpy()


def delta_identifiability(deltas):
    """Std of each delta across the cohort; large => data moves that delta."""
    return deltas.detach().float().cpu().std(dim=0).numpy()


def whatif_monotonic(patient, T=300):
    """More propofol maintenance -> at least as many minutes below threshold."""
    lo = torch.tensor(rate_vector([(0, 60, 200.0), (60, T, 10.0)], 1.0,
                      float(T - 1))[None, :], dtype=torch.float64)
    hi = torch.tensor(rate_vector([(0, 60, 200.0), (60, T, 40.0)], 1.0,
                      float(T - 1))[None, :], dtype=torch.float64)
    norepi = torch.zeros(1, lo.shape[1], dtype=torch.float64)
    d = torch.zeros(1, 6, dtype=torch.float64)
    map_lo = te.project_map_torch([patient], lo, norepi, d)
    map_hi = te.project_map_torch([patient], hi, norepi, d)
    below_lo = float(te.minutes_below_threshold(map_lo)[0])
    below_hi = float(te.minutes_below_threshold(map_hi)[0])
    return below_hi >= below_lo
