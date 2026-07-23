"""Differentiable, batched PyTorch reimplementation of the numpy PK-PD engine.

Mirrors twin/pkpd/{propofol,pd_model,norepi}.py exactly at delta=0 (explicit Euler,
per-minute rate constants converted with dt/60). Gradients flow deltas -> params ->
MAP, enabling the SP4 calibration head to be trained by backprop through the twin.

deltas order: (V1, V2, V3, ke0, EC50, gamma); theta = theta_pop * exp(delta).
Runs on the device of the input tensors (CUDA/MPS/CPU)."""
import torch

import twin.config as c
from twin.pkpd.params import POP_PROPOFOL, POP_PD, POP_NOREPI
from twin.pkpd.covariates import scale_propofol
from twin.pkpd.pd_model import MAP_MIN, MAP_MAX


def _patient_propofol_params(patient_batch, dtype, device):
    """Covariate-scale propofol params per patient (numpy) -> [B]-tensors."""
    keys = ("V1", "V2", "V3", "CL", "Q2", "Q3", "ke0")
    cols = {k: [] for k in keys}
    for p in patient_batch:
        sp = scale_propofol(POP_PROPOFOL, p.age, p.weight, p.height, p.sex)
        for k in keys:
            cols[k].append(getattr(sp, k))
    return {k: torch.tensor(v, dtype=dtype, device=device) for k, v in cols.items()}


def project_map_torch(patient_batch, prop_rate, norepi_rate, deltas, dt=1.0):
    dtype, device = prop_rate.dtype, prop_rate.device
    B, T = prop_rate.shape
    d = deltas
    base = _patient_propofol_params(patient_batch, dtype, device)

    # Apply multiplicative deltas (exp) to propofol PK + PD params.
    V1 = base["V1"] * torch.exp(d[:, 0])
    V2 = base["V2"] * torch.exp(d[:, 1])
    V3 = base["V3"] * torch.exp(d[:, 2])
    ke0 = base["ke0"] * torch.exp(d[:, 3])
    CL, Q2, Q3 = base["CL"], base["Q2"], base["Q3"]
    ec50 = torch.tensor(POP_PD.ec50, dtype=dtype, device=device) * torch.exp(d[:, 4])
    gamma = torch.tensor(POP_PD.gamma, dtype=dtype, device=device) * torch.exp(d[:, 5])
    emax = torch.tensor(POP_PD.emax, dtype=dtype, device=device)

    k10 = CL / V1
    k12 = Q2 / V1
    k21 = Q2 / V2
    k13 = Q3 / V1
    k31 = Q3 / V3
    step = dt / 60.0

    a1 = torch.zeros(B, dtype=dtype, device=device)
    a2 = torch.zeros(B, dtype=dtype, device=device)
    a3 = torch.zeros(B, dtype=dtype, device=device)
    ce_prev = torch.zeros(B, dtype=dtype, device=device)
    ce_list = []
    for i in range(T):
        infusion = prop_rate[:, i] * step
        da1 = (-(k10 + k12 + k13) * a1 + k21 * a2 + k31 * a3) * step + infusion
        da2 = (k12 * a1 - k21 * a2) * step
        da3 = (k13 * a1 - k31 * a3) * step
        a1 = a1 + da1
        a2 = a2 + da2
        a3 = a3 + da3
        cp = a1 / V1
        ce_prev = ce_prev + ke0 * step * (cp - ce_prev)
        ce_list.append(ce_prev)
    ce = torch.stack(ce_list, dim=1)  # [B, T]

    # PD reduction (sigmoid Emax)
    ce_g = torch.clamp(ce, min=0.0) ** gamma[:, None]
    reduction = emax * ce_g / (ec50[:, None] ** gamma[:, None] + ce_g)

    # Norepinephrine rise (two-stage cascade), population params (no delta).
    k = 1.0 / POP_NOREPI.tpeak_s
    plasma = torch.zeros(B, dtype=dtype, device=device)
    effect = torch.zeros(B, dtype=dtype, device=device)
    conc_list = []
    for i in range(T):
        inp = norepi_rate[:, i] / 60.0
        plasma = plasma + dt * (inp - k * plasma)
        effect = effect + dt * k * (plasma - effect)
        conc_list.append(effect)
    conc = torch.stack(conc_list, dim=1)
    conc_g = torch.clamp(conc, min=0.0) ** POP_NOREPI.gamma
    rise = POP_NOREPI.dmap_max * conc_g / (POP_NOREPI.ec50 ** POP_NOREPI.gamma + conc_g)

    map0 = torch.tensor([p.map0 for p in patient_batch], dtype=dtype, device=device)
    traj = map0[:, None] * (1.0 - reduction) + map0[:, None] * rise
    return torch.clamp(traj, MAP_MIN, MAP_MAX)


def minutes_below_threshold(map_traj, dt=1.0, thresh=None):
    thresh = c.MAP_THRESHOLD if thresh is None else thresh
    return (map_traj < thresh).to(map_traj.dtype).sum(dim=1) * dt / 60.0
