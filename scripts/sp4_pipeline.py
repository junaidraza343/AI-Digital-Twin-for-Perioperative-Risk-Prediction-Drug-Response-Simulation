"""SP4 non-smoke training pipeline (real or injected cohort).

Turns a cohort of cases into: per-case teacher deltas -> Stage 1 distillation of the
calibration head -> Stage 2 end-to-end fine-tune (BCE + MAP reconstruction through the
differentiable twin). The cohort is injected as a list of (caseid, static_row) plus a
loader_fn, so the whole pipeline is testable on a synthetic cohort with no VitalDB.
The GPU notebook (Task 8b) supplies the real cohort and runs this on CUDA.

Static-personalization milestone: Stage 2 reconstructs each case from the MEAN of its
per-window deltas -- the nested first step toward online delta(t).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from twin.data.vitaldb_loader import load_numeric_frame, _default_loader
from twin.data.labeling import make_map_series
from twin.data.drug_schedule import propofol_mg_min, vasopressor_ug_min
from scripts.build_dataset import build_windows_for_case
from twin.pkpd.engine import Patient
from twin.pkpd.teacher import fit_deltas_batch
from twin.pkpd import torch_engine as te
from twin.models.coupled import CoupledTwin
from twin.models.deepnet import pick_device
from scripts.train_calibration import FeaturePrep, distill_step

META_COLS = ("caseid", "t_end", "y", "category")


def feature_columns(windows_df):
    """Model input columns = window features minus label/metadata columns."""
    return [col for col in windows_df.columns if col not in META_COLS]


def load_case(caseid, static_row, loader_fn=_default_loader, device="cpu"):
    """Assemble one case: labeled window features + observed MAP + drug schedules.

    Returns a dict, or None if the case yields no prediction windows. Observed/drug
    tensors are float64 (teacher-fit accuracy); training tensors are built in float32.
    """
    windows = build_windows_for_case(caseid, static_row, loader_fn=loader_fn)
    if windows.empty:
        return None
    frame = load_numeric_frame(caseid, loader_fn=loader_fn)
    observed = make_map_series(frame)
    prop = propofol_mg_min(frame)
    norepi = vasopressor_ug_min(frame)
    valid = observed[~np.isnan(observed)]
    map0 = float(valid[0]) if valid.size else 90.0
    patient = Patient(age=float(static_row.get("age", 50)),
                      weight=float(static_row.get("weight", 70)),
                      height=float(static_row.get("height", 170)),
                      sex=str(static_row.get("sex", "M")), map0=map0)

    def t64(a):
        return torch.tensor(np.asarray(a)[None, :], dtype=torch.float64, device=device)

    return {"caseid": caseid, "windows": windows, "patient": patient,
            "observed": t64(observed), "prop": t64(prop), "norepi": t64(norepi)}


def teacher_delta(case, n_iters=200, lr=0.05, prior_lambda=1e-3):
    """Per-case teacher delta [1,6] fit through the differentiable twin."""
    delta, _ = fit_deltas_batch([case["patient"]], case["prop"], case["norepi"],
                                case["observed"], n_iters=n_iters, lr=lr,
                                prior_lambda=prior_lambda)
    return delta


def run_stage1_cohort(cohort, loader_fn=_default_loader, *, teacher_iters=200,
                      epochs=100, lr=1e-2, latent_dim=32, device=None, save_path=None):
    """Stage 1: distill the calibration head to per-case teacher deltas.

    Returns (model, prep, cols, info). info: n_cases, n_windows,
    distill_loss_first, distill_loss_last.
    """
    device = device or pick_device()
    frames, targets = [], []
    for caseid, static_row in cohort:
        case = load_case(caseid, static_row, loader_fn=loader_fn, device=device)
        if case is None:
            continue
        d = teacher_delta(case, n_iters=teacher_iters).cpu().numpy()   # [1,6]
        w = case["windows"]
        frames.append(w)
        targets.append(np.repeat(d, len(w), axis=0))
    if not frames:
        raise ValueError("No case in the cohort produced any prediction windows.")

    X_df = pd.concat(frames, ignore_index=True)
    target_delta = np.concatenate(targets, axis=0)
    cols = feature_columns(X_df)
    prep = FeaturePrep().fit(X_df[cols])
    Xt = torch.tensor(prep.transform(X_df[cols]), dtype=torch.float32, device=device)
    yt = torch.tensor(target_delta, dtype=torch.float32, device=device)

    model = CoupledTwin(n_features=len(cols), latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    first = distill_step(model, opt, Xt, yt)
    last = first
    for _ in range(max(epochs - 1, 0)):
        last = distill_step(model, opt, Xt, yt)
    info = {"n_cases": len(frames), "n_windows": int(len(X_df)),
            "distill_loss_first": first, "distill_loss_last": last}
    if save_path:
        torch.save({"state": model.state_dict(), "cols": cols, "latent_dim": latent_dim,
                    "prep": {"columns": prep.columns, "med": prep.med,
                             "mu": prep.mu, "sd": prep.sd}}, save_path)
    return model, prep, cols, info


def joint_step_case(model, opt, x, y, case, lam_recon=1.0, lam_prior=1e-3):
    """One Stage-2 step for a single case: window-level IOH BCE + case-level MAP
    reconstruction (mean of per-window deltas) through the twin + delta prior."""
    model.train()
    opt.zero_grad()
    logit, delta, _ = model(x)                        # [nw], [nw, 6]
    bce = F.binary_cross_entropy_with_logits(logit, y)

    delta_case = delta.mean(dim=0, keepdim=True)      # static per-case personalization
    prop = case["prop"].to(torch.float32)
    norepi = case["norepi"].to(torch.float32)
    pred = te.project_map_torch([case["patient"]], prop, norepi, delta_case)
    obs = case["observed"].to(torch.float32)
    mask = ~torch.isnan(obs)
    obs = torch.nan_to_num(obs, nan=0.0)
    m = min(pred.shape[1], obs.shape[1])
    err = (pred[:, :m] - obs[:, :m]) * mask[:, :m]
    recon = (err ** 2).sum() / mask[:, :m].sum().clamp(min=1.0)

    prior = (delta ** 2).sum(dim=1).mean()
    loss = bce + lam_recon * recon + lam_prior * prior
    loss.backward()
    opt.step()
    return {"loss": float(loss.detach()), "bce": float(bce.detach()),
            "recon": float(recon.detach()), "prior": float(prior.detach())}


def run_stage2_cohort(cohort, model, prep, cols, loader_fn=_default_loader, *,
                      epochs=50, lr=5e-3, lam_recon=1.0, lam_prior=1e-3,
                      device=None, save_path=None):
    """Stage 2: end-to-end fine-tune from a Stage-1 model. Returns (model, info)
    with info: n_cases, recon_first, recon_last (mean per-case recon MSE per epoch)."""
    device = device or pick_device()
    prepared = []
    for caseid, static_row in cohort:
        case = load_case(caseid, static_row, loader_fn=loader_fn, device=device)
        if case is None:
            continue
        w = case["windows"]
        x = torch.tensor(prep.transform(w[cols]), dtype=torch.float32, device=device)
        y = torch.tensor(w["y"].to_numpy(dtype=float), dtype=torch.float32, device=device)
        prepared.append((case, x, y))
    if not prepared:
        raise ValueError("No case in the cohort produced any prediction windows.")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    recon_first = recon_last = None
    for ep in range(epochs):
        recon_sum = 0.0
        for case, x, y in prepared:
            out = joint_step_case(model, opt, x, y, case, lam_recon, lam_prior)
            recon_sum += out["recon"]
        recon_mean = recon_sum / len(prepared)
        if ep == 0:
            recon_first = recon_mean
        recon_last = recon_mean
    info = {"n_cases": len(prepared), "recon_first": recon_first, "recon_last": recon_last}
    if save_path:
        torch.save({"state": model.state_dict(), "cols": cols}, save_path)
    return model, info
