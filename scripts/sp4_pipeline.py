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

import twin.config as c
import torch.nn.functional as F

from twin.data.vitaldb_loader import load_numeric_frame, _default_loader
from twin.data.labeling import make_map_series
from twin.data.drug_schedule import propofol_mg_min, vasopressor_ug_min
from scripts.build_dataset import build_windows_for_case
from twin.pkpd.engine import Patient
from twin.pkpd.teacher import fit_deltas_batch, fit_deltas_cohort
from twin.pkpd import torch_engine as te
from twin.models.coupled import CoupledTwin
from twin.models.deepnet import pick_device
from scripts.train_calibration import FeaturePrep, distill_step
from twin.eval.metrics import (auroc, auprc, brier, ppv_at_alarm_rate,
                              expected_calibration_error)

META_COLS = ("caseid", "t_end", "y", "category")


def feature_columns(windows_df):
    """Model input columns = window features minus label/metadata columns."""
    return [col for col in windows_df.columns if col not in META_COLS]


BASELINE_SAMPLES = 60          # ~1 min of arterial samples to median over
BASELINE_FALLBACK = 90.0       # population MAP when a case has no usable baseline


def clean_map_series(observed):
    """NaN out non-physiologic arterial samples (line zeroing, flush, damping).

    Real records open with transients -- case 16 starts at 12 mmHg and spikes to
    249 -- which are not blood pressure and must not enter the reconstruction loss.
    """
    out = np.asarray(observed, dtype=float).copy()
    out[(out < c.MAP_ARTIFACT_LO) | (out > c.MAP_ARTIFACT_HI)] = np.nan
    return out


def baseline_map(observed):
    """Robust starting MAP: median of the first minute of physiologic samples.

    The first valid sample is not safe to use -- seeding the twin from an artifact
    pins its whole trajectory to the MAP_MIN clamp, where the gradient is zero and
    the teacher silently fails to fit.
    """
    valid = observed[~np.isnan(observed)]
    if valid.size == 0:
        return BASELINE_FALLBACK
    return float(np.median(valid[:BASELINE_SAMPLES]))


def load_case(caseid, static_row, loader_fn=_default_loader, device="cpu"):
    """Assemble one case: labeled window features + observed MAP + drug schedules.

    Returns a dict, or None if the case yields no prediction windows. Observed/drug
    tensors are float64 (teacher-fit accuracy); training tensors are built in float32.
    """
    windows = build_windows_for_case(caseid, static_row, loader_fn=loader_fn)
    if windows.empty:
        return None
    frame = load_numeric_frame(caseid, loader_fn=loader_fn)
    observed = clean_map_series(make_map_series(frame))
    prop = propofol_mg_min(frame)
    norepi = vasopressor_ug_min(frame)
    map0 = baseline_map(observed)
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
                      epochs=100, lr=1e-2, latent_dim=32, device=None, save_path=None,
                      teacher_batch_size=64, progress=False):
    """Stage 1: distill the calibration head to per-case teacher deltas.

    Teacher deltas are fitted a chunk of cases at a time (see fit_deltas_cohort):
    the twin's cost is per-timestep, so a chunk of B cases costs about the same as
    one. Chunks are loaded, fitted, then released, so peak memory tracks the chunk
    rather than the cohort.

    Returns (model, prep, cols, info). info: n_cases, n_windows,
    distill_loss_first, distill_loss_last.
    """
    device = device or pick_device()
    frames, targets = [], []
    for start in range(0, len(cohort), teacher_batch_size):
        chunk, loaded = cohort[start:start + teacher_batch_size], []
        for caseid, static_row in chunk:
            try:  # one bad case (missing track, pull failure) must not abort the run
                case = load_case(caseid, static_row, loader_fn=loader_fn, device=device)
                if case is not None:
                    loaded.append(case)
            except Exception as exc:
                print(f"  skip case {caseid}: {exc}")
        if not loaded:
            continue
        deltas, _ = fit_deltas_cohort(loaded, n_iters=teacher_iters,
                                      batch_size=len(loaded))
        deltas = deltas.cpu().numpy()
        for case, d in zip(loaded, deltas):
            w = case["windows"]
            frames.append(w)
            targets.append(np.repeat(d[None, :], len(w), axis=0))
        if progress:
            print(f"  stage1: {len(frames)} cases fitted", flush=True)
        del loaded                      # release the padded [B,T] tensors
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
    recon = ((err ** 2).sum() / mask[:, :m].sum().clamp(min=1.0)
             / (c.MAP_RECON_SCALE ** 2))          # same normalization as the batch path

    prior = (delta ** 2).sum(dim=1).mean()
    loss = bce + lam_recon * recon + lam_prior * prior
    loss.backward()
    opt.step()
    return {"loss": float(loss.detach()), "bce": float(bce.detach()),
            "recon": float(recon.detach()), "prior": float(prior.detach())}


def joint_step_batch(model, opt, batch, lam_recon=1.0, lam_prior=1e-3):
    """One Stage-2 step over a MINIBATCH of cases.

    Windows from every case in the batch share one BCE term, and the twin is run
    once for the whole batch (padded to the longest record) so reconstruction costs
    one simulation per step instead of one per case. Each case is still
    reconstructed from the mean of its OWN windows' deltas -- the static
    personalization milestone -- and recon is averaged per case so the loss scale
    does not drift with batch width.
    """
    model.train()
    opt.zero_grad()
    xs = torch.cat([x for _, x, _ in batch], dim=0)
    ys = torch.cat([y for _, _, y in batch], dim=0)
    logit, delta, _ = model(xs)
    bce = F.binary_cross_entropy_with_logits(logit, ys)

    sizes = [x.shape[0] for _, x, _ in batch]
    delta_case = torch.stack([d.mean(dim=0) for d in torch.split(delta, sizes)])

    cases = [case for case, _, _ in batch]
    prop, norepi, obs = _pad_stage2(cases, delta_case.dtype)
    pred = te.project_map_torch([c["patient"] for c in cases], prop, norepi, delta_case)
    mask = ~torch.isnan(obs)
    obs = torch.nan_to_num(obs, nan=0.0)
    m = min(pred.shape[1], obs.shape[1])
    err = (pred[:, :m] - obs[:, :m]) * mask[:, :m]
    recon_per_case = (err ** 2).sum(dim=1) / mask[:, :m].sum(dim=1).clamp(min=1.0)
    # Normalized to MAP_RECON_SCALE^2 so this term is commensurate with the BCE
    # above rather than drowning it (see twin/config.py).
    recon = recon_per_case.mean() / (c.MAP_RECON_SCALE ** 2)

    prior = (delta ** 2).sum(dim=1).mean()
    loss = bce + lam_recon * recon + lam_prior * prior
    loss.backward()
    opt.step()
    return {"loss": float(loss.detach()), "bce": float(bce.detach()),
            "recon": float(recon.detach()), "prior": float(prior.detach()),
            "n_cases": len(batch)}


def _pad_stage2(cases, dtype):
    """Pad a Stage-2 case batch to [B, T_max] (rates with 0, observed with NaN)."""
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

    return stack("prop", 0.0), stack("norepi", 0.0), stack("observed", float("nan"))


def run_stage2_cohort(cohort, model, prep, cols, loader_fn=_default_loader, *,
                      epochs=50, lr=5e-3, lam_recon=1.0, lam_prior=1e-3,
                      device=None, save_path=None, case_batch_size=16,
                      progress=False):
    """Stage 2: end-to-end fine-tune from a Stage-1 model. Returns (model, info)
    with info: n_cases, recon_first, recon_last (mean per-case recon MSE per epoch)."""
    # Follow the model's own device so Stage 2 never mismatches a Stage-1 model
    # that was trained on a different (explicitly overridden) device.
    device = device or next(model.parameters()).device
    prepared = []
    for caseid, static_row in cohort:
        try:  # skip-and-log, same fault tolerance as Stage 1
            case = load_case(caseid, static_row, loader_fn=loader_fn, device=device)
            if case is None:
                continue
            w = case["windows"]
            x = torch.tensor(prep.transform(w[cols]), dtype=torch.float32, device=device)
            y = torch.tensor(w["y"].to_numpy(dtype=float), dtype=torch.float32, device=device)
        except Exception as exc:
            print(f"  skip case {caseid}: {exc}")
            continue
        prepared.append((case, x, y))
    if not prepared:
        raise ValueError("No case in the cohort produced any prediction windows.")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    recon_first = recon_last = None
    for ep in range(epochs):
        recon_sum = n_steps = 0
        for start in range(0, len(prepared), case_batch_size):
            batch = prepared[start:start + case_batch_size]
            out = joint_step_batch(model, opt, batch, lam_recon, lam_prior)
            recon_sum += out["recon"]
            n_steps += 1
        recon_mean = recon_sum / max(n_steps, 1)   # each step already averages its batch
        if ep == 0:
            recon_first = recon_mean
        recon_last = recon_mean
    info = {"n_cases": len(prepared), "recon_first": recon_first, "recon_last": recon_last}
    if save_path:
        # include prep so a Stage-2 checkpoint is self-contained (can preprocess features)
        torch.save({"state": model.state_dict(), "cols": cols,
                    "prep": {"columns": prep.columns, "med": prep.med,
                             "mu": prep.mu, "sd": prep.sd}}, save_path)
    return model, info


def evaluate_cohort(model, prep, cols, cohort, loader_fn=_default_loader, device=None):
    """Evaluate a trained coupled twin on cases it was not trained on.

    Runs the model forward only -- no teacher fit -- because on a new patient no
    fitted delta exists; that is the whole point of amortizing the teacher into a
    calibration head. Reports window-level IOH discrimination/calibration and, per
    case, the MAP reconstruction of the head's delta against the population twin
    (delta = 0), which is the personalization claim.
    """
    device = device or next(model.parameters()).device
    model.eval()
    ys, ps, pop_rmse, per_rmse, n_windows = [], [], [], [], 0
    for caseid, static_row in cohort:
        try:
            case = load_case(caseid, static_row, loader_fn=loader_fn, device=device)
        except Exception as exc:
            print(f"  skip case {caseid}: {exc}")
            continue
        if case is None:
            continue
        w = case["windows"]
        x = torch.tensor(prep.transform(w[cols]), dtype=torch.float32, device=device)
        with torch.no_grad():
            logit, delta, _ = model(x)
            prob = torch.sigmoid(logit).cpu().numpy()
            delta_case = delta.mean(dim=0, keepdim=True).to(torch.float64)
            zero = torch.zeros_like(delta_case)
            obs = case["observed"]
            mask = ~torch.isnan(obs)
            filled = torch.nan_to_num(obs, nan=0.0)

            def rmse(d):
                pred = te.project_map_torch([case["patient"]], case["prop"],
                                            case["norepi"], d)
                m = min(pred.shape[1], filled.shape[1])
                err = (pred[:, :m] - filled[:, :m]) * mask[:, :m]
                mse = (err ** 2).sum() / mask[:, :m].sum().clamp(min=1.0)
                return float(mse.sqrt())

            pop_rmse.append(rmse(zero))
            per_rmse.append(rmse(delta_case))
        ys.append(w["y"].to_numpy(dtype=float))
        ps.append(prob)
        n_windows += len(w)

    if not ys:
        raise ValueError("No held-out case produced any prediction windows.")
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pop_rmse = np.asarray(pop_rmse)
    per_rmse = np.asarray(per_rmse)
    return {
        "n_cases": len(pop_rmse), "n_windows": int(n_windows),
        "prevalence": float(y.mean()),
        "auroc": float(auroc(y, p)), "auprc": float(auprc(y, p)),
        "ppv": float(ppv_at_alarm_rate(y, p)),
        "ece": float(expected_calibration_error(y, p)), "brier": float(brier(y, p)),
        "population_rmse": float(pop_rmse.mean()),
        "personalized_rmse": float(per_rmse.mean()),
        "improved_case_count": int((per_rmse < pop_rmse).sum()),
    }
