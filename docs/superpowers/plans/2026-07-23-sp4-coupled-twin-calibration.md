# SP4 — Coupled Twin + Calibration Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a learned calibration head that personalizes a differentiable PK-PD twin online, coupled to the IOH predictor through a shared latent state, trained on GPU via distillation then end-to-end fine-tuning.

**Architecture:** A shared encoder `E` maps per-window features to a latent `z_t`; a prediction head emits the IOH logit and a calibration head emits 6 multiplicative δ that perturb a differentiable PyTorch PK-PD engine. Stage 1 distills the head from a per-case teacher (δ optimized directly through the twin); Stage 2 fine-tunes end-to-end on MAP reconstruction + IOH BCE + an identifiability prior.

**Tech Stack:** Python, PyTorch (device-agnostic via existing `pick_device()`), numpy/pandas, pytest, VitalDB. Builds on `twin/pkpd/` (numpy engine), `twin/models/deepnet.py` (torch patterns), `twin/data/` (loader/features).

**Spec:** `docs/superpowers/specs/2026-07-23-sp4-coupled-twin-calibration-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `twin/config.py` (modify) | Add drug + continuous-ART tracks and concentration constants |
| `twin/data/drug_schedule.py` (create) | Convert cached drug-rate columns → mg/min & µg/min vectors |
| `scripts/measure_drug_coverage.py` (create) | Measure how many cohort cases have propofol infusion + continuous ART; write SP4 cohort list |
| `twin/pkpd/torch_engine.py` (create) | Differentiable, batched PyTorch PK-PD twin |
| `twin/pkpd/teacher.py` (create) | Per-case δ optimizer (the distillation teacher) |
| `twin/models/coupled.py` (create) | Encoder `E` + prediction head `P` + calibration head `C` |
| `scripts/train_calibration.py` (create) | Stage 1 (distill) + Stage 2 (fine-tune) training entrypoint |
| `twin/eval/personalization.py` (create) | MAP-RMSE / identifiability / coupling / what-if evaluation |
| `notebooks/train_calibration_gpu.ipynb` (create) | Kaggle/Colab GPU harness driving the training script |

Test files mirror these under `tests/`.

**Conventions (match existing code):**
- Loaders follow the `loader_fn(caseid, tracks, interval) -> np.ndarray` signature (see `scripts/build_dataset.py`); columns are in `tracks` order. Tests inject a fake loader.
- Run tests with `python -m pytest` from repo root (`pytest.ini` present).
- Torch is optional at import time — guard with the `try: import torch` pattern from `twin/models/deepnet.py`; torch-dependent tests `pytest.importorskip("torch")`.

---

## Task 1: Drug-track cohort (data prerequisite)

**Files:**
- Modify: `twin/config.py`
- Create: `twin/data/drug_schedule.py`
- Create: `scripts/measure_drug_coverage.py`
- Test: `tests/test_drug_schedule.py`, `tests/test_measure_drug_coverage_smoke.py`

- [ ] **Step 1: Add drug tracks + constants to config**

Edit `twin/config.py`. After the existing `NUMERIC_TRACKS` block add:

```python
# Drug infusion tracks (Orchestra pumps) — needed by the PK-PD twin (SP4)
PROPOFOL_TRACK = "Orchestra/PPF20_RATE"       # mL/h of 20 mg/mL propofol
PHENYLEPHRINE_TRACK = "Orchestra/PHEN_RATE"   # mL/h of phenylephrine
NOREPI_TRACK = "Orchestra/NEPI_RATE"          # mL/h of norepinephrine
DRUG_TRACKS = [PROPOFOL_TRACK, PHENYLEPHRINE_TRACK, NOREPI_TRACK]

# Solution concentrations for rate conversion (documented approximations)
PPF20_MG_PER_ML = 20.0
PHEN_UG_PER_ML = 100.0
NEPI_UG_PER_ML = 20.0

# SP4 cohort selection thresholds
SP4_MIN_PROPOFOL_MINUTES = 5.0    # >=5 min of nonzero propofol infusion
SP4_MIN_ART_FRACTION = 0.5        # >=50% of samples have continuous ART_MBP
```

Then extend `NUMERIC_TRACKS` by appending `+ DRUG_TRACKS` on its own line right after the list literal:

```python
NUMERIC_TRACKS = NUMERIC_TRACKS + DRUG_TRACKS
```

> Note: `twin/data/vitaldb_loader._tracks_tag()` hashes `NUMERIC_TRACKS`, so this changes the cache filename tag — new tracks re-pull into *new* parquet files; the old vitals-only cache is untouched (no clobber).

- [ ] **Step 2: Write the failing test for drug_schedule**

Create `tests/test_drug_schedule.py`:

```python
import numpy as np
import pandas as pd
import twin.config as c
from twin.data.drug_schedule import propofol_mg_min, vasopressor_ug_min


def _frame(n=10, **cols):
    base = {t: np.full(n, np.nan) for t in c.NUMERIC_TRACKS}
    base.update(cols)
    return pd.DataFrame(base)


def test_propofol_mg_min_converts_ml_per_hour():
    # 60 mL/h of 20 mg/mL propofol = 60*20/60 = 20 mg/min
    f = _frame(n=5, **{c.PROPOFOL_TRACK: np.full(5, 60.0)})
    out = propofol_mg_min(f)
    assert out.shape == (5,)
    assert np.allclose(out, 20.0)


def test_propofol_nan_becomes_zero():
    f = _frame(n=4, **{c.PROPOFOL_TRACK: [np.nan, 60.0, np.nan, 0.0]})
    out = propofol_mg_min(f)
    assert np.allclose(out, [0.0, 20.0, 0.0, 0.0])


def test_vasopressor_sums_phen_and_nepi_in_ug_min():
    # 60 mL/h PHEN @100 ug/mL = 100 ug/min ; 60 mL/h NEPI @20 = 20 ug/min ; sum=120
    f = _frame(n=3, **{c.PHENYLEPHRINE_TRACK: np.full(3, 60.0),
                       c.NOREPI_TRACK: np.full(3, 60.0)})
    out = vasopressor_ug_min(f)
    assert np.allclose(out, 120.0)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_drug_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError: twin.data.drug_schedule`.

- [ ] **Step 4: Implement drug_schedule**

Create `twin/data/drug_schedule.py`:

```python
"""Convert cached Orchestra drug-rate tracks (mL/h) into engine input vectors.

Propofol -> mg/min (drives the PK-PD twin's central compartment).
Vasopressors (phenylephrine + norepinephrine) -> combined ug/min norepi-equivalent.
NaN (pump off / not charted) -> 0. Documented concentration approximations in config.
"""
import numpy as np
import pandas as pd
import twin.config as c


def _ml_per_hour_to_per_min(frame: pd.DataFrame, track: str) -> np.ndarray:
    if track not in frame:
        return np.zeros(len(frame))
    v = frame[track].to_numpy(dtype=float)
    v = np.nan_to_num(v, nan=0.0)
    return v / 60.0  # mL/h -> mL/min


def propofol_mg_min(frame: pd.DataFrame) -> np.ndarray:
    return _ml_per_hour_to_per_min(frame, c.PROPOFOL_TRACK) * c.PPF20_MG_PER_ML


def vasopressor_ug_min(frame: pd.DataFrame) -> np.ndarray:
    phen = _ml_per_hour_to_per_min(frame, c.PHENYLEPHRINE_TRACK) * c.PHEN_UG_PER_ML
    nepi = _ml_per_hour_to_per_min(frame, c.NOREPI_TRACK) * c.NEPI_UG_PER_ML
    return phen + nepi
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_drug_schedule.py -v`
Expected: 3 passed.

- [ ] **Step 6: Write the failing test for cohort selection + coverage script**

Create `tests/test_measure_drug_coverage_smoke.py`:

```python
import numpy as np
import pandas as pd
import twin.config as c
from scripts.measure_drug_coverage import case_qualifies, measure_coverage


def _arr(n, propofol=0.0, art=True):
    """Build a (n, len(NUMERIC_TRACKS)) array in NUMERIC_TRACKS column order."""
    cols = c.NUMERIC_TRACKS
    a = np.full((n, len(cols)), np.nan)
    a[:, cols.index(c.PROPOFOL_TRACK)] = propofol
    if art:
        a[:, cols.index(c.MAP_TRACK)] = 80.0
    return a


def test_case_qualifies_true_with_propofol_and_art():
    frame = pd.DataFrame(_arr(600, propofol=60.0, art=True), columns=c.NUMERIC_TRACKS)
    ok, reason = case_qualifies(frame)
    assert ok is True


def test_case_rejected_without_propofol():
    frame = pd.DataFrame(_arr(600, propofol=0.0, art=True), columns=c.NUMERIC_TRACKS)
    ok, reason = case_qualifies(frame)
    assert ok is False and "propofol" in reason


def test_measure_coverage_counts_qualifying_cases():
    eligible = pd.DataFrame({"caseid": [1, 2]})

    def fake_loader(caseid, tracks, interval):
        return _arr(600, propofol=60.0 if caseid == 1 else 0.0, art=True)

    summary = measure_coverage(eligible, loader_fn=fake_loader)
    assert summary["n_qualifying"] == 1
    assert 1 in summary["caseids"]
```

- [ ] **Step 7: Run it to verify it fails**

Run: `python -m pytest tests/test_measure_drug_coverage_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.measure_drug_coverage`.

- [ ] **Step 8: Implement the coverage script**

Create `scripts/measure_drug_coverage.py`:

```python
"""Measure SP4 cohort coverage: which eligible cases have propofol infusion +
continuous ART. Writes results/sp4_cohort.csv (qualifying caseids) and prints a
coverage summary. Re-pull is a manual VitalDB run (see plan Step 10)."""
import sys
import numpy as np
import pandas as pd
import twin.config as c
from twin.data.vitaldb_loader import load_numeric_frame, _default_loader
from twin.data.drug_schedule import propofol_mg_min


def case_qualifies(frame: pd.DataFrame):
    prop = propofol_mg_min(frame)
    prop_minutes = float((prop > 0).sum()) / 60.0
    if prop_minutes < c.SP4_MIN_PROPOFOL_MINUTES:
        return False, f"insufficient propofol ({prop_minutes:.1f} min)"
    if c.MAP_TRACK not in frame:
        return False, "no ART_MBP column"
    art_frac = float(frame[c.MAP_TRACK].notna().mean())
    if art_frac < c.SP4_MIN_ART_FRACTION:
        return False, f"ART coverage {art_frac:.2f} < {c.SP4_MIN_ART_FRACTION}"
    return True, "ok"


def measure_coverage(eligible_cases: pd.DataFrame, loader_fn=_default_loader) -> dict:
    caseids, reasons = [], []
    for _, row in eligible_cases.iterrows():
        cid = int(row["caseid"])
        frame = load_numeric_frame(cid, loader_fn=loader_fn)
        ok, reason = case_qualifies(frame)
        if ok:
            caseids.append(cid)
        reasons.append(reason)
    return {"n_total": len(eligible_cases), "n_qualifying": len(caseids),
            "caseids": caseids, "reasons": reasons}


def main():
    eligible = pd.read_csv(c.RESULTS_DIR / "eligible_cases.csv")
    summary = measure_coverage(eligible)
    out = c.RESULTS_DIR / "sp4_cohort.csv"
    pd.DataFrame({"caseid": summary["caseids"]}).to_csv(out, index=False)
    print(f"SP4 cohort: {summary['n_qualifying']}/{summary['n_total']} cases "
          f"qualify -> {out}")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `python -m pytest tests/test_drug_schedule.py tests/test_measure_drug_coverage_smoke.py -v`
Expected: all passed.

- [ ] **Step 10: (MANUAL, user) Re-pull the SP4 cohort with drug tracks**

Requires the user's VitalDB session (DUA already accepted). From repo root:

```bash
python -m scripts.measure_drug_coverage
```

This downloads the new-track parquet files (new cache tag) for eligible cases and writes `results/sp4_cohort.csv`. Record the printed coverage number — it sizes SP4. **Verify** `Orchestra/PPF20_RATE` returns data (some cases use TCI tracks like `Orchestra/PPF20_CE` instead; if propofol coverage is near-zero, revisit the track name in config against `vitaldb.vital_recs` track listings).

- [ ] **Step 11: Commit**

```bash
git add twin/config.py twin/data/drug_schedule.py scripts/measure_drug_coverage.py tests/test_drug_schedule.py tests/test_measure_drug_coverage_smoke.py
git commit -m "feat(sp4): drug-track cohort selection + rate conversion"
```

---

## Task 2: Differentiable PK-PD twin (`torch_engine.py`)

**Files:**
- Create: `twin/pkpd/torch_engine.py`
- Test: `tests/test_torch_engine.py`

**Interface (batched):** all rate vectors are `[B, T]` tensors, δ is `[B, 6]` in order `(V1, V2, V3, ke0, EC50, gamma)`, params are `[B]` tensors. `project_map_torch(...) -> [B, T]` MAP tensor.

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_torch_engine.py`:

```python
import numpy as np
import pytest
pytest.importorskip("torch")
import torch

from twin.pkpd.params import POP_PROPOFOL, POP_PD, POP_NOREPI
from twin.pkpd.covariates import scale_propofol
from twin.pkpd.schedule import rate_vector
from twin.pkpd.engine import Patient, project_map
from twin.pkpd import torch_engine as te


def _sample_case(duration=600.0, dt=1.0):
    patient = Patient(age=50, weight=70, height=170, sex="M", map0=90.0)
    prop = [(0, 60, 200.0), (60, int(duration), 20.0)]   # induction bolus + maintenance
    norepi = [(120, 121, 8.0)]
    prop_rate = rate_vector(prop, dt, duration)
    norepi_rate = rate_vector(norepi, dt, duration)
    return patient, prop_rate, norepi_rate, dt, duration


def test_matches_numpy_engine_at_delta_zero():
    patient, prop_rate, norepi_rate, dt, duration = _sample_case()
    ref = project_map(patient, [(0, 60, 200.0), (60, int(duration), 20.0)],
                      [(120, 121, 8.0)], duration=duration, dt=dt).map
    out = te.project_map_torch(
        patient_batch=[patient],
        prop_rate=torch.tensor(prop_rate[None, :], dtype=torch.float64),
        norepi_rate=torch.tensor(norepi_rate[None, :], dtype=torch.float64),
        deltas=torch.zeros(1, 6, dtype=torch.float64),
        dt=dt,
    )
    got = out.detach().numpy()[0]
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-4)


def test_gradient_flows_to_deltas():
    patient, prop_rate, norepi_rate, dt, duration = _sample_case()
    deltas = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
    out = te.project_map_torch(
        patient_batch=[patient],
        prop_rate=torch.tensor(prop_rate[None, :], dtype=torch.float64),
        norepi_rate=torch.tensor(norepi_rate[None, :], dtype=torch.float64),
        deltas=deltas, dt=dt,
    )
    out.sum().backward()
    # EC50 (index 4) and ke0 (index 3) must have nonzero gradient
    g = deltas.grad[0]
    assert abs(g[4].item()) > 1e-6
    assert abs(g[3].item()) > 1e-6
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_torch_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: twin.pkpd.torch_engine`.

- [ ] **Step 3: Implement torch_engine**

Create `twin/pkpd/torch_engine.py`:

```python
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

MAP_MIN, MAP_MAX = 20.0, 200.0


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
    V1 = base["V1"] * torch.exp(d[:, 0]); V2 = base["V2"] * torch.exp(d[:, 1])
    V3 = base["V3"] * torch.exp(d[:, 2]); ke0 = base["ke0"] * torch.exp(d[:, 3])
    CL, Q2, Q3 = base["CL"], base["Q2"], base["Q3"]
    ec50 = torch.tensor(POP_PD.ec50, dtype=dtype, device=device) * torch.exp(d[:, 4])
    gamma = torch.tensor(POP_PD.gamma, dtype=dtype, device=device) * torch.exp(d[:, 5])
    emax = torch.tensor(POP_PD.emax, dtype=dtype, device=device)

    k10 = CL / V1; k12 = Q2 / V1; k21 = Q2 / V2; k13 = Q3 / V1; k31 = Q3 / V3
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
        a1 = a1 + da1; a2 = a2 + da2; a3 = a3 + da3
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_torch_engine.py -v`
Expected: 2 passed. (If parity fails at `atol=1e-4`, confirm both engines use the same `POP_PD.emax` and identical Euler ordering; the numpy `combine_map` clip range is `MAP_MIN/MAX = 20/200`, matched here.)

- [ ] **Step 5: Commit**

```bash
git add twin/pkpd/torch_engine.py tests/test_torch_engine.py
git commit -m "feat(sp4): differentiable batched PyTorch PK-PD engine"
```

---

## Task 3: Per-case δ teacher (`teacher.py`)

The teacher directly optimizes δ per case (through the differentiable twin) to match observed MAP, with the identifiability prior. Its output δ are the regression targets for Stage 1 distillation.

**Files:**
- Create: `twin/pkpd/teacher.py`
- Test: `tests/test_teacher.py`

- [ ] **Step 1: Write the failing test (recovers known δ on synthetic data)**

Create `tests/test_teacher.py`:

```python
import numpy as np
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
    """Fitted deltas reconstruct an observed MAP far better than the population twin.

    NOTE: we assert trajectory reconstruction, NOT exact recovery of the EC50 delta.
    With all 6 deltas free the model is degenerate (other deltas compensate for EC50)
    -- the project's documented identifiability limitation. Distillation only needs
    the teacher to match the observed MAP; the head learns whatever deltas it emits.
    """
    patient, prop, norepi = _case()
    true_delta = torch.zeros(1, 6, dtype=torch.float64)
    true_delta[0, 4] = 0.4
    observed = te.project_map_torch([patient], prop, norepi, true_delta).detach()

    zero = torch.zeros(1, 6, dtype=torch.float64)
    base_mse = float(((te.project_map_torch([patient], prop, norepi, zero) - observed) ** 2).mean())

    fitted, loss = fit_deltas_batch([patient], prop, norepi, observed,
                                    n_iters=200, lr=0.05, prior_lambda=1e-3)
    fit_mse = float(((te.project_map_torch([patient], prop, norepi, fitted) - observed) ** 2).mean())

    assert fit_mse < 1.0                 # reconstructs to well under 1 mmHg RMSE
    assert fit_mse < 0.1 * base_mse      # large improvement over population
    assert loss == pytest.approx(fit_mse, abs=0.05)


def test_prior_keeps_deltas_small_when_no_personalization_needed():
    patient, prop, norepi = _case()
    observed = te.project_map_torch([patient], prop, norepi,
                                    torch.zeros(1, 6, dtype=torch.float64)).detach()
    fitted, _ = fit_deltas_batch([patient], prop, norepi, observed,
                                 n_iters=100, lr=0.05, prior_lambda=1e-2)
    assert float(fitted.abs().max()) < 0.1
```

> **Design note (correction during execution):** the original plan asserted exact
> recovery of the EC50 delta. Empirically that is not achievable — with all 6 deltas
> free the fit is degenerate (other deltas compensate), which is the identifiability
> limitation the project explicitly accepts. The teacher's real contract for
> distillation is trajectory reconstruction, which it satisfies (population MSE ~30 →
> fitted MSE ~0.03). Tests were corrected accordingly and shortened (dur=300) for
> speed.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_teacher.py -v`
Expected: FAIL with `ModuleNotFoundError: twin.pkpd.teacher`.

- [ ] **Step 3: Implement teacher**

Create `twin/pkpd/teacher.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_teacher.py -v`
Expected: 2 passed (~35s; the differentiable twin runs a Python loop over T per iteration).

- [ ] **Step 5: Commit**

```bash
git add twin/pkpd/teacher.py tests/test_teacher.py
git commit -m "feat(sp4): per-case delta teacher via differentiable twin"
```

---

## Task 4: Coupled model (`coupled.py`)

Shared encoder → latent `z` → prediction head (IOH logit) + calibration head (6 δ). Encoder is a swappable module (SP2 replaces its body).

**Files:**
- Create: `twin/models/coupled.py`
- Test: `tests/test_coupled.py`

- [ ] **Step 1: Write the failing test (forward shapes + δ bounds)**

Create `tests/test_coupled.py`:

```python
import pytest
pytest.importorskip("torch")
import torch
from twin.models.coupled import CoupledTwin


def test_forward_shapes_and_delta_bounds():
    model = CoupledTwin(n_features=12, latent_dim=32, delta_bound=0.7)
    x = torch.randn(8, 12)
    logit, delta, z = model(x)
    assert logit.shape == (8,)
    assert delta.shape == (8, 6)
    assert z.shape == (8, 32)
    assert torch.all(delta.abs() <= 0.7 + 1e-5)


def test_encoder_is_swappable_module():
    # The encoder must be an attribute we can replace (SP2 hand-off).
    model = CoupledTwin(n_features=12, latent_dim=32)
    assert hasattr(model, "encoder")
    assert isinstance(model.encoder, torch.nn.Module)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_coupled.py -v`
Expected: FAIL with `ModuleNotFoundError: twin.models.coupled`.

- [ ] **Step 3: Implement coupled model**

Create `twin/models/coupled.py`:

```python
"""Coupled twin: shared encoder E -> latent z; prediction head P -> IOH logit;
calibration head C -> 6 personalization deltas. The shared z is the coupling.

Encoder is a plain nn.Module attribute so SP2 can swap its body (CNN-GRU waveform
stack) without touching the heads, losses, or training loop."""
import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    """MLP over standardized window features -> latent z. Replaced in SP2."""

    def __init__(self, n_features, latent_dim=32, hidden=64, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, latent_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CoupledTwin(nn.Module):
    def __init__(self, n_features, latent_dim=32, delta_bound=0.7, encoder=None):
        super().__init__()
        self.delta_bound = delta_bound
        self.encoder = encoder or TabularEncoder(n_features, latent_dim)
        self.pred_head = nn.Linear(latent_dim, 1)
        self.calib_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 6),
        )

    def forward(self, x):
        z = self.encoder(x)
        logit = self.pred_head(z).squeeze(-1)
        delta = self.delta_bound * torch.tanh(self.calib_head(z))
        return logit, delta, z
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_coupled.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add twin/models/coupled.py tests/test_coupled.py
git commit -m "feat(sp4): coupled twin (shared encoder + prediction + calibration heads)"
```

---

## Task 5: Stage 1 training — distillation (`train_calibration.py`)

Train the calibration head to regress the teacher's per-case δ. Includes a feature-prep helper (impute+standardize) reused across stages.

**Files:**
- Create: `scripts/train_calibration.py`
- Test: `tests/test_train_calibration_stage1.py`

- [ ] **Step 1: Write the failing test (distill overfits a toy batch)**

Create `tests/test_train_calibration_stage1.py`:

```python
import numpy as np
import pytest
pytest.importorskip("torch")
import torch
from scripts.train_calibration import FeaturePrep, distill_step


def test_distill_step_reduces_delta_mse():
    torch.manual_seed(0)
    from twin.models.coupled import CoupledTwin
    model = CoupledTwin(n_features=6, latent_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(32, 6)
    target_delta = torch.zeros(32, 6)
    target_delta[:, 4] = 0.3  # constant EC50 target

    first = distill_step(model, opt, x, target_delta)
    for _ in range(200):
        last = distill_step(model, opt, x, target_delta)
    assert last < first
    assert last < 0.05


def test_feature_prep_imputes_and_standardizes():
    import pandas as pd
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 20.0, 30.0]})
    prep = FeaturePrep().fit(df)
    A = prep.transform(df)
    assert not np.isnan(A).any()
    assert abs(A[:, 1].mean()) < 1e-6  # standardized column ~zero mean
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_train_calibration_stage1.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.train_calibration`.

- [ ] **Step 3: Implement Stage 1 (script with FeaturePrep + distill_step + CLI)**

Create `scripts/train_calibration.py`:

```python
"""Train the SP4 calibration head. Stage 1 distills from the per-case teacher;
Stage 2 fine-tunes end-to-end (added in Task 6). Device-agnostic (pick_device).

Usage:
  python -m scripts.train_calibration --stage 1 [--smoke]
  python -m scripts.train_calibration --stage 2 [--smoke]
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from twin.models.deepnet import pick_device
from twin.models.coupled import CoupledTwin


class FeaturePrep:
    """Median-impute + standardize numeric feature columns (fit on train)."""

    def __init__(self):
        self.columns = self.med = self.mu = self.sd = None

    def fit(self, df):
        self.columns = [c for c in df.columns if df[c].dtype != object]
        A = df[self.columns].to_numpy(dtype=float)
        self.med = np.nanmedian(A, axis=0)
        A = self._impute(A)
        self.mu = A.mean(axis=0); self.sd = A.std(axis=0) + 1e-6
        return self

    def _impute(self, A):
        idx = np.where(np.isnan(A))
        A = A.copy(); A[idx] = np.take(self.med, idx[1])
        return A

    def transform(self, df):
        A = self._impute(df[self.columns].to_numpy(dtype=float))
        return (A - self.mu) / self.sd


def distill_step(model, opt, x, target_delta):
    """One optimization step regressing calib head -> target_delta. Returns MSE."""
    model.train()
    opt.zero_grad()
    _, delta, _ = model(x)
    loss = nn.functional.mse_loss(delta, target_delta)
    loss.backward(); opt.step()
    return float(loss.detach())


# --- orchestration (Stage 1) -------------------------------------------------

def run_stage1(smoke=False):
    """Build features + per-case teacher deltas, train the calib head to match.
    In --smoke mode uses a tiny synthetic dataset so it runs on CPU in seconds."""
    device = pick_device()
    if smoke:
        n, f = 64, 6
        x = torch.randn(n, f, device=device)
        target = torch.zeros(n, 6, device=device); target[:, 4] = 0.3
        model = CoupledTwin(n_features=f, latent_dim=16).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(100):
            loss = distill_step(model, opt, x, target)
        print(f"[stage1-smoke] final distill MSE={loss:.4f}")
        return model
    raise NotImplementedError(
        "Full Stage 1 needs the SP4 cohort (Task 1 Step 10) + teacher deltas; "
        "run with --smoke on laptop, full run on the GPU harness (Task 8).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=(1, 2))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.stage == 1:
        run_stage1(smoke=args.smoke)
    else:
        from scripts.train_calibration_stage2 import run_stage2  # Task 6
        run_stage2(smoke=args.smoke)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_train_calibration_stage1.py -v`
Expected: 2 passed.

- [ ] **Step 5: Verify the smoke CLI runs on CPU**

Run: `python -m scripts.train_calibration --stage 1 --smoke`
Expected: prints `[stage1-smoke] final distill MSE=<small>` and exits 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_calibration.py tests/test_train_calibration_stage1.py
git commit -m "feat(sp4): Stage 1 distillation training (calib head <- teacher delta)"
```

---

## Task 6: Stage 2 training — end-to-end fine-tune

Joint loss: `BCE(IOH) + λ_recon·MAP-reconstruction (through the twin) + λ_prior·‖δ‖²`, from the Stage-1 init.

**Files:**
- Create: `scripts/train_calibration_stage2.py`
- Test: `tests/test_train_calibration_stage2.py`

- [ ] **Step 1: Write the failing test (recon loss decreases)**

Create `tests/test_train_calibration_stage2.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_train_calibration_stage2.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.train_calibration_stage2`.

- [ ] **Step 3: Implement Stage 2**

Create `scripts/train_calibration_stage2.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_train_calibration_stage2.py -v`
Expected: 1 passed.

- [ ] **Step 5: Verify Stage 2 smoke CLI**

Run: `python -m scripts.train_calibration --stage 2 --smoke`
Expected: prints `[stage2-smoke] {...}` and exits 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_calibration_stage2.py tests/test_train_calibration_stage2.py
git commit -m "feat(sp4): Stage 2 end-to-end fine-tune through differentiable twin"
```

---

## Task 7: Evaluation (`personalization.py`)

**Files:**
- Create: `twin/eval/personalization.py`
- Test: `tests/test_personalization.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_personalization.py`:

```python
import numpy as np
import pytest
pytest.importorskip("torch")
import torch
from twin.pkpd.engine import Patient
from twin.pkpd.schedule import rate_vector
from twin.pkpd import torch_engine as te
from twin.eval.personalization import (
    per_patient_map_rmse, delta_identifiability, whatif_monotonic)


def _case(delta_ec50=0.3, T=300):
    patient = Patient(50, 70, 170, "M", 90.0)
    prop = torch.tensor(rate_vector([(0, 60, 200.0), (60, T, 20.0)], 1.0,
                        float(T - 1))[None, :], dtype=torch.float64)
    norepi = torch.tensor(rate_vector([(120, 121, 8.0)], 1.0,
                          float(T - 1))[None, :], dtype=torch.float64)
    d = torch.zeros(1, 6, dtype=torch.float64); d[0, 4] = delta_ec50
    observed = te.project_map_torch([patient], prop, norepi, d).detach()
    return patient, prop, norepi, observed, d


def test_personalized_beats_population_rmse():
    patient, prop, norepi, observed, d = _case()
    rmse_pop = per_patient_map_rmse([patient], prop, norepi, observed,
                                    torch.zeros(1, 6, dtype=torch.float64))
    rmse_pers = per_patient_map_rmse([patient], prop, norepi, observed, d)
    assert rmse_pers[0] < rmse_pop[0]
    assert rmse_pers[0] < 1e-3


def test_delta_identifiability_returns_six_values():
    deltas = torch.zeros(5, 6, dtype=torch.float64)
    deltas[:, 4] = torch.linspace(-0.3, 0.3, 5)  # EC50 varies, others 0
    spread = delta_identifiability(deltas)
    assert spread.shape == (6,)
    assert spread[4] > spread[2]  # EC50 moves more than V3


def test_whatif_monotonic_more_propofol_more_risk():
    patient = Patient(50, 70, 170, "M", 90.0)
    assert whatif_monotonic(patient) is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_personalization.py -v`
Expected: FAIL with `ModuleNotFoundError: twin.eval.personalization`.

- [ ] **Step 3: Implement personalization eval**

Create `twin/eval/personalization.py`:

```python
"""SP4 evaluation: does personalization improve the twin, and is it identifiable?

- per_patient_map_rmse: masked RMSE of projected vs observed MAP, per patient.
- delta_identifiability: per-delta spread across the cohort (which delta move).
- whatif_monotonic: sanity that more propofol -> more projected hypotension."""
import numpy as np
import torch

from twin.pkpd import torch_engine as te
from twin.pkpd.schedule import rate_vector


def per_patient_map_rmse(patient_batch, prop_rate, norepi_rate, observed_map, deltas):
    pred = te.project_map_torch(patient_batch, prop_rate, norepi_rate,
                                deltas.to(prop_rate.dtype))
    obs = observed_map.to(pred.dtype)
    mask = ~torch.isnan(obs)
    obs = torch.nan_to_num(obs, nan=0.0)
    m = min(pred.shape[1], obs.shape[1])
    err = (pred[:, :m] - obs[:, :m]) * mask[:, :m]
    mse = (err ** 2).sum(dim=1) / mask[:, :m].sum(dim=1).clamp(min=1.0)
    return torch.sqrt(mse).detach().numpy()


def delta_identifiability(deltas):
    """Std of each delta across the cohort; large => data moves that delta."""
    return deltas.detach().float().std(dim=0).numpy()


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_personalization.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add twin/eval/personalization.py tests/test_personalization.py
git commit -m "feat(sp4): personalization evaluation (RMSE, identifiability, what-if)"
```

---

## Task 8: GPU training harness (Kaggle / Colab notebook)

Drives the full (non-smoke) Stage 1 + Stage 2 runs on a free GPU, pulling cases from VitalDB in-session. Notebooks aren't unit-tested; the guard is that the underlying script runs on CPU (verified in Tasks 5–6) and the notebook only orchestrates it.

**Files:**
- Create: `notebooks/train_calibration_gpu.ipynb`

- [ ] **Step 1: Create the notebook (model after `notebooks/train_deepnet_colab.ipynb`)**

Cells, in order:

1. **Setup** — `!pip install -q vitaldb torch pandas pyarrow` and `!git clone <repo>` (or `%cd` into a Kaggle dataset copy of the repo).
2. **Device check** — `from twin.models.deepnet import pick_device; print(pick_device())` → expect `cuda`.
3. **Cohort** — run `scripts/measure_drug_coverage.py` (pulls drug-track cases from VitalDB directly in-session) to produce `results/sp4_cohort.csv`.
4. **Teacher deltas** — batch `twin/pkpd/teacher.fit_deltas_batch` over cohort cases (GPU), cache to `results/sp4_teacher_deltas.parquet`.
5. **Stage 1** — call `scripts.train_calibration.run_stage1(smoke=False)` (implement the non-smoke path here using the cached teacher deltas + window features), save `results/coupled_stage1.pt`.
6. **Stage 2** — call `scripts.train_calibration_stage2.run_stage2(smoke=False)` from the Stage-1 init, save `results/coupled_stage2.pt`.
7. **Eval** — run `twin/eval/personalization` over a held-out split; print per-patient MAP-RMSE (personalized vs population), per-δ identifiability, coupling IOH metrics (reuse `twin/eval/selection_bias`), what-if monotonicity.
8. **Download** — save `.pt` + metrics CSV back to the repo / Kaggle output.

> The non-smoke `run_stage1`/`run_stage2` bodies (currently `NotImplementedError`) are filled in during this task, wiring: window features (`scripts/build_dataset.build_windows_for_case`) → `FeaturePrep` → `CoupledTwin`, teacher δ targets per case, and the observed-MAP tensors for reconstruction. Keep the batching identical to the smoke path so the tested `distill_step`/`joint_step` remain the core.

- [ ] **Step 2: Smoke-verify the orchestration locally (CPU)**

Run: `python -m scripts.train_calibration --stage 1 --smoke && python -m scripts.train_calibration --stage 2 --smoke`
Expected: both print their smoke summaries and exit 0.

- [ ] **Step 3: Commit**

```bash
git add notebooks/train_calibration_gpu.ipynb scripts/train_calibration.py scripts/train_calibration_stage2.py
git commit -m "feat(sp4): GPU training harness notebook (Kaggle/Colab) + non-smoke wiring"
```

- [ ] **Step 4: (MANUAL, user) Run the full training on GPU**

Upload the notebook to Kaggle (GPU T4/P100) or Colab, run all cells. Record: SP4 cohort size, per-patient MAP-RMSE (personalized vs population), per-δ identifiability, coupling IOH metrics. These are the SP4 results for the thesis/checkpoint.

---

## Full test run

- [ ] Run the whole suite to confirm nothing regressed:

Run: `python -m pytest -q`
Expected: all prior SP1/SP3 tests + the new SP4 tests pass.

---

## Self-Review notes (spec coverage)

- Spec §3 data prereq → Task 1 (config tracks, drug_schedule, coverage script, manual re-pull).
- Spec §5 differentiable twin → Task 2 (parity + gradient gates).
- Spec §6 Stage 1 distill → Tasks 3 (teacher) + 5 (distill); Stage 2 → Task 6.
- Spec §4 coupling / shared latent → Task 4 (`CoupledTwin`, swappable encoder).
- Spec §7 evaluation → Task 7.
- Spec §8 GPU harness → Task 8.
- Spec §10 SP2 hand-off → Task 4's swappable `encoder` attribute (asserted in test).
```
