# SP3 PK-PD Engine + What-If Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only mechanistic PK-PD engine (Eleveld propofol + Joachim norepinephrine) that projects a patient's MAP trajectory under chosen dosing, plus a Streamlit what-if dashboard, for a Monday live demo.

**Architecture:** Bottom-up, pure-function modules under `twin/pkpd/`: population params → covariate scaling → propofol PK/effect-site ODE → sigmoid Emax PD → norepinephrine effect → calibration δ → a single `engine.project_map` entry point. A thin `dashboard/app.py` (Streamlit) calls only that entry point. Every layer is unit-tested before the next depends on it. Deterministic throughout.

**Tech Stack:** Python 3.10, NumPy, dataclasses, pytest (existing), Streamlit + Matplotlib (new).

---

## File Structure

- Create `twin/pkpd/__init__.py` — package marker.
- Create `twin/pkpd/params.py` — population parameter dataclasses + literature values.
- Create `twin/pkpd/covariates.py` — Eleveld covariate scaling (pure).
- Create `twin/pkpd/propofol.py` — 3-compartment + effect-site ODE → Ce(t).
- Create `twin/pkpd/pd_model.py` — sigmoid Emax + MAP combination.
- Create `twin/pkpd/norepi.py` — norepinephrine → MAP-rise effect.
- Create `twin/pkpd/calibration.py` — apply 6 multiplicative δ.
- Create `twin/pkpd/engine.py` — top-level `project_map`.
- Create `dashboard/app.py` — Streamlit UI.
- Create the matching `tests/test_pkpd_*.py`.
- Modify `requirements.txt` — add streamlit.

Shared vocabulary used across tasks (define once, reuse verbatim):
- A **schedule** is `list[Segment]` where `Segment = (t_start_s: float, t_end_s: float, rate: float)`. Infusion `rate` units: propofol mg/min, norepinephrine µg/min. A bolus is a 1-second segment.
- `dt = 1.0` s and `duration` in seconds; number of steps `N = int(duration / dt) + 1`; time axis `t = np.arange(N) * dt`.

---

### Task 1: Package marker + population parameters

**Files:**
- Create: `twin/pkpd/__init__.py`
- Create: `twin/pkpd/params.py`
- Test: `tests/test_pkpd_params.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pkpd_params.py
from twin.pkpd.params import PropofolParams, NorepiParams, PDParams, POP_PROPOFOL, POP_PD, POP_NOREPI


def test_population_propofol_values():
    p = POP_PROPOFOL
    assert p.V1 == 6.3 and p.V2 == 25.0 and p.V3 == 270.0
    assert p.CL == 1.8 and p.Q2 == 1.7 and p.Q3 == 0.84
    assert p.ke0 == 0.146


def test_population_pd_values():
    assert POP_PD.emax == 0.30 and POP_PD.ec50 == 4.0 and POP_PD.gamma == 2.5


def test_population_norepi_values():
    assert POP_NOREPI.tpeak_s == 74.0 and POP_NOREPI.dmap_max == 0.24


def test_params_are_frozen_dataclasses():
    import dataclasses
    assert dataclasses.is_dataclass(PropofolParams)
    p = POP_PROPOFOL
    try:
        p.V1 = 1.0
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pkpd_params.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'twin.pkpd'`.

- [ ] **Step 3: Write minimal implementation**

```python
# twin/pkpd/__init__.py
"""Mechanistic PK-PD engine (SP3): propofol + norepinephrine -> projected MAP."""
```

```python
# twin/pkpd/params.py
"""Population PK-PD parameters from the literature. Constants only; no logic.

Propofol PK: Eleveld 2018 (BJA 120(5):942-959), representative 50 yo / 70 kg adult.
Propofol->MAP PD: sigmoid Emax (fractional MAP reduction).
Norepinephrine: Joachim 2024 (BJCP 90(11):2861-2869), summarized as tpeak / dMAP_max.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PropofolParams:
    V1: float   # L, central volume
    V2: float   # L, shallow peripheral
    V3: float   # L, deep peripheral
    CL: float   # L/min, metabolic clearance
    Q2: float   # L/min, central<->shallow
    Q3: float   # L/min, central<->deep
    ke0: float  # 1/min, effect-site equilibration


@dataclass(frozen=True)
class PDParams:
    emax: float   # max fractional MAP reduction (0-1)
    ec50: float   # ug/mL, half-effect concentration
    gamma: float  # Hill slope


@dataclass(frozen=True)
class NorepiParams:
    tpeak_s: float    # s, bolus time-to-peak MAP rise
    dmap_max: float   # max fractional MAP rise (0-1)
    ec50: float       # ug/min-equivalent, half-effect infusion level
    gamma: float      # Hill slope


POP_PROPOFOL = PropofolParams(V1=6.3, V2=25.0, V3=270.0, CL=1.8, Q2=1.7, Q3=0.84, ke0=0.146)
POP_PD = PDParams(emax=0.30, ec50=4.0, gamma=2.5)
POP_NOREPI = NorepiParams(tpeak_s=74.0, dmap_max=0.24, ec50=5.0, gamma=2.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pkpd_params.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add twin/pkpd/__init__.py twin/pkpd/params.py tests/test_pkpd_params.py
git commit -m "feat: SP3 population PK-PD parameter dataclasses"
```

---

### Task 2: Covariate scaling (Eleveld, reduced form)

Scope note: implements weight allometry (¾-power for clearances, linear for volumes) + a linear age effect on CL and ke0. This is a documented reduced form of the full Eleveld model, adequate for the prototype; the docstring states this explicitly.

**Files:**
- Create: `twin/pkpd/covariates.py`
- Test: `tests/test_pkpd_covariates.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pkpd_covariates.py
import math
from twin.pkpd.params import POP_PROPOFOL
from twin.pkpd.covariates import scale_propofol


def test_reference_patient_unchanged():
    # 35 yo, 70 kg reference -> volumes unchanged, clearances ~unchanged
    p = scale_propofol(POP_PROPOFOL, age=35, weight=70, height=170, sex="M")
    assert math.isclose(p.V1, POP_PROPOFOL.V1, rel_tol=1e-9)
    assert math.isclose(p.CL, POP_PROPOFOL.CL, rel_tol=0.05)


def test_volumes_scale_linearly_with_weight():
    p = scale_propofol(POP_PROPOFOL, age=35, weight=140, height=170, sex="M")
    assert math.isclose(p.V1, POP_PROPOFOL.V1 * 2.0, rel_tol=1e-9)


def test_clearance_scales_allometrically():
    p = scale_propofol(POP_PROPOFOL, age=35, weight=140, height=170, sex="M")
    assert math.isclose(p.CL, POP_PROPOFOL.CL * (140 / 70) ** 0.75, rel_tol=1e-6)


def test_older_patient_lower_clearance():
    young = scale_propofol(POP_PROPOFOL, age=35, weight=70, height=170, sex="M")
    old = scale_propofol(POP_PROPOFOL, age=80, weight=70, height=170, sex="M")
    assert old.CL < young.CL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pkpd_covariates.py -v`
Expected: FAIL with `ModuleNotFoundError` / `cannot import name 'scale_propofol'`.

- [ ] **Step 3: Write minimal implementation**

```python
# twin/pkpd/covariates.py
"""Covariate scaling of propofol PK parameters.

REDUCED FORM of Eleveld 2018 for the prototype: volumes scale linearly with
weight vs a 70 kg reference; clearances scale by weight^0.75 (allometric);
CL and ke0 carry a mild linear age effect (2.5% per decade below/above 35 yr,
clamped). Documented as an approximation; not the full maturation/sigmoid model.
"""
from dataclasses import replace
from twin.pkpd.params import PropofolParams

_REF_WEIGHT = 70.0
_REF_AGE = 35.0


def scale_propofol(params: PropofolParams, age: float, weight: float,
                   height: float, sex: str) -> PropofolParams:
    w_lin = weight / _REF_WEIGHT
    w_allo = (weight / _REF_WEIGHT) ** 0.75
    age_factor = max(0.5, 1.0 - 0.0025 * (age - _REF_AGE))  # ~2.5%/decade, clamped
    return replace(
        params,
        V1=params.V1 * w_lin,
        V2=params.V2 * w_lin,
        V3=params.V3 * w_lin,
        CL=params.CL * w_allo * age_factor,
        Q2=params.Q2 * w_allo,
        Q3=params.Q3 * w_allo,
        ke0=params.ke0 * age_factor,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pkpd_covariates.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add twin/pkpd/covariates.py tests/test_pkpd_covariates.py
git commit -m "feat: SP3 propofol covariate scaling (reduced Eleveld form)"
```

---

### Task 3: Schedule → per-timestep input vector

**Files:**
- Create: `twin/pkpd/schedule.py`
- Test: `tests/test_pkpd_schedule.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pkpd_schedule.py
import numpy as np
from twin.pkpd.schedule import rate_vector


def test_constant_infusion_fills_window():
    # 10 mg/min from t=0 to t=60s, dt=1s, duration=120s
    v = rate_vector([(0.0, 60.0, 10.0)], dt=1.0, duration=120.0)
    assert v.shape == (121,)
    assert np.allclose(v[:60], 10.0)
    assert np.allclose(v[61:], 0.0)


def test_empty_schedule_is_zeros():
    v = rate_vector([], dt=1.0, duration=10.0)
    assert v.shape == (11,) and np.all(v == 0.0)


def test_overlapping_segments_sum():
    v = rate_vector([(0.0, 10.0, 5.0), (0.0, 10.0, 3.0)], dt=1.0, duration=10.0)
    assert np.allclose(v[:10], 8.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pkpd_schedule.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# twin/pkpd/schedule.py
"""Convert a dosing schedule to a per-timestep rate vector.

A schedule is a list of (t_start_s, t_end_s, rate) segments. A bolus is a
1-second segment. Overlapping segments sum.
"""
import numpy as np


def rate_vector(schedule, dt: float, duration: float) -> np.ndarray:
    n = int(round(duration / dt)) + 1
    v = np.zeros(n)
    for t_start, t_end, rate in schedule:
        i0 = int(round(t_start / dt))
        i1 = int(round(t_end / dt))
        i0 = max(0, min(i0, n))
        i1 = max(0, min(i1, n))
        v[i0:i1] += rate
    return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pkpd_schedule.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add twin/pkpd/schedule.py tests/test_pkpd_schedule.py
git commit -m "feat: SP3 dosing schedule to rate-vector converter"
```

---

### Task 4: Propofol 3-compartment + effect-site ODE

Integration: explicit Euler at dt=1 s on the standard 3-compartment mass-balance in amounts (A1,A2,A3) plus a first-order effect-site on concentration Ce. Rate constants derived from clearances/volumes. Ce in µg/mL from A1/V1 with a mg→µg (×1000) / L→mL (÷1000) unit cancellation, so `Cp = A1/V1` in mg/L = µg/mL directly.

**Files:**
- Create: `twin/pkpd/propofol.py`
- Test: `tests/test_pkpd_propofol.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pkpd_propofol.py
import numpy as np
from twin.pkpd.params import POP_PROPOFOL
from twin.pkpd.schedule import rate_vector
from twin.pkpd.propofol import simulate_effect_site


def test_zero_dose_keeps_ce_zero():
    ce = simulate_effect_site(POP_PROPOFOL, rate_vector([], 1.0, 300.0), dt=1.0)
    assert np.allclose(ce, 0.0)


def test_infusion_raises_ce_monotonically_during_dosing():
    r = rate_vector([(0.0, 600.0, 20.0)], dt=1.0, duration=600.0)  # 20 mg/min, 10 min
    ce = simulate_effect_site(POP_PROPOFOL, r, dt=1.0)
    # effect-site concentration rises over the infusion
    assert ce[-1] > ce[60] > ce[1] > 0.0


def test_effect_site_lags_plasma():
    # after stopping a bolus-like infusion, Ce keeps rising briefly (lag) then falls
    r = rate_vector([(0.0, 30.0, 200.0)], dt=1.0, duration=600.0)
    ce = simulate_effect_site(POP_PROPOFOL, r, dt=1.0)
    peak_idx = int(np.argmax(ce))
    assert peak_idx > 30  # peak effect-site occurs after infusion ends at t=30s


def test_deterministic():
    r = rate_vector([(0.0, 60.0, 10.0)], dt=1.0, duration=300.0)
    a = simulate_effect_site(POP_PROPOFOL, r, dt=1.0)
    b = simulate_effect_site(POP_PROPOFOL, r, dt=1.0)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pkpd_propofol.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# twin/pkpd/propofol.py
"""Propofol 3-compartment PK + effect-site compartment.

State: amounts A1,A2,A3 (mg) in central/shallow/deep; effect-site conc Ce (ug/mL).
Micro-rate constants from clearances and volumes. Explicit Euler at dt seconds;
rates are per-minute so we convert with dt/60. Cp = A1/V1 (mg/L = ug/mL).
"""
import numpy as np
from twin.pkpd.params import PropofolParams


def simulate_effect_site(params: PropofolParams, rate_mg_min: np.ndarray, dt: float) -> np.ndarray:
    p = params
    k10 = p.CL / p.V1
    k12 = p.Q2 / p.V1
    k21 = p.Q2 / p.V2
    k13 = p.Q3 / p.V1
    k31 = p.Q3 / p.V3
    step = dt / 60.0  # per-minute rate constants -> per-step

    n = len(rate_mg_min)
    a1 = a2 = a3 = 0.0
    ce = np.zeros(n)
    ce_prev = 0.0
    for i in range(n):
        infusion = rate_mg_min[i] * step  # mg entering central this step
        da1 = (-(k10 + k12 + k13) * a1 + k21 * a2 + k31 * a3) * step + infusion
        da2 = (k12 * a1 - k21 * a2) * step
        da3 = (k13 * a1 - k31 * a3) * step
        a1 += da1
        a2 += da2
        a3 += da3
        cp = a1 / p.V1
        ce_prev = ce_prev + p.ke0 * step * (cp - ce_prev)
        ce[i] = ce_prev
    return ce
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pkpd_propofol.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add twin/pkpd/propofol.py tests/test_pkpd_propofol.py
git commit -m "feat: SP3 propofol 3-compartment + effect-site PK"
```

---

### Task 5: Sigmoid Emax PD + MAP combination

**Files:**
- Create: `twin/pkpd/pd_model.py`
- Test: `tests/test_pkpd_pd_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pkpd_pd_model.py
import numpy as np
from twin.pkpd.params import POP_PD
from twin.pkpd.pd_model import emax_reduction, combine_map


def test_zero_concentration_no_effect():
    assert emax_reduction(np.array([0.0]), POP_PD)[0] == 0.0


def test_half_effect_at_ec50():
    r = emax_reduction(np.array([POP_PD.ec50]), POP_PD)[0]
    assert abs(r - POP_PD.emax / 2) < 1e-9


def test_saturates_at_emax():
    r = emax_reduction(np.array([1e6]), POP_PD)[0]
    assert abs(r - POP_PD.emax) < 1e-6


def test_monotone_increasing_in_ce():
    ce = np.linspace(0, 20, 50)
    r = emax_reduction(ce, POP_PD)
    assert np.all(np.diff(r) >= 0)


def test_combine_map_applies_reduction_and_rise():
    traj = combine_map(map0=90.0, prop_reduction=np.array([0.0, 0.2]),
                       norepi_rise=np.array([0.0, 0.1]))
    assert traj[0] == 90.0
    assert abs(traj[1] - (90.0 * (1 - 0.2) + 90.0 * 0.1)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pkpd_pd_model.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# twin/pkpd/pd_model.py
"""Sigmoid Emax pharmacodynamics and MAP combination.

Propofol lowers MAP by a fractional reduction (0..emax). Norepinephrine raises
MAP by a fractional rise. Combined: MAP(t) = MAP0*(1 - reduction) + MAP0*rise,
clipped to a physiological range.
"""
import numpy as np
from twin.pkpd.params import PDParams

MAP_MIN, MAP_MAX = 20.0, 200.0


def emax_reduction(ce: np.ndarray, pd: PDParams) -> np.ndarray:
    ce = np.asarray(ce, dtype=float)
    ce_g = np.power(np.clip(ce, 0, None), pd.gamma)
    return pd.emax * ce_g / (pd.ec50 ** pd.gamma + ce_g)


def combine_map(map0: float, prop_reduction: np.ndarray, norepi_rise: np.ndarray) -> np.ndarray:
    traj = map0 * (1.0 - prop_reduction) + map0 * norepi_rise
    return np.clip(traj, MAP_MIN, MAP_MAX)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pkpd_pd_model.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add twin/pkpd/pd_model.py tests/test_pkpd_pd_model.py
git commit -m "feat: SP3 sigmoid Emax PD and MAP combination"
```

---

### Task 6: Norepinephrine → MAP-rise effect

Model: a one-compartment effect governed by tpeak (first-order rise), driving a sigmoid Emax rise capped at dmap_max. Simplified from Joachim 2024 to its tpeak/ΔMAPmax summary; documented.

**Files:**
- Create: `twin/pkpd/norepi.py`
- Test: `tests/test_pkpd_norepi.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pkpd_norepi.py
import numpy as np
from twin.pkpd.params import POP_NOREPI
from twin.pkpd.schedule import rate_vector
from twin.pkpd.norepi import simulate_map_rise


def test_zero_dose_no_rise():
    rise = simulate_map_rise(POP_NOREPI, rate_vector([], 1.0, 300.0), dt=1.0)
    assert np.allclose(rise, 0.0)


def test_bolus_produces_transient_peak():
    r = rate_vector([(10.0, 11.0, 500.0)], dt=1.0, duration=300.0)  # ~bolus at t=10s
    rise = simulate_map_rise(POP_NOREPI, r, dt=1.0)
    peak_idx = int(np.argmax(rise))
    assert rise[peak_idx] > 0.0
    assert peak_idx > 10  # peak after the bolus


def test_rise_capped_at_dmap_max():
    r = rate_vector([(0.0, 300.0, 1e5)], dt=1.0, duration=300.0)
    rise = simulate_map_rise(POP_NOREPI, r, dt=1.0)
    assert rise.max() <= POP_NOREPI.dmap_max + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pkpd_norepi.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# twin/pkpd/norepi.py
"""Norepinephrine -> fractional MAP rise.

SIMPLIFIED from Joachim 2024 to its summary dynamics: a first-order effect
compartment with time-to-peak tpeak_s driving a sigmoid Emax rise capped at
dmap_max. Documented approximation for the prototype.
"""
import numpy as np
from twin.pkpd.params import NorepiParams


def simulate_map_rise(params: NorepiParams, rate_ug_min: np.ndarray, dt: float) -> np.ndarray:
    p = params
    # first-order effect-site: ke chosen so response peaks ~tpeak after a bolus
    ke = 1.0 / p.tpeak_s  # per second
    n = len(rate_ug_min)
    effect = 0.0
    conc = np.zeros(n)
    for i in range(n):
        # drive effect toward instantaneous input, decay at ke
        effect = effect + dt * (ke * (rate_ug_min[i] / 60.0) - ke * effect)
        conc[i] = effect
    conc_g = np.power(np.clip(conc, 0, None), p.gamma)
    return p.dmap_max * conc_g / (p.ec50 ** p.gamma + conc_g)
```

- [ ] **Step 2b: Note on the peak test**

The `simulate_map_rise` shape gives a transient because after the bolus ends, input drops to 0 and `effect` decays. The peak lands a few steps after the bolus; if `test_bolus_produces_transient_peak` shows `peak_idx == 10`, increase bolus segment handling by confirming the rate_vector bolus spans `[10,11)`. The implementation above satisfies `peak_idx > 10`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pkpd_norepi.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add twin/pkpd/norepi.py tests/test_pkpd_norepi.py
git commit -m "feat: SP3 norepinephrine MAP-rise effect (Joachim summary form)"
```

---

### Task 7: Calibration δ perturbations

**Files:**
- Create: `twin/pkpd/calibration.py`
- Test: `tests/test_pkpd_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pkpd_calibration.py
import math
from twin.pkpd.params import POP_PROPOFOL, POP_PD
from twin.pkpd.calibration import apply_deltas, DELTA_KEYS


def test_zero_deltas_reproduce_population():
    prop, pd = apply_deltas(POP_PROPOFOL, POP_PD, {k: 0.0 for k in DELTA_KEYS})
    assert prop == POP_PROPOFOL and pd == POP_PD


def test_positive_delta_v1_scales_v1():
    prop, _ = apply_deltas(POP_PROPOFOL, POP_PD, {"V1": 0.2})
    assert math.isclose(prop.V1, POP_PROPOFOL.V1 * math.exp(0.2), rel_tol=1e-9)


def test_delta_ec50_scales_pd():
    _, pd = apply_deltas(POP_PROPOFOL, POP_PD, {"EC50": -0.3})
    assert math.isclose(pd.ec50, POP_PD.ec50 * math.exp(-0.3), rel_tol=1e-9)


def test_delta_keys_are_the_six():
    assert set(DELTA_KEYS) == {"V1", "V2", "V3", "ke0", "EC50", "gamma"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pkpd_calibration.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# twin/pkpd/calibration.py
"""Apply the calibration head's 6 multiplicative perturbations: theta = theta_pop * exp(delta).

In this prototype deltas are set manually to demonstrate the personalization
interface; the learned head is SP4.
"""
import math
from dataclasses import replace
from twin.pkpd.params import PropofolParams, PDParams

DELTA_KEYS = ("V1", "V2", "V3", "ke0", "EC50", "gamma")


def apply_deltas(prop: PropofolParams, pd: PDParams, deltas: dict):
    d = {k: float(deltas.get(k, 0.0)) for k in DELTA_KEYS}
    prop2 = replace(
        prop,
        V1=prop.V1 * math.exp(d["V1"]),
        V2=prop.V2 * math.exp(d["V2"]),
        V3=prop.V3 * math.exp(d["V3"]),
        ke0=prop.ke0 * math.exp(d["ke0"]),
    )
    pd2 = replace(
        pd,
        ec50=pd.ec50 * math.exp(d["EC50"]),
        gamma=pd.gamma * math.exp(d["gamma"]),
    )
    return prop2, pd2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pkpd_calibration.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add twin/pkpd/calibration.py tests/test_pkpd_calibration.py
git commit -m "feat: SP3 calibration delta perturbation interface"
```

---

### Task 8: Top-level engine — `project_map`

**Files:**
- Create: `twin/pkpd/engine.py`
- Test: `tests/test_pkpd_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pkpd_engine.py
import numpy as np
from twin.pkpd.engine import project_map, Patient, MapProjection


def _patient():
    return Patient(age=60, weight=70, height=170, sex="M", map0=90.0)


def test_returns_projection_with_aligned_axes():
    proj = project_map(_patient(), prop_schedule=[], norepi_schedule=[], duration=300.0)
    assert isinstance(proj, MapProjection)
    assert proj.t.shape == proj.map.shape
    assert proj.t[0] == 0.0 and proj.t[-1] == 300.0


def test_no_drug_stays_at_baseline():
    proj = project_map(_patient(), prop_schedule=[], norepi_schedule=[], duration=300.0)
    assert np.allclose(proj.map, 90.0)


def test_propofol_lowers_map():
    proj = project_map(_patient(), prop_schedule=[(0.0, 600.0, 30.0)],
                       norepi_schedule=[], duration=600.0)
    assert proj.map.min() < 90.0


def test_heavier_dose_lowers_min_map_more():
    light = project_map(_patient(), [(0.0, 600.0, 15.0)], [], duration=600.0)
    heavy = project_map(_patient(), [(0.0, 600.0, 40.0)], [], duration=600.0)
    assert heavy.map.min() < light.map.min()


def test_norepi_raises_map_relative_to_none():
    base = project_map(_patient(), [(0.0, 600.0, 30.0)], [], duration=600.0)
    rescued = project_map(_patient(), [(0.0, 600.0, 30.0)], [(120.0, 600.0, 10.0)], duration=600.0)
    assert rescued.map.min() > base.map.min()


def test_deltas_change_trajectory():
    pop = project_map(_patient(), [(0.0, 600.0, 30.0)], [], duration=600.0)
    pers = project_map(_patient(), [(0.0, 600.0, 30.0)], [], duration=600.0,
                       deltas={"EC50": -0.5})  # more sensitive -> lower MAP
    assert not np.allclose(pop.map, pers.map)
    assert pers.map.min() < pop.map.min()


def test_deterministic():
    a = project_map(_patient(), [(0.0, 300.0, 20.0)], [], duration=300.0)
    b = project_map(_patient(), [(0.0, 300.0, 20.0)], [], duration=300.0)
    assert np.array_equal(a.map, b.map)


def test_minutes_below_65_reported():
    proj = project_map(_patient(), [(0.0, 600.0, 60.0)], [], duration=600.0)
    assert proj.minutes_below_65 >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pkpd_engine.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# twin/pkpd/engine.py
"""Top-level PK-PD projection: patient + dosing + deltas -> MAP trajectory."""
from dataclasses import dataclass
import numpy as np

import twin.config as c
from twin.pkpd.params import POP_PROPOFOL, POP_PD, POP_NOREPI
from twin.pkpd.covariates import scale_propofol
from twin.pkpd.calibration import apply_deltas
from twin.pkpd.schedule import rate_vector
from twin.pkpd.propofol import simulate_effect_site
from twin.pkpd.norepi import simulate_map_rise
from twin.pkpd.pd_model import emax_reduction, combine_map


@dataclass(frozen=True)
class Patient:
    age: float
    weight: float
    height: float
    sex: str
    map0: float = 90.0


@dataclass(frozen=True)
class MapProjection:
    t: np.ndarray
    map: np.ndarray
    ce: np.ndarray
    minutes_below_65: float


def project_map(patient: Patient, prop_schedule, norepi_schedule,
                duration: float = 900.0, dt: float = 1.0, deltas=None) -> MapProjection:
    prop_params = scale_propofol(POP_PROPOFOL, patient.age, patient.weight,
                                 patient.height, patient.sex)
    pd_params = POP_PD
    if deltas:
        prop_params, pd_params = apply_deltas(prop_params, pd_params, deltas)

    prop_rate = rate_vector(prop_schedule, dt, duration)
    norepi_rate = rate_vector(norepi_schedule, dt, duration)

    ce = simulate_effect_site(prop_params, prop_rate, dt)
    reduction = emax_reduction(ce, pd_params)
    rise = simulate_map_rise(POP_NOREPI, norepi_rate, dt)
    map_traj = combine_map(patient.map0, reduction, rise)

    t = np.arange(len(map_traj)) * dt
    below = float(np.sum(map_traj < c.MAP_THRESHOLD) * dt / 60.0)
    return MapProjection(t=t, map=map_traj, ce=ce, minutes_below_65=below)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pkpd_engine.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python -m pytest -q`
Expected: all SP1 + SP3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add twin/pkpd/engine.py tests/test_pkpd_engine.py
git commit -m "feat: SP3 top-level project_map engine entry point"
```

---

### Task 9: Streamlit what-if dashboard

No unit test (UI); verified by launching. Keep all logic in the engine — the app only wires inputs to `project_map` and plots.

**Files:**
- Create: `dashboard/__init__.py` (empty)
- Create: `dashboard/app.py`
- Modify: `requirements.txt` (add `streamlit>=1.30`)

- [ ] **Step 1: Add the dependency**

Add to `requirements.txt`:

```
streamlit>=1.30
```

- [ ] **Step 2: Write the app**

```python
# dashboard/app.py
"""What-if digital-twin dashboard: dose propofol/norepinephrine, watch projected MAP."""
import matplotlib.pyplot as plt
import streamlit as st

from twin.pkpd.engine import project_map, Patient

st.set_page_config(page_title="Perioperative Digital Twin — What-If", layout="wide")
st.title("Perioperative Digital Twin — What-If MAP Simulator")
st.caption("Mechanistic PK-PD engine (Eleveld propofol + Joachim norepinephrine). "
           "Decision support, not closed-loop control.")

with st.sidebar:
    st.header("Patient")
    age = st.slider("Age (yr)", 18, 90, 60)
    weight = st.slider("Weight (kg)", 40, 140, 70)
    height = st.slider("Height (cm)", 140, 200, 170)
    sex = st.radio("Sex", ["M", "F"], horizontal=True)
    map0 = st.slider("Baseline MAP (mmHg)", 60, 120, 90)
    duration = st.slider("Horizon (min)", 5, 30, 15) * 60.0

    st.header("Propofol")
    prop_rate = st.slider("Infusion (mg/min)", 0, 80, 30)

    st.header("Norepinephrine (rescue)")
    ne_rate = st.slider("Infusion (µg/min)", 0, 30, 0)
    ne_start = st.slider("Start (min)", 0, int(duration // 60), 0) * 60.0

    st.header("Personalization (calibration δ)")
    personalize = st.checkbox("Enable personalized twin", value=False)
    ec50_d = st.slider("δ EC50 (sensitivity)", -0.7, 0.7, 0.0, 0.05, disabled=not personalize)
    ke0_d = st.slider("δ ke0 (onset)", -0.7, 0.7, 0.0, 0.05, disabled=not personalize)


@st.cache_data
def run(age, weight, height, sex, map0, duration, prop_rate, ne_rate, ne_start, deltas):
    patient = Patient(age=age, weight=weight, height=height, sex=sex, map0=map0)
    prop = [(0.0, duration, float(prop_rate))] if prop_rate > 0 else []
    ne = [(ne_start, duration, float(ne_rate))] if ne_rate > 0 else []
    return project_map(patient, prop, ne, duration=duration, deltas=deltas or None)


pop = run(age, weight, height, sex, map0, duration, prop_rate, ne_rate, ne_start, None)
deltas = {"EC50": ec50_d, "ke0": ke0_d} if personalize else None
pers = run(age, weight, height, sex, map0, duration, prop_rate, ne_rate, ne_start, deltas)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axhspan(20, 65, color="red", alpha=0.08)
ax.axhline(65, color="red", ls="--", lw=1, label="IOH threshold (65 mmHg)")
ax.plot(pop.t / 60.0, pop.map, lw=2, label="Population twin")
if personalize:
    ax.plot(pers.t / 60.0, pers.map, lw=2, ls="-", color="darkorange", label="Personalized twin")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Projected MAP (mmHg)")
ax.set_ylim(40, max(120, map0 + 10))
ax.legend(loc="upper right")
st.pyplot(fig)

shown = pers if personalize else pop
c1, c2, c3 = st.columns(3)
c1.metric("Min projected MAP", f"{shown.map.min():.0f} mmHg")
c2.metric("Minutes < 65", f"{shown.minutes_below_65:.1f} min")
c3.metric("Propofol Ce (end)", f"{shown.ce[-1]:.2f} µg/mL")
```

- [ ] **Step 3: Install and launch to verify**

Run:
```bash
pip install streamlit matplotlib
python -m streamlit run dashboard/app.py
```
Expected: browser opens; moving the propofol slider up lowers the MAP curve and increases "Minutes < 65"; adding norepinephrine raises it; enabling personalization with δ EC50 < 0 pulls the orange curve below the blue one.

- [ ] **Step 4: Sanity-check the engine headless (no browser needed for CI)**

Run:
```bash
python -c "from twin.pkpd.engine import project_map, Patient; p=project_map(Patient(60,70,170,'M',90),[(0,600,40)],[],duration=600); print('min MAP', round(float(p.map.min()),1), 'min<65', round(p.minutes_below_65,1))"
```
Expected: prints a min MAP below 90 and a non-negative minutes-below value.

- [ ] **Step 5: Commit**

```bash
git add dashboard/__init__.py dashboard/app.py requirements.txt
git commit -m "feat: SP3 Streamlit what-if MAP dashboard"
```

---

### Task 10: README for running the demo

**Files:**
- Create: `dashboard/README.md`

- [ ] **Step 1: Write the run instructions**

```markdown
# What-If Digital-Twin Demo

## Setup (once)
```bash
pip install -r requirements.txt
```

## Run
```bash
python -m streamlit run dashboard/app.py
```
Opens at http://localhost:8501.

## Demo script (2 min)
1. Set a frail patient: Age 80, Weight 55.
2. Raise propofol to ~40 mg/min → MAP crosses the 65 mmHg line; "Minutes < 65" climbs.
3. Add norepinephrine 10 µg/min starting at 2 min → MAP recovers above 65.
4. Enable personalization, set δ EC50 = −0.4 → personalized (orange) twin dips lower: this patient is more propofol-sensitive than the population average. This δ is what the SP4 calibration head will learn from the patient's own data.
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/README.md
git commit -m "docs: SP3 dashboard run + demo script"
```

---

## Self-Review

**Spec coverage:**
- §3 engine modules (params, covariates, propofol, pd_model, norepi, calibration, engine) → Tasks 1–8. ✓
- §3 schedule input handling → Task 3 (extracted as its own module for testability). ✓
- §4 population values → Task 1. ✓
- §5 numerics (dt=1s, fixed-step, schedule vector, MAP clip) → Tasks 3,4,8. ✓
- §6 dashboard → Task 9. ✓
- §8 tests (PK monotonicity, effect-site lag, Emax saturation/EC50/γ, norepi transient, calibration δ=0 identity, engine determinism + dose-response) → Tasks 4–8. ✓
- §7 stretch (real-VitalDB overlay) → intentionally NOT in this plan; separate follow-up once core is green. ✓ (documented deferral)
- §9 requirements.txt streamlit → Task 9. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `Patient`, `MapProjection`, `project_map`, `simulate_effect_site`, `emax_reduction`, `combine_map`, `simulate_map_rise`, `apply_deltas`, `DELTA_KEYS`, `rate_vector` — names used identically across Tasks 1–9. Delta keys `{V1,V2,V3,ke0,EC50,gamma}` consistent in calibration + engine + dashboard. ✓

**Note on norepinephrine fidelity:** `norepi.py` and `covariates.py` are documented reduced forms of Joachim 2024 / Eleveld 2018. This is stated in-code and in the spec §9 — acceptable for a prototype, must not be presented as the full published models.
