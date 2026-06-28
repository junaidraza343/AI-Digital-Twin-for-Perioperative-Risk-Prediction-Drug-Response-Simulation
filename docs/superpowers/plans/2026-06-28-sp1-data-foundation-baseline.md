# SP1 — Data Foundation + Tabular Baseline + Selection-Bias Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only VitalDB pipeline plus two baseline IOH predictors and a selection-bias evaluation harness that reproduces the 2025 biased-vs-unbiased performance gap on our own cohort.

**Architecture:** Pure-logic, network-free modules (config, labeling, cohort, splits, metrics, decision-curve, features) are built and unit-tested first via TDD; the network-dependent VitalDB loader is built behind an injectable interface so its caching is testable with a fake; baselines and the selection-bias harness compose the above; thin CLI scripts and two notebooks produce the checkpoint deliverables.

**Tech Stack:** Python 3.10, pandas, NumPy, scikit-learn, LightGBM (fallback: sklearn `HistGradientBoostingClassifier`), `vitaldb` package, pytest, Jupyter.

---

## Canonical interfaces (locked here, used across tasks)

```
twin/config.py                     constants only
twin/data/labeling.py              make_map_series(frame) -> np.ndarray
                                   label_windows(map_series) -> pd.DataFrame[t_end,y,category]
twin/data/cohort.py                filter_cohort(cases_df) -> (eligible_df, funnel_list)
twin/data/splits.py                make_splits(meta_df, ratios, seed) -> dict[caseid -> split]
twin/data/features.py              extract_features(frame, t_end, static_row) -> dict
twin/data/vitaldb_loader.py        load_numeric_frame(caseid, loader_fn=...) -> pd.DataFrame
                                   load_cases_table() -> pd.DataFrame
twin/eval/metrics.py               auroc, auprc, ppv_at_alarm_rate, expected_calibration_error, brier
twin/eval/decision_curve.py        net_benefit(y, p, pt) -> float ; net_benefit_curve(y,p,thresholds)
twin/eval/selection_bias.py        make_biased(windows_df), make_unbiased(windows_df), evaluate(...)
twin/models/baselines.py           MapOnlyModel, GbdtModel  (fit / predict_proba)
```

All randomness derives from `twin.config.SEED`. Window schema everywhere: columns `caseid, t_end, y, category` (+ feature columns after `features`).

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `twin/__init__.py`, `twin/data/__init__.py`, `twin/eval/__init__.py`, `twin/models/__init__.py`
- Create: `tests/__init__.py`
- Create: `twin/config.py`

- [ ] **Step 1: Create `requirements.txt`**

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
lightgbm>=4.0
vitaldb>=1.4.0
pyarrow>=14.0
pytest>=7.4
jupyter>=1.0
matplotlib>=3.7
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 3: Create empty package files**

Create these files, each containing a single comment line `# package`:
`twin/__init__.py`, `twin/data/__init__.py`, `twin/eval/__init__.py`, `twin/models/__init__.py`, `tests/__init__.py`

- [ ] **Step 4: Create `twin/config.py`**

```python
# Central configuration. Constants only; no logic.
from pathlib import Path

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"

# Label definition
MAP_THRESHOLD = 65.0          # mmHg, hypotension threshold
EVENT_MIN_SECONDS = 60        # cumulative seconds below threshold to count as an event
HORIZON_SECONDS = 5 * 60      # prediction horizon
OBS_WINDOW_SECONDS = 10 * 60  # observation window length before t_end
STRIDE_SECONDS = 30           # window stride
HORIZON_MIN_VALID = 0.5       # min fraction of non-NaN samples required in horizon

# Gray-zone band (upper bound for "some dip" categorization)
GRAY_HIGH = 75.0

# 1 Hz numeric tracks to load from VitalDB
MAP_TRACK = "Solar8000/ART_MBP"
MAP_TRACK_FALLBACK = "Solar8000/NIBP_MBP"
NUMERIC_TRACKS = [
    MAP_TRACK, MAP_TRACK_FALLBACK,
    "Solar8000/ART_SBP", "Solar8000/ART_DBP",
    "Solar8000/HR", "Solar8000/PLETH_SPO2",
    "Solar8000/ETCO2", "Solar8000/RR", "Solar8000/BT",
]

# Cohort filtering
EXCLUDED_DEPARTMENTS = {"Cardiac surgery", "Thoracic surgery", "Gynecology"}
MIN_AGE = 18
MIN_DURATION_SECONDS = 1800
EXCLUDED_ASA = {5, 6}

# Splits
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# Evaluation
ALARM_RATE = 0.10             # operating point for PPV reporting
ECE_BINS = 15
```

- [ ] **Step 5: Verify imports work**

Run: `python -c "import twin.config as c; print(c.SEED, c.MAP_THRESHOLD, len(c.NUMERIC_TRACKS))"`
Expected: `42 65.0 9`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini twin tests
git commit -m "chore: scaffold twin package and config"
```

---

### Task 2: Labeling (the core; fully TDD)

**Files:**
- Create: `twin/data/labeling.py`
- Test: `tests/test_labeling.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_labeling.py
import numpy as np
import twin.config as c
from twin.data.labeling import label_windows


def _flat(value, n=3000):
    return np.full(n, value, dtype=float)


def test_stable_window_all_high():
    m = _flat(80.0)
    df = label_windows(m)
    assert (df.y == 0).all()
    assert (df.category == "stable").all()


def test_overt_event_90s_dip():
    m = _flat(80.0)
    # 90s sustained dip below 65 inside the horizon of the first valid window
    start = c.OBS_WINDOW_SECONDS + 30
    m[start:start + 90] = 60.0
    df = label_windows(m)
    pos = df[df.t_end < start]  # windows whose horizon covers the dip
    assert (pos.y == 1).any()
    assert (df.loc[df.y == 1, "category"] == "overt").all()


def test_short_dip_40s_is_negative_graylike():
    m = _flat(80.0)
    start = c.OBS_WINDOW_SECONDS + 30
    m[start:start + 40] = 60.0  # only 40s < 65, fails the 60s rule
    df = label_windows(m)
    covering = df[(df.t_end < start) & (df.t_end + c.HORIZON_SECONDS >= start + 40)]
    assert (covering.y == 0).all()
    assert (covering.category == "gray").all()  # min horizon MAP < 75


def test_gray_band_70_is_gray_negative():
    m = _flat(80.0)
    start = c.OBS_WINDOW_SECONDS + 30
    m[start:start + 120] = 70.0  # in [65,75), never below 65
    df = label_windows(m)
    covering = df[(df.t_end < start) & (df.t_end + c.HORIZON_SECONDS >= start + 120)]
    assert (covering.y == 0).all()
    assert (covering.category == "gray").all()


def test_already_hypotensive_window_dropped():
    m = _flat(80.0)
    t = c.OBS_WINDOW_SECONDS + 300
    m[t] = 60.0  # current MAP at t_end already < 65 -> that window excluded
    df = label_windows(m)
    assert (df.t_end != t).all()


def test_no_windows_before_obs_or_after_horizon():
    m = _flat(80.0, n=c.OBS_WINDOW_SECONDS + c.HORIZON_SECONDS + 5)
    df = label_windows(m)
    assert df.t_end.min() >= c.OBS_WINDOW_SECONDS
    assert (df.t_end + c.HORIZON_SECONDS <= len(m) - 1).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_labeling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'twin.data.labeling'`

- [ ] **Step 3: Write the implementation**

```python
# twin/data/labeling.py
"""Turn a 1 Hz MAP series into labeled prediction windows."""
import numpy as np
import pandas as pd
import twin.config as c


def make_map_series(frame: pd.DataFrame) -> np.ndarray:
    """Extract a 1 Hz MAP array, preferring the arterial track, else NIBP."""
    if c.MAP_TRACK in frame and frame[c.MAP_TRACK].notna().any():
        s = frame[c.MAP_TRACK].astype(float)
    elif c.MAP_TRACK_FALLBACK in frame:
        s = frame[c.MAP_TRACK_FALLBACK].astype(float)
    else:
        raise ValueError("No MAP track available in frame")
    # Physiologically impossible values -> NaN
    arr = s.to_numpy(dtype=float)
    arr[(arr < 10) | (arr > 250)] = np.nan
    return arr


def label_windows(map_series: np.ndarray) -> pd.DataFrame:
    """Roll windows over a 1 Hz MAP array and assign label + category.

    A window ends at t_end (seconds). It is a prediction point only if the full
    observation window precedes it and the full horizon follows it. The current
    MAP at t_end must be valid and >= threshold (otherwise the event is already
    underway -> detection, not prediction, so we drop it).

    y = 1 iff MAP < threshold for >= EVENT_MIN_SECONDS cumulative seconds in the
    horizon (t_end, t_end + HORIZON]. category: overt (y==1), else gray if the
    minimum valid horizon MAP < GRAY_HIGH, else stable.
    """
    n = len(map_series)
    rows = []
    last_end = n - c.HORIZON_SECONDS - 1
    for t_end in range(c.OBS_WINDOW_SECONDS, last_end + 1, c.STRIDE_SECONDS):
        cur = map_series[t_end]
        if np.isnan(cur) or cur < c.MAP_THRESHOLD:
            continue
        horizon = map_series[t_end + 1: t_end + 1 + c.HORIZON_SECONDS]
        valid = horizon[~np.isnan(horizon)]
        if len(valid) < c.HORIZON_MIN_VALID * c.HORIZON_SECONDS:
            continue
        seconds_below = int(np.sum(valid < c.MAP_THRESHOLD))
        y = 1 if seconds_below >= c.EVENT_MIN_SECONDS else 0
        if y == 1:
            category = "overt"
        elif valid.min() < c.GRAY_HIGH:
            category = "gray"
        else:
            category = "stable"
        rows.append((t_end, y, category))
    return pd.DataFrame(rows, columns=["t_end", "y", "category"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_labeling.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add twin/data/labeling.py tests/test_labeling.py
git commit -m "feat: MAP windowing and IOH labeling with gray-zone categories"
```

---

### Task 3: Cohort filtering (TDD)

**Files:**
- Create: `twin/data/cohort.py`
- Test: `tests/test_cohort.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cohort.py
import pandas as pd
from twin.data.cohort import filter_cohort


def _base_row(**over):
    row = dict(caseid=1, age=40, department="General surgery",
               ane_type="General", casestart=0, caseend=4000,
               asa=2, preop_sbp=120, preop_dbp=80, preop_htn=False)
    row.update(over)
    return row


def test_keeps_eligible_case():
    df = pd.DataFrame([_base_row()])
    eligible, funnel = filter_cohort(df)
    assert len(eligible) == 1
    assert funnel[0] == ("start", 1)


def test_excludes_minor():
    df = pd.DataFrame([_base_row(age=15)])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_excludes_cardiac_department():
    df = pd.DataFrame([_base_row(department="Cardiac surgery")])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_excludes_non_general_anesthesia():
    df = pd.DataFrame([_base_row(ane_type="Spinal")])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_excludes_short_case():
    df = pd.DataFrame([_base_row(casestart=0, caseend=1000)])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_excludes_asa_5():
    df = pd.DataFrame([_base_row(asa=5)])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_excludes_preop_hypotension():
    # preop MAP = 50 + (60-50)/3 ~= 53 < 65
    df = pd.DataFrame([_base_row(preop_sbp=60, preop_dbp=50)])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_funnel_records_each_step():
    df = pd.DataFrame([_base_row(), _base_row(caseid=2, age=10)])
    _, funnel = filter_cohort(df)
    steps = [name for name, _ in funnel]
    assert steps[0] == "start"
    assert "age>=18" in steps
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cohort.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# twin/data/cohort.py
"""Filter the VitalDB clinical-information table to the study cohort."""
import pandas as pd
import twin.config as c


def _preop_map(df: pd.DataFrame) -> pd.Series:
    return df["preop_dbp"] + (df["preop_sbp"] - df["preop_dbp"]) / 3.0


def filter_cohort(cases_df: pd.DataFrame):
    """Return (eligible_df, funnel) where funnel is a list of (step, n_remaining)."""
    df = cases_df.copy()
    funnel = [("start", len(df))]

    df = df[df["age"] >= c.MIN_AGE]
    funnel.append(("age>=18", len(df)))

    df = df[~df["department"].isin(c.EXCLUDED_DEPARTMENTS)]
    funnel.append(("department", len(df)))

    df = df[df["ane_type"].astype(str).str.contains("General", case=False, na=False)]
    funnel.append(("general_anesthesia", len(df)))

    df = df[(df["caseend"] - df["casestart"]) >= c.MIN_DURATION_SECONDS]
    funnel.append(("duration>=30min", len(df)))

    df = df[~df["asa"].isin(c.EXCLUDED_ASA)]
    funnel.append(("asa_not_5_6", len(df)))

    df = df[_preop_map(df) >= c.MAP_THRESHOLD]
    funnel.append(("preop_map>=65", len(df)))

    return df.reset_index(drop=True), funnel
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cohort.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add twin/data/cohort.py tests/test_cohort.py
git commit -m "feat: VitalDB cohort filtering with funnel log"
```

---

### Task 4: Case-level stratified splits (TDD, leakage guard)

**Files:**
- Create: `twin/data/splits.py`
- Test: `tests/test_splits.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_splits.py
import pandas as pd
import twin.config as c
from twin.data.splits import make_splits


def _meta(n=200):
    return pd.DataFrame({
        "caseid": range(n),
        "age": [20 + (i % 60) for i in range(n)],
        "asa": [1 + (i % 3) for i in range(n)],
        "preop_htn": [bool(i % 2) for i in range(n)],
    })


def test_every_case_assigned_one_split():
    assign = make_splits(_meta())
    assert set(assign.values()) <= {"train", "val", "test"}
    assert len(assign) == 200


def test_no_leakage_caseids_disjoint():
    assign = make_splits(_meta())
    buckets = {"train": set(), "val": set(), "test": set()}
    for cid, sp in assign.items():
        buckets[sp].add(cid)
    assert buckets["train"].isdisjoint(buckets["val"])
    assert buckets["train"].isdisjoint(buckets["test"])
    assert buckets["val"].isdisjoint(buckets["test"])


def test_ratios_approximately_hold():
    assign = make_splits(_meta(1000))
    frac_train = sum(v == "train" for v in assign.values()) / 1000
    assert abs(frac_train - c.SPLIT_RATIOS[0]) < 0.05


def test_deterministic_with_seed():
    assert make_splits(_meta()) == make_splits(_meta())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# twin/data/splits.py
"""Case-level stratified train/val/test assignment (no segment leakage)."""
import numpy as np
import pandas as pd
import twin.config as c


def make_splits(meta_df: pd.DataFrame, ratios=c.SPLIT_RATIOS, seed=c.SEED) -> dict:
    """Assign each caseid to 'train'/'val'/'test', stratified by
    ASA x age-decile x antihypertensive-medication, deterministically."""
    rng = np.random.default_rng(seed)
    meta = meta_df.copy()
    asa = meta["asa"].fillna(-1).astype(int)
    decile = (meta["age"] // 10).clip(upper=9).astype(int)
    htn = meta["preop_htn"].fillna(False).astype(bool)
    meta["stratum"] = list(zip(asa, decile, htn))

    assign = {}
    for _, grp in meta.groupby("stratum"):
        ids = grp["caseid"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        for i, cid in enumerate(ids):
            if i < n_tr:
                assign[cid] = "train"
            elif i < n_tr + n_va:
                assign[cid] = "val"
            else:
                assign[cid] = "test"
    return assign
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_splits.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add twin/data/splits.py tests/test_splits.py
git commit -m "feat: case-level stratified splits with leakage tests"
```

---

### Task 5: Metrics (TDD)

**Files:**
- Create: `twin/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import numpy as np
from twin.eval.metrics import (auroc, auprc, ppv_at_alarm_rate,
                               expected_calibration_error, brier)


def test_auroc_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(y, p) == 1.0


def test_auprc_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auprc(y, p) == 1.0


def test_brier_zero_for_perfect():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    assert brier(y, p) == 0.0


def test_ppv_at_alarm_rate_top_decile():
    # 100 samples, top 10 by score are all positive -> PPV = 1.0
    y = np.zeros(100); y[:10] = 1
    p = np.linspace(1.0, 0.0, 100)  # first 10 highest scores are the positives
    assert ppv_at_alarm_rate(y, p, 0.10) == 1.0


def test_ece_zero_for_calibrated_extremes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# twin/eval/metrics.py
"""Evaluation metrics for IOH prediction."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import twin.config as c


def auroc(y, p):
    return float(roc_auc_score(y, p))


def auprc(y, p):
    return float(average_precision_score(y, p))


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def ppv_at_alarm_rate(y, p, alarm_rate=c.ALARM_RATE):
    """Precision when the top `alarm_rate` fraction of scores fire an alarm."""
    y = np.asarray(y); p = np.asarray(p)
    n = len(p)
    k = max(1, int(round(alarm_rate * n)))
    thresh = np.sort(p)[::-1][k - 1]
    pred = p >= thresh
    denom = int(np.sum(pred))
    if denom == 0:
        return 0.0
    return float(np.sum((pred) & (y == 1)) / denom)


def expected_calibration_error(y, p, n_bins=c.ECE_BINS):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(p)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add twin/eval/metrics.py tests/test_metrics.py
git commit -m "feat: discrimination, calibration, and PPV metrics"
```

---

### Task 6: Decision-curve analysis (TDD)

**Files:**
- Create: `twin/eval/decision_curve.py`
- Test: `tests/test_decision_curve.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_decision_curve.py
import numpy as np
from twin.eval.decision_curve import net_benefit, net_benefit_curve


def test_net_benefit_treat_all_at_low_threshold():
    # At a very low threshold everyone is treated -> NB ~= prevalence
    y = np.array([0, 0, 1, 1])  # prevalence 0.5
    p = np.array([0.4, 0.4, 0.4, 0.4])
    nb = net_benefit(y, p, pt=0.001)
    assert abs(nb - 0.5) < 0.05


def test_net_benefit_zero_when_nothing_fires():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.1, 0.1, 0.1])
    assert net_benefit(y, p, pt=0.9) == 0.0


def test_curve_has_one_value_per_threshold():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    ts = np.array([0.1, 0.5, 0.9])
    curve = net_benefit_curve(y, p, ts)
    assert len(curve) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_decision_curve.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# twin/eval/decision_curve.py
"""Decision-curve analysis: clinical net benefit across thresholds."""
import numpy as np


def net_benefit(y, p, pt):
    """Net benefit at threshold probability pt (Vickers & Elkin 2006)."""
    y = np.asarray(y); p = np.asarray(p)
    n = len(y)
    pred = p >= pt
    tp = int(np.sum((pred) & (y == 1)))
    fp = int(np.sum((pred) & (y == 0)))
    return float(tp / n - (fp / n) * (pt / (1 - pt)))


def net_benefit_curve(y, p, thresholds):
    return np.array([net_benefit(y, p, t) for t in thresholds])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_decision_curve.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add twin/eval/decision_curve.py tests/test_decision_curve.py
git commit -m "feat: decision-curve net-benefit analysis"
```

---

### Task 7: Feature extraction (TDD)

**Files:**
- Create: `twin/data/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py
import numpy as np
import pandas as pd
import twin.config as c
from twin.data.features import extract_features


def _frame(n=2000):
    # MAP ramps linearly from 100 down to 80 over the whole frame
    idx = np.arange(n)
    return pd.DataFrame({
        c.MAP_TRACK: np.linspace(100.0, 80.0, n),
        "Solar8000/HR": np.full(n, 70.0),
    }, index=idx)


def test_returns_static_and_trend_features():
    frame = _frame()
    static = {"age": 50, "sex": 1, "bmi": 24.0}
    t_end = c.OBS_WINDOW_SECONDS + 100
    feats = extract_features(frame, t_end, static)
    assert feats["age"] == 50
    assert "map_mean" in feats
    assert "map_slope" in feats
    assert "map_last" in feats


def test_map_last_matches_value_at_t_end():
    frame = _frame()
    t_end = c.OBS_WINDOW_SECONDS + 100
    feats = extract_features(frame, t_end, {})
    assert abs(feats["map_last"] - frame[c.MAP_TRACK].iloc[t_end]) < 1e-6


def test_map_slope_is_negative_for_downward_ramp():
    frame = _frame()
    t_end = c.OBS_WINDOW_SECONDS + 100
    feats = extract_features(frame, t_end, {})
    assert feats["map_slope"] < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# twin/data/features.py
"""Extract static + numeric-trend features over an observation window."""
import numpy as np
import pandas as pd
import twin.config as c

# Numeric tracks summarized as trend features (MAP handled explicitly).
_TREND_TRACKS = [
    "Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/ETCO2",
    "Solar8000/RR", "Solar8000/BT", "Solar8000/ART_SBP", "Solar8000/ART_DBP",
]


def _slope(values: np.ndarray) -> float:
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return 0.0
    x = np.arange(len(values))[valid]
    y = values[valid]
    return float(np.polyfit(x, y, 1)[0])


def _summ(prefix, values, out):
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        out[f"{prefix}_mean"] = np.nan
        out[f"{prefix}_std"] = np.nan
        out[f"{prefix}_min"] = np.nan
        out[f"{prefix}_max"] = np.nan
        out[f"{prefix}_last"] = np.nan
        out[f"{prefix}_slope"] = 0.0
        return
    out[f"{prefix}_mean"] = float(valid.mean())
    out[f"{prefix}_std"] = float(valid.std())
    out[f"{prefix}_min"] = float(valid.min())
    out[f"{prefix}_max"] = float(valid.max())
    out[f"{prefix}_last"] = float(valid[-1])
    out[f"{prefix}_slope"] = _slope(values)


def extract_features(frame: pd.DataFrame, t_end: int, static_row: dict) -> dict:
    """Build a feature dict for the window ending at t_end."""
    start = t_end - c.OBS_WINDOW_SECONDS
    out = dict(static_row)  # static covariates pass through unchanged

    map_track = c.MAP_TRACK if (c.MAP_TRACK in frame and frame[c.MAP_TRACK].notna().any()) else c.MAP_TRACK_FALLBACK
    map_win = frame[map_track].to_numpy(dtype=float)[start:t_end + 1]
    _summ("map", map_win, out)

    for track in _TREND_TRACKS:
        if track in frame:
            win = frame[track].to_numpy(dtype=float)[start:t_end + 1]
            _summ(track.split("/")[-1].lower(), win, out)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add twin/data/features.py tests/test_features.py
git commit -m "feat: static + numeric-trend feature extraction"
```

---

### Task 8: VitalDB loader with testable caching (TDD via fake)

**Files:**
- Create: `twin/data/vitaldb_loader.py`
- Test: `tests/test_vitaldb_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vitaldb_loader.py
import numpy as np
import pandas as pd
import twin.config as c
from twin.data.vitaldb_loader import load_numeric_frame


def test_loader_called_once_then_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "DATA_CACHE", tmp_path)
    calls = {"n": 0}

    def fake_loader(caseid, tracks, interval):
        calls["n"] += 1
        # vitaldb returns a 2D array: rows = time, cols = tracks
        return np.tile(np.arange(5.0), (len(tracks), 1)).T

    f1 = load_numeric_frame(1, loader_fn=fake_loader)
    f2 = load_numeric_frame(1, loader_fn=fake_loader)
    assert calls["n"] == 1           # second call served from cache
    assert isinstance(f1, pd.DataFrame)
    assert list(f1.columns) == c.NUMERIC_TRACKS
    pd.testing.assert_frame_equal(f1, f2)


def test_returns_one_column_per_track(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "DATA_CACHE", tmp_path)

    def fake_loader(caseid, tracks, interval):
        return np.zeros((10, len(tracks)))

    frame = load_numeric_frame(7, loader_fn=fake_loader)
    assert frame.shape == (10, len(c.NUMERIC_TRACKS))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vitaldb_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# twin/data/vitaldb_loader.py
"""Lazy, cached access to VitalDB numeric tracks and the clinical table."""
import numpy as np
import pandas as pd
import twin.config as c

CASES_URL = "https://api.vitaldb.net/cases"


def _default_loader(caseid, tracks, interval):
    import vitaldb
    return vitaldb.load_case(caseid, tracks, interval)


def load_numeric_frame(caseid, loader_fn=_default_loader) -> pd.DataFrame:
    """Return a 1 Hz DataFrame of NUMERIC_TRACKS for one case, cached on disk."""
    c.DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = c.DATA_CACHE / f"case_{caseid}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    arr = loader_fn(caseid, c.NUMERIC_TRACKS, 1)
    arr = np.asarray(arr, dtype=float)
    frame = pd.DataFrame(arr, columns=c.NUMERIC_TRACKS)
    frame.to_parquet(cache_path)
    return frame


def load_cases_table() -> pd.DataFrame:
    """Download the VitalDB clinical-information table (cases.csv)."""
    return pd.read_csv(CASES_URL)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vitaldb_loader.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add twin/data/vitaldb_loader.py tests/test_vitaldb_loader.py
git commit -m "feat: cached VitalDB numeric-frame loader"
```

---

### Task 9: Baseline models (TDD)

**Files:**
- Create: `twin/models/baselines.py`
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_baselines.py
import numpy as np
import pandas as pd
from twin.models.baselines import MapOnlyModel, GbdtModel


def _separable():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({
        "map_last": np.concatenate([rng.normal(85, 3, n), rng.normal(68, 3, n)]),
        "map_slope": np.concatenate([rng.normal(0.0, 0.1, n), rng.normal(-0.2, 0.1, n)]),
        "hr_mean": rng.normal(70, 5, 2 * n),
    })
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y


def test_map_only_uses_only_map_features_and_learns():
    X, y = _separable()
    m = MapOnlyModel().fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(y),)
    assert ((p >= 0) & (p <= 1)).all()
    from twin.eval.metrics import auroc
    assert auroc(y, p) > 0.8


def test_gbdt_learns_and_is_deterministic():
    X, y = _separable()
    p1 = GbdtModel().fit(X, y).predict_proba(X)
    p2 = GbdtModel().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1, p2)
    from twin.eval.metrics import auroc
    assert auroc(y, p1) > 0.85
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# twin/models/baselines.py
"""Baseline IOH predictors: MAP-only logistic and gradient-boosted trees."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import twin.config as c

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:  # pragma: no cover - environment dependent
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_LGBM = False

MAP_FEATURES = ["map_last", "map_slope"]


class MapOnlyModel:
    """Logistic regression on current MAP + MAP slope only."""

    def __init__(self):
        self.pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )

    def fit(self, X, y):
        self.pipe.fit(X[MAP_FEATURES], y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X[MAP_FEATURES])[:, 1]


class GbdtModel:
    """Gradient-boosted trees on the full feature set."""

    def __init__(self):
        if _HAS_LGBM:
            self.model = LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                random_state=c.SEED, n_jobs=-1, verbose=-1,
            )
        else:  # pragma: no cover
            self.model = HistGradientBoostingClassifier(random_state=c.SEED)
        self.columns = None

    def fit(self, X, y):
        self.columns = [col for col in X.columns if X[col].dtype != object]
        self.model.fit(X[self.columns], y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.columns])[:, 1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_baselines.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add twin/models/baselines.py tests/test_baselines.py
git commit -m "feat: MAP-only and GBDT baseline models"
```

---

### Task 10: Selection-bias harness (TDD)

**Files:**
- Create: `twin/eval/selection_bias.py`
- Test: `tests/test_selection_bias.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_selection_bias.py
import numpy as np
import pandas as pd
from twin.eval.selection_bias import make_biased, make_unbiased, evaluate


def _windows():
    return pd.DataFrame({
        "caseid": [1, 1, 2, 2, 3],
        "t_end": [600, 630, 600, 630, 600],
        "y": [0, 1, 0, 0, 1],
        "category": ["stable", "overt", "gray", "stable", "overt"],
    })


def test_biased_drops_gray_zone():
    out = make_biased(_windows())
    assert "gray" not in set(out.category)
    assert len(out) == 4


def test_unbiased_keeps_everything():
    out = make_unbiased(_windows())
    assert len(out) == 5


def test_evaluate_returns_expected_metric_keys():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(y, p)
    for k in ["auroc", "auprc", "ppv", "ece", "brier"]:
        assert k in m
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_selection_bias.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# twin/eval/selection_bias.py
"""Biased vs unbiased evaluation harness for the selection-bias study."""
import pandas as pd
from twin.eval import metrics


def make_biased(windows_df: pd.DataFrame) -> pd.DataFrame:
    """HPI-style: keep only clearly-stable negatives and overt positives."""
    return windows_df[windows_df["category"] != "gray"].reset_index(drop=True)


def make_unbiased(windows_df: pd.DataFrame) -> pd.DataFrame:
    """Realistic: keep all windows, gray zone included."""
    return windows_df.reset_index(drop=True)


def evaluate(y, p) -> dict:
    """Compute the standard metric bundle for one (model, regime)."""
    return {
        "auroc": metrics.auroc(y, p),
        "auprc": metrics.auprc(y, p),
        "ppv": metrics.ppv_at_alarm_rate(y, p),
        "ece": metrics.expected_calibration_error(y, p),
        "brier": metrics.brier(y, p),
        "n": int(len(y)),
        "prevalence": float(sum(y) / len(y)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_selection_bias.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add twin/eval/selection_bias.py tests/test_selection_bias.py
git commit -m "feat: biased-vs-unbiased evaluation harness"
```

---

### Task 11: Cohort-build script (integration)

**Files:**
- Create: `scripts/run_cohort.py`
- Test: `tests/test_run_cohort_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_run_cohort_smoke.py
import pandas as pd
import twin.config as c
from scripts.run_cohort import build_cohort


def test_build_cohort_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "RESULTS_DIR", tmp_path)
    fake_cases = pd.DataFrame([
        dict(caseid=1, age=40, department="General surgery", ane_type="General",
             casestart=0, caseend=4000, asa=2, preop_sbp=120, preop_dbp=80,
             preop_htn=False),
        dict(caseid=2, age=10, department="General surgery", ane_type="General",
             casestart=0, caseend=4000, asa=2, preop_sbp=120, preop_dbp=80,
             preop_htn=False),
    ])
    eligible, funnel = build_cohort(cases_df=fake_cases)
    assert list(eligible.caseid) == [1]
    assert (tmp_path / "cohort_funnel.csv").exists()
    assert (tmp_path / "eligible_cases.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_cohort_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_cohort'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/run_cohort.py
"""Build the study cohort from the VitalDB clinical table and save it."""
import pandas as pd
import twin.config as c
from twin.data.cohort import filter_cohort
from twin.data.vitaldb_loader import load_cases_table


def build_cohort(cases_df=None):
    if cases_df is None:
        cases_df = load_cases_table()
    eligible, funnel = filter_cohort(cases_df)
    c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(funnel, columns=["step", "n_remaining"]).to_csv(
        c.RESULTS_DIR / "cohort_funnel.csv", index=False)
    eligible.to_csv(c.RESULTS_DIR / "eligible_cases.csv", index=False)
    return eligible, funnel


if __name__ == "__main__":  # pragma: no cover
    eligible, funnel = build_cohort()
    print("Cohort funnel:")
    for step, n in funnel:
        print(f"  {step:20s} {n}")
    print(f"Eligible cases: {len(eligible)}")
```

Create `scripts/__init__.py` with the single line `# package`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_cohort_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/run_cohort.py tests/test_run_cohort_smoke.py
git commit -m "feat: cohort-build script"
```

---

### Task 12: Dataset-build + baseline-train script (integration)

**Files:**
- Create: `scripts/build_dataset.py`
- Create: `scripts/run_baseline.py`
- Test: `tests/test_build_dataset_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_build_dataset_smoke.py
import numpy as np
import pandas as pd
import twin.config as c
from scripts.build_dataset import build_windows_for_case


def test_build_windows_for_case_returns_labeled_features(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "DATA_CACHE", tmp_path)
    n = c.OBS_WINDOW_SECONDS + c.HORIZON_SECONDS + 200
    # Build a frame: stable MAP at 85 with one 90s dip near the end.
    arr = np.full((n, len(c.NUMERIC_TRACKS)), 85.0)
    map_col = c.NUMERIC_TRACKS.index(c.MAP_TRACK)
    dip = c.OBS_WINDOW_SECONDS + 60
    arr[dip:dip + 90, map_col] = 60.0

    def fake_loader(caseid, tracks, interval):
        return arr

    static = {"caseid": 1, "age": 50, "sex": 1, "bmi": 24.0}
    out = build_windows_for_case(1, static, loader_fn=fake_loader)
    assert {"caseid", "t_end", "y", "category", "map_last"} <= set(out.columns)
    assert (out.caseid == 1).all()
    assert out.y.isin([0, 1]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_dataset_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `scripts/build_dataset.py`**

```python
# scripts/build_dataset.py
"""Turn cached VitalDB cases into a labeled feature table."""
import pandas as pd
from twin.data.vitaldb_loader import load_numeric_frame, _default_loader
from twin.data.labeling import make_map_series, label_windows
from twin.data.features import extract_features


def build_windows_for_case(caseid, static_row, loader_fn=_default_loader) -> pd.DataFrame:
    """Load one case, label its windows, and attach features. Returns a DataFrame."""
    frame = load_numeric_frame(caseid, loader_fn=loader_fn)
    map_series = make_map_series(frame)
    windows = label_windows(map_series)
    if windows.empty:
        return windows
    rows = []
    for _, w in windows.iterrows():
        feats = extract_features(frame, int(w.t_end), dict(static_row))
        feats["caseid"] = caseid
        feats["t_end"] = int(w.t_end)
        feats["y"] = int(w.y)
        feats["category"] = w.category
        rows.append(feats)
    return pd.DataFrame(rows)


def build_dataset(eligible_cases: pd.DataFrame, loader_fn=_default_loader) -> pd.DataFrame:
    """Build the full window table across all eligible cases."""
    parts = []
    for _, row in eligible_cases.iterrows():
        try:
            part = build_windows_for_case(int(row["caseid"]), row.to_dict(), loader_fn=loader_fn)
            if not part.empty:
                parts.append(part)
        except Exception as exc:  # skip unreadable cases, keep going
            print(f"  skip case {row['caseid']}: {exc}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_dataset_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Write `scripts/run_baseline.py`**

```python
# scripts/run_baseline.py
"""Train baselines and run the biased-vs-unbiased selection-bias study."""
import pandas as pd
import twin.config as c
from twin.data.splits import make_splits
from twin.models.baselines import MapOnlyModel, GbdtModel
from twin.eval.selection_bias import make_biased, make_unbiased, evaluate

FEATURE_DROP = {"caseid", "t_end", "y", "category"}


def _xy(df):
    X = df[[col for col in df.columns if col not in FEATURE_DROP]]
    return X, df["y"].to_numpy()


def run(windows: pd.DataFrame, case_meta: pd.DataFrame) -> pd.DataFrame:
    assign = make_splits(case_meta)
    windows = windows.copy()
    windows["split"] = windows["caseid"].map(assign)
    train = windows[windows.split == "train"]
    test = windows[windows.split == "test"]

    Xtr, ytr = _xy(train)
    results = []
    for name, model in [("map_only", MapOnlyModel()), ("gbdt", GbdtModel())]:
        model.fit(Xtr, ytr)
        for regime, subset in [("biased", make_biased(test)), ("unbiased", make_unbiased(test))]:
            Xte, yte = _xy(subset)
            p = model.predict_proba(Xte)
            row = {"model": name, "regime": regime}
            row.update(evaluate(yte, p))
            results.append(row)
    out = pd.DataFrame(results)
    c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(c.RESULTS_DIR / "selection_bias_results.csv", index=False)
    return out


if __name__ == "__main__":  # pragma: no cover
    windows = pd.read_parquet(c.RESULTS_DIR / "windows.parquet")
    case_meta = pd.read_csv(c.RESULTS_DIR / "eligible_cases.csv")
    print(run(windows, case_meta).to_string(index=False))
```

- [ ] **Step 6: Run full test suite**

Run: `pytest -q`
Expected: PASS (all tests green)

- [ ] **Step 7: Commit**

```bash
git add scripts/build_dataset.py scripts/run_baseline.py tests/test_build_dataset_smoke.py
git commit -m "feat: dataset-build and baseline/selection-bias runner"
```

---

### Task 13: EDA & results notebooks + end-to-end subset run

**Files:**
- Create: `notebooks/01_eda.ipynb`
- Create: `notebooks/02_results.ipynb`
- Create: `scripts/run_all_subset.py`

- [ ] **Step 1: Write `scripts/run_all_subset.py`** (the real end-to-end driver on a small subset)

```python
# scripts/run_all_subset.py
"""End-to-end on a small subset: cohort -> windows -> baselines -> results."""
import argparse
import pandas as pd
import twin.config as c
from scripts.run_cohort import build_cohort
from scripts.build_dataset import build_dataset
from scripts.run_baseline import run


def main(n_cases: int):
    eligible, funnel = build_cohort()
    subset = eligible.head(n_cases)
    print(f"Building windows for {len(subset)} cases...")
    windows = build_dataset(subset)
    c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(c.RESULTS_DIR / "windows.parquet")
    print(f"Windows: {len(windows)} | positives: {int(windows.y.sum())} "
          f"({100 * windows.y.mean():.1f}%)")
    results = run(windows, subset)
    print(results.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=300)
    main(ap.parse_args().n_cases)
```

- [ ] **Step 2: Run the end-to-end subset driver** (requires accepted VitalDB DUA + network)

Run: `python -m scripts.run_all_subset --n-cases 50`
Expected: prints the cohort funnel, a window count with IOH base rate (sanity: a few % to low tens of %), and a 4-row results table (`map_only`/`gbdt` × `biased`/`unbiased`). First run downloads ~50 cases to `data_cache/`.

If `vitaldb` import or download fails, fix access before continuing (accept the DUA at vitaldb.net, confirm `pip install vitaldb`). Do not fake data here — this step's purpose is to prove real access works.

- [ ] **Step 3: Create `notebooks/01_eda.ipynb`** with these cells (use the cached outputs):

Cell 1 (markdown): `# SP1 EDA — cohort funnel, base rate, gray-zone fraction`
Cell 2 (code):
```python
import pandas as pd, matplotlib.pyplot as plt
import twin.config as c
funnel = pd.read_csv(c.RESULTS_DIR / "cohort_funnel.csv")
windows = pd.read_parquet(c.RESULTS_DIR / "windows.parquet")
funnel
```
Cell 3 (code):
```python
print("IOH base rate:", round(100 * windows.y.mean(), 2), "%")
windows.category.value_counts(normalize=True).mul(100).round(1)
```
Cell 4 (code):
```python
windows.category.value_counts().plot.bar(title="Window categories")
plt.tight_layout()
```

- [ ] **Step 4: Create `notebooks/02_results.ipynb`** with these cells:

Cell 1 (markdown): `# SP1 Results — biased vs unbiased baselines`
Cell 2 (code):
```python
import pandas as pd
import twin.config as c
res = pd.read_csv(c.RESULTS_DIR / "selection_bias_results.csv")
res.pivot_table(index="model", columns="regime", values=["auroc", "auprc", "ppv"]).round(3)
```
Cell 3 (markdown): Write 3-4 sentences interpreting the biased-vs-unbiased PPV gap and the MAP-only-vs-GBDT comparison, citing Yang/Celi 2025 and Chaari 2025.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_all_subset.py notebooks/01_eda.ipynb notebooks/02_results.ipynb
git commit -m "feat: end-to-end subset driver and EDA/results notebooks"
```

---

## Self-Review (completed)

**Spec coverage:** §3 architecture → Tasks 1-12 file-for-file. §4 data flow → Tasks 8 (load), 2 (label), 7 (features), 4 (splits), 12 (assemble). §5 models + selection bias + metrics → Tasks 9, 10, 5, 6. §6 testing → every logic task is TDD; labeling/splits/metrics have dedicated tests. §6 reproducibility → `config.SEED` everywhere, parquet cache keyed by caseid, funnel + config saved. §7 deliverables → Task 13 (EDA notebook, results notebook, written finding). Prototype-then-scale → Task 13 `--n-cases`. No gaps.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step shows the exact command + expected output.

**Type consistency:** Window schema `caseid,t_end,y,category` is produced in Task 12 and consumed identically in Tasks 10/12. `predict_proba` returns a 1-D array in Task 9, consumed that way in Tasks 5/6/10/12. `make_splits` returns `dict[caseid->split]`, consumed via `.map()` in Task 12. `load_numeric_frame(caseid, loader_fn=...)` signature matches across Tasks 8/12. Consistent.

**Known real-world caveats to watch during execution (not plan defects):**
- VitalDB `department` values must match `EXCLUDED_DEPARTMENTS` exactly; verify actual category strings during the Task 13 run and adjust `config.py` if needed.
- `cases.csv` column names (`preop_sbp`, `preop_dbp`, `preop_htn`, `ane_type`, `casestart/caseend`) are per the VitalDB schema; confirm on first download.
