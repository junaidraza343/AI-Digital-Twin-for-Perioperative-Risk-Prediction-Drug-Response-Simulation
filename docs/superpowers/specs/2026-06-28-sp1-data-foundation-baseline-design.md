# SP1 — Data Foundation + Tabular Baseline + Selection-Bias Study

**Date:** 2026-06-28
**Project:** AI Digital Twin for Personalized Intraoperative Risk Prediction and Drug Response Simulation (FYP)
**Sub-project:** SP1 of 5 (see Decomposition below)
**Status:** Design approved; pending implementation plan

---

## 1. Context

The FYP builds a perioperative digital twin coupling a deep-learning intraoperative-hypotension (IOH) predictor with a personalized PK-PD engine. The full system is too large for one build. It decomposes into five sub-projects:

| # | Sub-project | Compute | Depends on |
|---|---|---|---|
| **SP1** | **Data foundation + tabular baseline + selection-bias study** | **CPU** | — |
| SP2 | Deep waveform predictor (CNN-GRU on 100 Hz signals) | GPU | SP1 |
| SP3 | Mechanistic PK-PD engine (Eleveld propofol + Joachim norepinephrine) | CPU | — |
| SP4 | Calibration head + coupling (the novel contribution) | GPU | SP2, SP3 |
| SP5 | What-if engine + decision-support dashboard | CPU | SP4 |

**Constraints driving this design:** developer machine is an Intel MacBook Pro (i9, 32 GB RAM, no CUDA GPU); developer is fluent in Python/pandas/PyTorch; next FYP checkpoint is a few weeks away. SP1 is the first build because it is CPU-friendly, runnable fast, showable, and de-risks the riskiest assumptions (data access, label definition, selection-bias inflation) before any GPU time is spent.

**Research motivation:** 2025 primary work (Yang/Celi/Lee, *Br J Anaesth* 2025;135(3):571-581; Chaari et al., *J Clin Monit Comput* 2025) shows selection bias massively inflates reported VitalDB IOH performance (5-min PPV 0.937 biased vs 0.068 unbiased) and that deep learning offers no clinically meaningful advantage over a simple MAP-only model on unbiased data — except in the 65-75 mmHg gray zone. SP1 reproduces this on our own cohort and establishes the mandatory MAP-only baseline.

## 2. Scope & Success Criteria

### Delivers
- Reproducible VitalDB pipeline: lazy per-case loading (no 110 GB bulk download), cohort filtering to the ~4,200-case spec, tested MAP<65 labeler with gray-zone categorization.
- Two baselines at the 5-min horizon: MAP-only model (mandatory benchmark) and gradient-boosted trees on static covariates + 1 Hz numeric-trend features. No 100 Hz waveforms; all CPU.
- Selection-bias evaluation harness reporting every model under biased vs unbiased sampling with AUROC / AUPRC / PPV / calibration / decision-curve.
- EDA + results report for the checkpoint.

### Success = all true
1. Pipeline turns 6,388 raw cases into the filtered cohort deterministically, with a documented funnel.
2. Both baselines train and evaluate on CPU in minutes-to-an-hour on a working subset.
3. The biased-vs-unbiased gap is quantified on our cohort, reproducing the 2025 selection-bias finding and establishing the MAP-only baseline.

### Explicitly deferred
All 100 Hz waveform modeling and the CNN-GRU (SP2); the PK-PD engine (SP3); the calibration head/coupling (SP4); the dashboard (SP5). SP1 touches none of these.

### Prototype-then-scale
Develop on a ~300-500 case subset for fast CPU iteration, then scale to the full filtered cohort once the pipeline is verified.

## 3. Architecture

Fresh git repo, Python package layout:

```
fyp-twin/
  twin/
    config.py            # paths, seed, horizon=5min, MAP threshold=65, obs window, stride
    data/
      vitaldb_loader.py  # API wrapper: lazy per-case load + on-disk cache
      cohort.py          # filter cases.csv -> eligible caseids + funnel log
      labeling.py        # 1Hz MAP series -> windows -> y(t) + gray-zone class
      features.py        # static covariates + numeric-trend window features
      splits.py          # case-level stratified 70/15/15, fixed seed
    models/
      baselines.py       # MAP-only logistic + GBDT (LightGBM)
    eval/
      metrics.py         # AUROC, AUPRC, PPV@rate, ECE, Brier
      decision_curve.py  # net-benefit analysis
      selection_bias.py  # biased vs unbiased eval harness
  scripts/               # run_cohort.py, run_baseline.py (thin CLIs)
  notebooks/             # 01_eda.ipynb, 02_results.ipynb
  tests/                 # labeling, splits-leakage, metrics, cohort-filter
  data_cache/            # gitignored downloaded arrays
  docs/superpowers/specs/
```

Each module has one job and a clean interface (e.g., `labeling.label_case(map_series) -> windows_df`) so each is independently testable. The labeler and split logic are the two places a bug would silently corrupt all results.

## 4. Data Flow

```
cases.csv (6,388 rows)
  -> cohort.py: age>=18; dept not in {Cardiac,Thoracic,Gyn}; ane_type~General;
                duration>=1800s; ASA not in {5,6}; preop MAP>=65
  -> eligible caseids (~4,200) + funnel log + static covariates per case
  -> vitaldb_loader.py: per case, load ONLY 1 Hz numeric tracks
       (ART_MBP->fallback NIBP_MBP, ART_SBP/DBP, HR, SpO2, ETCO2, RR, BT)
       via vitaldb.load_case(caseid, tracks, interval=1); cache to data_cache/
  -> per-case 1 Hz numeric frames
  -> labeling.py: build MAP series; roll windows (obs=10min, stride=30s);
       drop windows where current MAP<65 (event already underway);
       y(t)=1 if MAP<65 for >=60 cumulative sec in (t, t+5min];
       tag each window: stable / gray-zone(65-75) / overt
  -> labeled windows (caseid, t, y, category)
  -> features.py: static covariates + trend features over obs window
       (mean/std/min/max/slope/last per numeric track; current MAP & MAP slope)
  -> feature matrix X, label y, group=caseid, category
  -> splits.py: case-level stratified 70/15/15 (ASA x age-decile x HTN-med), fixed seed
  -> train / val / test -> models + selection-bias eval harness
```

Key design points:
- Dropping windows where MAP is already <65 keeps the task *prediction*, not *detection* (common silent bug).
- Loading only 1 Hz numeric tracks (not 500 Hz waveforms) keeps storage/compute laptop-sized: a filtered case is ~KB of numerics, not ~MB of waveforms.

## 5. Modeling, Selection-Bias Study & Metrics

### Models (both CPU)
1. **MAP-only baseline** — logistic regression on `current MAP` + `MAP slope` over the obs window. The "is the fancy model better than watching the MAP line?" benchmark.
2. **GBDT (LightGBM)** — full static + trend feature set. Native missingness handling, fast on CPU.

### Selection-bias study
Evaluate every model twice:
- **Biased regime** — test set built like the old HPI protocol: only clearly-stable vs clearly-overt windows, gray zone excluded -> inflated PPV.
- **Unbiased regime** — all windows at realistic prevalence, gray zone included.

Report per (model x regime): AUROC, AUPRC, PPV at a fixed alarm rate, ECE + reliability curve, Brier, decision-curve net benefit; plus the MAP-only-vs-GBDT gap in each regime and metrics on the gray-zone subset specifically. Expected, defensible result: the gap collapses on unbiased data except in the gray zone (Yang/Celi 2025; Chaari 2025), reproduced on our cohort.

Operating points are chosen on validation, never test. All randomness flows from one logged seed.

## 6. Testing & Reproducibility

- **Labeling tests** — synthetic 1 Hz MAP series with known events: 90 s dip <65 -> positive; 40 s dip -> negative (fails 60 s rule); dip starting before window -> excluded; gray-zone-only window -> tagged gray, label 0.
- **Split-leakage test** — no caseid in more than one of train/val/test; stratification proportions hold within tolerance.
- **Metrics tests** — AUROC/AUPRC/ECE/decision-curve vs hand-computed values on toy inputs.
- **Cohort-filter test** — synthetic cases.csv rows exercising each include/exclude rule.
- **Reproducibility** — pin `vitaldb` version; single fixed seed logged per run; cache keyed by `(caseid, tracks, interval)`; cohort funnel + config saved with results so every reported number is traceable.

## 7. Checkpoint Deliverables

1. **EDA report** (`01_eda.ipynb`) — cohort funnel (6,388 -> ~4,200), IOH base rate, class balance, gray-zone fraction, missingness summary.
2. **Results report** (`02_results.ipynb`) — master table (model x regime x metrics) + figures: ROC, PR, calibration/reliability, decision curve, biased-vs-unbiased gap chart.
3. **Written finding** tying numbers to Yang/Celi 2025 + Chaari 2025.
4. **Tested, documented pipeline** that SP2 (waveform CNN-GRU) plugs into once a GPU is available: same cohort, same splits, same labeler.

## 8. Open Items / Assumptions

- Assumes VitalDB click-through DUA accepted and `pip install vitaldb` works (no credentialing needed for VitalDB itself).
- Obs window (10 min) and stride (30 s) are starting values; may be tuned during EDA.
- Exact filtered cohort size depends on edge-case handling and will be reported from the funnel, not assumed to be exactly 4,200.
- GBDT library: LightGBM (fallback to scikit-learn HistGradientBoosting if install friction on this machine).
