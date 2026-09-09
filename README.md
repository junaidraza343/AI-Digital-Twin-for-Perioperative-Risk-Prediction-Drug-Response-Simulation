---
title: Perioperative Digital Twin
emoji: 🫀
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 8000
pinned: false
---

# AI Digital Twin for Perioperative Risk Prediction & Drug-Response Simulation

A patient-specific **perioperative digital twin** that couples a data-driven
**intraoperative-hypotension (IOH) predictor** with a mechanistic **PK-PD engine**
(Eleveld propofol + Joachim norepinephrine), exposed through a clinical what-if
monitor and an ML-Ops console. Decision support — **not** closed-loop control.

Final-year project (BS AI). Primary dataset: **VitalDB**.

---

## What's in the box

| Component | What it does |
|---|---|
| **Clinical monitor** (`/`) | Interactive what-if twin: set patient + propofol/norepinephrine dosing, watch the **projected MAP** vs the 65 mmHg threshold, with a trained **ML IOH-probability**, a **confidence band**, **effect-site Ce**, population-vs-**personalized** twin, and **real VitalDB case** grounding. |
| **ML-Ops console** (`/admin`) | Training pipeline + model performance: selection-bias study table (model × regime), ROC/PR/calibration/decision-curve plots, one-click **retrain**, and a **background VitalDB dataset build** with a live progress bar. |
| **PK-PD engine** (`twin/pkpd/`) | Mechanistic propofol/norepinephrine → MAP simulator with a 6-δ personalization hook. |
| **Predictors** (`twin/models/`) | MAP-only logistic, ElasticNet (full-feature), Random Forest, GBDT, and a **GPU-capable PyTorch deep net**. |

## Repo layout

```
twin/            # core package
  pkpd/          # mechanistic PK-PD engine (params, propofol, norepi, pd_model, engine, fitting)
  models/        # baselines.py (sklearn) + deepnet.py (PyTorch)
  data/          # vitaldb_loader, cohort, labeling, features, splits, case_api
  eval/          # metrics, decision_curve, selection_bias
  predict.py     # run the trained IOH predictor on a MAP trajectory
api/             # FastAPI backend: main.py (clinical) + admin.py (ML-ops)
frontend/        # clinical monitor + admin console (HTML/CSS/canvas JS)
dashboard/       # Streamlit fallback UI
scripts/         # run_cohort, build_dataset, run_baseline, run_all_subset,
                 #   train_predictor, train_deepnet
notebooks/       # 01_eda, 02_results, train_deepnet_colab (GPU)
tests/           # pytest suite
Dockerfile, docker-compose.yml
```

---

## Quick start (Docker — easiest)

Requires Docker Desktop (Linux containers; works on macOS, Linux, **and Windows**).

```bash
docker compose up --build
```

Open **http://localhost:8000** (clinical monitor) and **/admin** (ML-Ops console).
The image bundles the trained predictor, the training-window matrix, and 40
curated real VitalDB cases, so both consoles run **fully offline** — verified by
running the container with `--network none`. Only the *dataset build* reaches out
to VitalDB, since it downloads new cases.

---

## Live deployment (Hugging Face Spaces)

The hosted demo is the same image, running the full stack — clinical monitor and
a working ML-Ops console.

**Why Spaces:** free without a credit card, and 16 GB RAM — `/api/admin/curves`
holds 598 k × 64 windows in memory and retrains two models, which OOMs on a
512 MB free tier. It also doesn't cold-sleep the way Render's free tier does
(~50 s wake-up), so an evaluator's click lands on a live page.

```bash
git lfs install                       # windows.parquet (68 MB) is tracked via LFS
pip install huggingface_hub && huggingface-cli login
huggingface-cli repo create perioperative-twin --type space --space_sdk docker

git remote add space https://huggingface.co/spaces/<user>/perioperative-twin
git push space main
```

The Space reads its config from the YAML front-matter at the top of this file
(`sdk: docker`, `app_port: 8000`). First build takes ~5 min.

**Timings on the free 2-vCPU tier** (measured locally on 4 cores, so expect ~2×):

| Action | Cost |
|---|---|
| Clinical monitor, `/api/simulate`, real-case load | instant |
| `/admin` first load (trains both models for the curves) | ~10 s, then cached |
| **Retrain** (4 models × 3 regimes over 598 k windows) | **~8 min → ~20 min hosted** |
| Dataset build | minutes; scales with case count |

Retrain and build run as **polled background jobs**, not long requests — a
synchronous retrain would exceed the gateway timeout of every hosting proxy.
Start a retrain *before* the demo rather than clicking it live; the console shows
elapsed time while it runs and refreshes the tables when it lands.

---

## Local setup

Python **3.10**.

```bash
python3 -m pip install -r requirements.txt
```

> **macOS SSL note:** VitalDB access needs a valid CA bundle. If you hit
> `CERTIFICATE_VERIFY_FAILED`, prefix VitalDB commands with:
> ```bash
> export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
> export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
> ```

### Run the app (clinical monitor + ML-Ops console)

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- Clinical monitor: http://localhost:8000
- ML-Ops console:  http://localhost:8000/admin

### Streamlit fallback UI

```bash
python3 -m streamlit run dashboard/app.py    # http://localhost:8501
```

---

## Data & training pipeline

VitalDB is open (no login) — the pipeline lazily downloads only the 1 Hz numeric
tracks it needs and caches them in `data_cache/`.

```bash
# with the SSL env exported (see note above):

# 1. Build a labeled dataset from N VitalDB cases (cohort -> windows -> features)
python3 -m scripts.run_all_subset --n-cases 300      # writes results/windows.parquet + selection_bias_results.csv

# 2. Train the MAP-only predictor used live by the clinical UI
python3 -m scripts.train_predictor                   # writes results/predictor_map_only.joblib (+ meta)

# 3. Train the PyTorch deep net (uses CUDA > MPS > CPU automatically)
python3 -m scripts.train_deepnet --epochs 40         # writes results/deepnet.pt + deepnet_results.csv

# 4. Curate demo cases for the real-patient dropdown
python3 -c "from twin.data.case_api import scan_demo_cases; scan_demo_cases()"
```

Or do steps 1–2 from the **ML-Ops console** (`/admin`): **Build dataset** (live
progress bar) and **Retrain predictor + GBDT** (regenerates the selection-bias
table + curves). Note: *Build dataset replaces the current dataset.*

### GPU training on Colab

The deep net trains on any CUDA GPU. Open
`notebooks/train_deepnet_colab.ipynb` in Google Colab (**Runtime → GPU**), run it
to build data + train on GPU, then download `deepnet.pt` + `deepnet_results.csv`
into your local `results/`.

---

## The selection-bias study (headline result)

Every model is evaluated under three sampling regimes — **biased** (only clearly
stable vs clearly overt windows), **unbiased** (all windows at realistic
prevalence), and **gray-zone** (65–75 mmHg). Reproduces the 2025 literature: PPV
collapses from biased → unbiased, and complex models show **no clinically
meaningful gain over the MAP-only baseline** on unbiased data. See the table +
curves in `/admin`, or `results/selection_bias_results.csv`.

## External datasets (roadmap)

- **INSPIRE** (PhysioNet, ~130k perioperative cases, per-minute MAP) — large
  training expansion; needs PhysioNet credentialing + CITI training + DUA.
- **MOVER** (UC Irvine, ~59k surgical patients, US cohort) — **external
  validation**; needs a DUA.

---

## Tests

```bash
python3 -m pytest -q
```

## Scope & honesty notes

- Covariate scaling and the norepinephrine model are **documented reduced forms**
  of Eleveld 2018 / Joachim 2024 — adequate for a prototype, not the full models.
- The calibration δ are set manually or fit by optimization (a CPU stand-in for
  the learned SP4 calibration head).
- The deep **waveform** CNN-GRU (100 Hz, SP2) is the GPU-bound next step and is
  not included; the deep net here operates on the tabular window features.

## Tech stack

Python 3.10 · FastAPI · NumPy/Pandas · scikit-learn · PyTorch · vitaldb ·
Streamlit · Docker · vanilla HTML/CSS/canvas JS.

---

## Production deployment

```bash
docker build -t fyp-twin:prod .
docker run -d -p 8000:8000 --name fyp-twin fyp-twin:prod
```

Clinical monitor at `/`, ML-Ops console at `/admin`, liveness at `/health`.
The image bundles the trained predictor and the curated real VitalDB cases, so
the clinical monitor runs **fully offline**.

### Admin controls are disabled by default

`/api/admin/train-predictor`, `/api/admin/rescan-cases` and `/api/admin/build`
spawn compute — the last one launches a VitalDB download subprocess. They are
therefore **gated and fail closed**: with no `ADMIN_TOKEN` set they return `503`,
so a default deployment is read-only and cannot be used to burn someone else's
CPU or bandwidth.

To enable them:

```bash
docker run -d -p 8000:8000 -e ADMIN_TOKEN="$(openssl rand -hex 24)" fyp-twin:prod
```

Then paste the same value into **Admin access → Admin token** on `/admin`; the
browser stores it locally and sends it as `X-Admin-Token`. A wrong token gives
`401`. Read-only status, metrics and curves never require it.

Serve over HTTPS in front of the container — the token is sent as a plain header.

### Reproducibility

Image dependencies are pinned to the versions the reported results were produced
with. Bundled case files are keyed by a hash of `NUMERIC_TRACKS`; if you change
that list, re-run `python -m scripts.build_demo_cache` or the image will miss its
cache at runtime and try to reach VitalDB.
