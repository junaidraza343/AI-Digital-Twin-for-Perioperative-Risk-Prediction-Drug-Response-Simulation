"""Admin / ML-ops endpoints: status, metrics, curves, and training jobs.

Primarily a local developer console (training that touches VitalDB needs network
+ certifi). Read-only status/metrics/curves work anywhere the artifacts exist.
"""
import os
import re
import secrets
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Query

import twin.config as c

router = APIRouter(prefix="/api/admin")
REPO = c.REPO_ROOT

# In-memory job registry for background training runs.
_JOBS: dict[str, dict] = {}


# ------------------------------------------------------------------ auth
ADMIN_TOKEN_ENV = "ADMIN_TOKEN"
MAX_BUILD_CASES = 2000


def require_admin(x_admin_token: str | None = Header(default=None)):
    """Gate every endpoint that spawns work.

    These endpoints start training threads and a VitalDB subprocess, so on a
    public deployment an unauthenticated caller could exhaust the host's CPU and
    network. Fails CLOSED: with no ADMIN_TOKEN configured the controls are
    disabled outright rather than left open, so shipping without setting the
    variable cannot silently expose them.
    """
    expected = os.environ.get(ADMIN_TOKEN_ENV)
    if not expected:
        raise HTTPException(503, "Admin controls are disabled: ADMIN_TOKEN is not "
                                 "configured on this deployment.")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(401, "Invalid or missing X-Admin-Token.")
    return True


# ------------------------------------------------------------------ helpers
def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe records (NaN -> None)."""
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def _read_csv(name: str):
    p = c.RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


# ------------------------------------------------------------------ status
@router.get("/status")
def status():
    n_windows = prevalence = None
    wpath = c.RESULTS_DIR / "windows.parquet"
    if wpath.exists():
        w = pd.read_parquet(wpath, columns=["y"])
        n_windows, prevalence = int(len(w)), round(float(w["y"].mean()), 4)
    n_cached = len(list(c.DATA_CACHE.glob("*.parquet"))) if c.DATA_CACHE.exists() else 0

    funnel_df = _read_csv("cohort_funnel.csv")
    from twin.predict import load_predictor, MODEL_PATH
    from twin.data.case_api import available_cases
    model, meta = load_predictor()

    last_train = artifact = None
    if MODEL_PATH.exists():
        import hashlib
        mtime = MODEL_PATH.stat().st_mtime
        mins = int((time.time() - mtime) / 60)
        last_train = "just now" if mins < 1 else f"{mins}m ago" if mins < 120 else f"{mins // 60}h ago"
        artifact = "sha " + hashlib.sha1(MODEL_PATH.read_bytes()).hexdigest()[:8]
    return {
        "cached_cases": n_cached,
        "n_windows": n_windows,
        "prevalence": prevalence,
        "funnel": _records(funnel_df) if funnel_df is not None else [],
        "model_trained": model is not None,
        "model_meta": meta or {},
        "demo_cases": len(available_cases()),
        "seed": c.SEED,
        "horizon_min": c.HORIZON_SECONDS // 60,
        "map_threshold": c.MAP_THRESHOLD,
        "runtime": "uvicorn · local",
        "last_train": last_train,
        "artifact": artifact,
    }


# ------------------------------------------------------------------ metrics
@router.get("/metrics")
def metrics():
    sb = _read_csv("selection_bias_results.csv")
    if sb is None:
        return {"available": False, "rows": []}
    dn = _read_csv("deepnet_results.csv")   # optional GPU deep-net rows
    if dn is not None:
        sb = pd.concat([sb, dn], ignore_index=True)
    return {"available": True, "rows": _records(sb)}


# ------------------------------------------------------------------ curves
def _train_both(unbiased_only=True):
    """Train MAP-only + GBDT on the train split; return (models, unbiased test frame)."""
    from twin.data.splits import make_splits
    from twin.models.baselines import MapOnlyModel, GbdtModel
    from twin.eval.selection_bias import make_unbiased

    windows = pd.read_parquet(c.RESULTS_DIR / "windows.parquet")
    case_meta = pd.read_csv(c.RESULTS_DIR / "eligible_cases.csv")
    assign = make_splits(case_meta)
    windows = windows.copy()
    windows["split"] = windows["caseid"].map(assign)
    train = windows[windows.split == "train"]
    test = windows[windows.split == "test"]
    drop = {"caseid", "t_end", "y", "category", "split"}
    Xtr = train[[col for col in train.columns if col not in drop]]
    ytr = train["y"].to_numpy()
    models = {"map_only": MapOnlyModel().fit(Xtr, ytr),
              "gbdt": GbdtModel().fit(Xtr, ytr)}
    return models, make_unbiased(test), drop


# Curves cost a full retrain of both models over every window (~9 s on 4 cores,
# noticeably worse on a small cloud instance), and the result only changes when
# the predictor is retrained. Cache it; /train-predictor clears it.
_CURVES_CACHE: dict = {}


@router.get("/curves")
def curves():
    if not (c.RESULTS_DIR / "windows.parquet").exists():
        return {"available": False}
    if "payload" in _CURVES_CACHE:
        return _CURVES_CACHE["payload"]
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    models, unbiased, drop = _train_both()
    Xu = unbiased[[col for col in unbiased.columns if col not in drop]]
    yu = unbiased["y"].to_numpy()

    def thin(a, k=120):
        a = np.asarray(a, float)
        if len(a) <= k:
            return a.round(4).tolist()
        idx = np.linspace(0, len(a) - 1, k).astype(int)
        return a[idx].round(4).tolist()

    out = {"available": True, "prevalence": round(float(yu.mean()), 4), "models": {}}
    for name, m in models.items():
        p = m.predict_proba(Xu)
        fpr, tpr, _ = roc_curve(yu, p)
        prec, rec, _ = precision_recall_curve(yu, p)
        # reliability: 10 equal-width bins
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        frac, conf, cnt = [], [], []
        for b in range(10):
            mask = idx == b
            if mask.sum():
                frac.append(round(float(yu[mask].mean()), 4))
                conf.append(round(float(p[mask].mean()), 4))
                cnt.append(int(mask.sum()))
        out["models"][name] = {
            "auroc": round(float(roc_auc_score(yu, p)), 4),
            "auprc": round(float(average_precision_score(yu, p)), 4),
            "roc": {"fpr": thin(fpr), "tpr": thin(tpr)},
            "pr": {"recall": thin(rec), "precision": thin(prec)},
            "reliability": {"conf": conf, "frac": frac, "count": cnt},
        }
    dca = _read_csv("decision_curve.csv")
    if dca is not None:
        out["decision_curve"] = _records(dca)
    _CURVES_CACHE["payload"] = out
    return out


# ------------------------------------------------------------------ training (async)
def _run_train(job_id: str):
    """Retrain MAP-only + GBDT, regenerate the selection-bias table, persist predictor."""
    from scripts.run_baseline import run
    from scripts.train_predictor import train
    try:
        windows = pd.read_parquet(c.RESULTS_DIR / "windows.parquet")
        case_meta = pd.read_csv(c.RESULTS_DIR / "eligible_cases.csv")
        table = run(windows, case_meta)      # both models, all regimes -> csv
        meta = train()                       # persist MAP-only predictor + meta
        _CURVES_CACHE.clear()                # curves now reflect a stale model
        # refresh the in-process predictor used by the clinical UI
        import api.main as main_mod
        from twin.predict import load_predictor
        main_mod.PREDICTOR, main_mod.PREDICTOR_META = load_predictor()
        _JOBS[job_id].update(status="done", rows=_records(table), predictor_meta=meta)
    except Exception as exc:                 # surface the reason in the console
        _JOBS[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")
    _JOBS[job_id]["ended"] = time.time()


@router.post("/train-predictor")
def train_predictor_endpoint(_: bool = Depends(require_admin)):
    """Kick off a retrain in the background and return a job id to poll.

    A full retrain is ~8 min on 4 cores and longer on a small cloud instance —
    well past the request timeout most hosting proxies enforce — so this cannot
    be a synchronous POST if the deployed console is to work.
    """
    if not (c.RESULTS_DIR / "windows.parquet").exists():
        raise HTTPException(400, "No windows.parquet — build a dataset first.")
    job_id = uuid.uuid4().hex[:8]
    _JOBS[job_id] = {"status": "running", "kind": "train", "started": time.time()}
    threading.Thread(target=_run_train, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@router.post("/rescan-cases")
def rescan_cases(_: bool = Depends(require_admin)):
    from twin.data.case_api import scan_demo_cases
    cat = scan_demo_cases()
    return {"ok": True, "n": len(cat)}


# ------------------------------------------------------------------ dataset build (async)
def _run_build(n_cases: int, job_id: str, log_path: Path):
    env = os.environ.copy()
    env["TWIN_PROGRESS"] = "1"          # emit "PROGRESS k/N" lines for the UI bar
    env["PYTHONUNBUFFERED"] = "1"
    try:
        import certifi
        env["SSL_CERT_FILE"] = certifi.where()
        env["REQUESTS_CA_BUNDLE"] = certifi.where()
    except Exception:
        pass
    with open(log_path, "w") as f:
        proc = subprocess_popen(n_cases, f, env)
        _JOBS[job_id]["pid"] = proc.pid
        rc = proc.wait()
    _JOBS[job_id]["status"] = "done" if rc == 0 else "failed"
    _JOBS[job_id]["returncode"] = rc
    _JOBS[job_id]["ended"] = time.time()
    if rc == 0:
        # A successful build rewrites windows.parquet, so cached curves and the
        # memoized case lists no longer describe the dataset on disk.
        from twin.data.case_api import _all_cases, available_cases
        _CURVES_CACHE.clear()
        _all_cases.cache_clear()
        available_cases.cache_clear()


def subprocess_popen(n_cases, f, env):
    import subprocess
    return subprocess.Popen(
        [sys.executable, "-m", "scripts.run_all_subset", "--n-cases", str(n_cases)],
        cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, env=env,
    )


@router.post("/build")
def build(n_cases: int = Query(100, ge=5, le=MAX_BUILD_CASES),
          _: bool = Depends(require_admin)):
    job_id = uuid.uuid4().hex[:8]
    log_path = c.RESULTS_DIR / f"build_{job_id}.log"
    c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _JOBS[job_id] = {"status": "running", "kind": "build", "n_cases": n_cases,
                     "log_path": str(log_path), "started": time.time()}
    threading.Thread(target=_run_build, args=(n_cases, job_id, log_path), daemon=True).start()
    return {"job_id": job_id, "n_cases": n_cases}


@router.get("/job/{job_id}")
def job(job_id: str):
    j = _JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    log = ""
    lp = Path(j["log_path"]) if j.get("log_path") else None
    if lp is not None and lp.exists():
        log = lp.read_text()[-4000:]
    # progress parse from the pipeline's "PROGRESS k/N" lines
    stage, percent = "starting", 0
    if "Building windows" in log:
        stage = "building windows"
    matches = re.findall(r"PROGRESS (\d+)/(\d+)", log)
    if matches:
        k, n = matches[-1]
        percent = int(round(90 * int(k) / max(1, int(n))))  # windows = 0-90%
        stage = f"building windows · {k}/{n} cases"
    if "Windows:" in log:
        stage, percent = "training + evaluating", 95
    if j.get("kind") == "train":
        # No subprocess log to parse; report an indeterminate but honest stage.
        stage, percent = "training MAP-only + GBDT across regimes", 50
    if j["status"] == "done":
        stage, percent = "complete", 100
    if j["status"] == "failed":
        stage = "failed"
    return {"status": j["status"], "stage": stage, "percent": percent,
            "n_cases": j.get("n_cases"), "error": j.get("error"),
            "rows": j.get("rows"), "predictor_meta": j.get("predictor_meta"),
            "elapsed": round(time.time() - j["started"], 1), "log": log}
