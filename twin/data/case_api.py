"""Serve real VitalDB cases to the twin UI for grounding against actual MAP.

Reuses the tested training pipeline (cached numeric frame -> clean MAP series) so
it works offline for any case already downloaded to data_cache/. Selects an
informative ~15-min window (preferring one containing a real hypotensive dip),
and scores it with the trained predictor for a real IOH-probability on real data.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

import twin.config as c
from twin.data.vitaldb_loader import load_numeric_frame, cached_caseids
from twin.data.labeling import make_map_series

WINDOW_S = 900          # 15 min shown
MAP_LO, MAP_HI = 40.0, 200.0   # <40 mmHg treated as arterial-line artifact
DEMO_CASES_PATH = c.RESULTS_DIR / "demo_cases.json"


@lru_cache(maxsize=1)
def _all_cases():
    path = c.RESULTS_DIR / "windows.parquet"
    if not path.exists():
        return []
    return sorted(int(x) for x in pd.read_parquet(path, columns=["caseid"])["caseid"].unique())


@lru_cache(maxsize=1)
def available_cases():
    """Curated demo cases (clean baseline + real dip) if scanned, else all loadable."""
    if DEMO_CASES_PATH.exists():
        return [d["caseid"] for d in json.loads(DEMO_CASES_PATH.read_text())]
    return _all_cases()


def demo_case_catalog():
    """List of {caseid, baseline, dip_min} for the UI dropdown."""
    if DEMO_CASES_PATH.exists():
        return json.loads(DEMO_CASES_PATH.read_text())
    return [{"caseid": cid} for cid in _all_cases()]


def scan_demo_cases(max_cases: int = 40) -> list:
    """Offline scan of cached cases for ones with a sane baseline and a real dip.

    Writes results/demo_cases.json. Run once after a data build.
    """
    # Offline means offline: only consider cases already on disk for the current
    # track set. Walking every eligible case would download the misses, which on a
    # deployment that bundles a handful of cases is thousands of requests.
    on_disk = cached_caseids()
    out = []
    for cid in (cid for cid in _all_cases() if cid in on_disk):
        try:
            v = _clean(make_map_series(load_numeric_frame(cid)))
            v = v[~np.isnan(v)]
            if len(v) < 1200:
                continue
            base = float(np.percentile(v, 60))
            dip = float(np.percentile(v, 2))
            if 72 <= base <= 108 and 45 <= dip <= 64:
                out.append({"caseid": int(cid), "baseline": round(base), "dip_min": round(dip)})
        except Exception:
            continue
    out.sort(key=lambda d: d["dip_min"])
    out = out[:max_cases]
    c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_CASES_PATH.write_text(json.dumps(out, indent=2))
    available_cases.cache_clear()
    return out


def _clean(series: np.ndarray) -> np.ndarray:
    s = np.asarray(series, dtype=float).copy()
    s[(s < MAP_LO) | (s > MAP_HI)] = np.nan
    # forward-fill short gaps (<= 3 min), then back-fill the lead-in
    ser = pd.Series(s).ffill(limit=180).bfill(limit=180)
    return ser.to_numpy()


def _select_window(map_clean: np.ndarray) -> int:
    """Slide to the window with a healthy baseline that also contains a real dip.

    Scores each candidate by (baseline + dip depth); rejects windows whose robust
    floor looks like an artifact. Falls back to the first valid window.
    """
    valid = ~np.isnan(map_clean)
    if not valid.any():
        return 0
    first_valid = int(np.argmax(valid))
    last = len(map_clean) - WINDOW_S
    if last <= first_valid:
        return first_valid

    best, best_score = first_valid, -np.inf
    for start in range(first_valid, last + 1, 60):
        w = map_clean[start:start + WINDOW_S]
        w = w[~np.isnan(w)]
        if len(w) < WINDOW_S * 0.6:
            continue
        base = np.percentile(w, 60)
        floor = np.percentile(w, 2)
        if floor < 40 or not (72 <= base <= 110):
            continue
        has_dip = float(np.min(w)) < c.MAP_THRESHOLD
        if not has_dip:
            continue
        score = base + (c.MAP_THRESHOLD - min(float(np.min(w)), c.MAP_THRESHOLD))
        if score > best_score:
            best_score, best = score, start
    return best


def _patient_row(caseid: int) -> dict:
    path = c.RESULTS_DIR / "eligible_cases.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    row = df[df["caseid"] == caseid]
    return row.iloc[0].to_dict() if len(row) else {}


def load_case_bundle(caseid: int, model=None, max_points: int = 450) -> dict:
    """Return real MAP window, patient covariates, and (optional) predictor prob."""
    frame = load_numeric_frame(caseid)
    map_series = _clean(make_map_series(frame))
    start = _select_window(map_series)
    win = map_series[start:start + WINDOW_S]
    win = win[~np.isnan(win)] if np.isnan(win).all() else np.nan_to_num(win, nan=float(np.nanmedian(win)))

    t = np.arange(len(win))
    prob = None
    if model is not None and len(win) > 0:
        from twin.predict import ioh_probability
        prob = ioh_probability(win, dt=1.0, model=model)

    # downsample for transport
    stride = max(1, len(win) // max_points)
    sl = slice(None, None, stride)

    row = _patient_row(caseid)
    # baseline = robust non-hypotensive MAP (70th percentile of the window),
    # clamped to the twin's valid range. Represents the patient's starting MAP
    # before the propofol-driven fall.
    base = float(np.percentile(win, 70)) if len(win) else 90.0
    map0 = float(np.clip(round(base, 1), 62.0, 118.0))
    patient = {
        "age": float(row.get("age", 60)), "weight": float(row.get("weight", 70)),
        "height": float(row.get("height", 170)),
        "sex": "M" if str(row.get("sex", "M")).upper().startswith(("M", "1")) else "F",
        "map0": round(map0, 1),
    }
    out = {
        "caseid": int(caseid),
        "t_min": (t[sl] / 60.0).round(3).tolist(),
        "map_real": win[sl].round(2).tolist(),
        "patient": patient,
        "min_map": round(float(np.min(win)), 1) if len(win) else None,
        "has_breach": bool(np.any(win < c.MAP_THRESHOLD)) if len(win) else False,
    }
    if prob is not None:
        out["prob_real"] = prob[sl].round(4).tolist()
        out["peak_prob"] = round(float(np.max(prob)), 4)
    return out
