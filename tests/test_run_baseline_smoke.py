"""Smoke test for scripts/run_baseline.run() — uses synthetic data."""
import numpy as np
import pandas as pd
import pytest
import twin.config as c
from scripts.run_baseline import run


def _make_synthetic(n_cases=60, rng_seed=0):
    """Build a minimal windows DataFrame + case_meta with both classes."""
    rng = np.random.default_rng(rng_seed)

    # case_meta: needs caseid, age, asa, preop_htn for make_splits
    case_meta = pd.DataFrame({
        "caseid": np.arange(n_cases),
        "age": rng.integers(30, 70, size=n_cases),
        "asa": rng.integers(1, 4, size=n_cases),
        "preop_htn": rng.integers(0, 2, size=n_cases).astype(bool),
    })

    # windows: each case gets ~5 windows
    rows = []
    categories = ["stable", "gray", "overt"]
    for cid in range(n_cases):
        for i in range(5):
            y = int(rng.integers(0, 2))
            cat = rng.choice(categories)
            rows.append({
                "caseid": int(cid),
                "t_end": 1000 + i * 30,
                "y": y,
                "category": cat,
                "map_last": float(rng.normal(80, 10)),
                "map_slope": float(rng.normal(-0.01, 0.05)),
                "hr_mean": float(rng.normal(70, 10)),
            })

    windows = pd.DataFrame(rows)
    return windows, case_meta


def test_run_baseline_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "RESULTS_DIR", tmp_path)

    windows, case_meta = _make_synthetic(n_cases=60)
    result = run(windows, case_meta)

    # Must return a DataFrame with all three regime names for both models
    assert isinstance(result, pd.DataFrame)
    assert set(result["regime"].unique()) >= {"biased", "unbiased", "gray_challenge"}
    models = {"map_only", "elastic_full", "random_forest", "gbdt"}
    assert set(result["model"].unique()) == models

    # Check every (model, regime) combination is present
    expected_pairs = {(m, r) for m in models
                      for r in ("biased", "unbiased", "gray_challenge")}
    actual_pairs = set(zip(result["model"], result["regime"]))
    assert expected_pairs == actual_pairs

    # selection_bias_results.csv must be written
    assert (tmp_path / "selection_bias_results.csv").exists()

    # decision_curve.csv must be written
    assert (tmp_path / "decision_curve.csv").exists()

    # decision_curve.csv should have content (model + threshold + net_benefit columns)
    dca = pd.read_csv(tmp_path / "decision_curve.csv")
    assert set(dca.columns) >= {"model", "threshold", "net_benefit"}
