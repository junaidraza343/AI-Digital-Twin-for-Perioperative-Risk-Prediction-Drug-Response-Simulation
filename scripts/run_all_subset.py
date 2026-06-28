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
