"""Norepinephrine case series for SP4.

VitalDB records a norepinephrine infusion in only 88 cases (25 of them inside this
project's eligible cohort), far too few to train a calibration head on. The pressor
arm is therefore reported as an explicit case series: for every SP4 case that did
receive a vasopressor, fit the teacher through the SAME differentiable twin and
report whether personalization beats the population model.

Writes results/sp4_norepi_series.csv and results/sp4_norepi_summary.json.
"""
import json

import numpy as np
import pandas as pd
import torch

import twin.config as c
from scripts.measure_drug_coverage import track_index
from scripts.sp4_pipeline import load_case
from twin.pkpd.teacher import fit_deltas_cohort, _pad_cohort
from twin.pkpd import torch_engine as te

DELTA_NAMES = ["V1", "V2", "V3", "ke0", "EC50", "gamma"]


def vasopressor_caseids(cohort_ids, index):
    """SP4 cases the track index says carry a norepinephrine or phenylephrine line."""
    have = lambda n: set(index.loc[index["tname"] == n, "caseid"].astype(int))
    return sorted(set(cohort_ids) & (have(c.NOREPI_TRACK) | have(c.PHENYLEPHRINE_TRACK)))


def recon_mse(cases, deltas):
    """Masked per-case MSE of projected vs observed MAP at the given deltas."""
    prop, norepi, obs = _pad_cohort(cases)
    pred = te.project_map_torch([x["patient"] for x in cases], prop, norepi, deltas)
    mask = ~torch.isnan(obs)
    filled = torch.nan_to_num(obs, nan=0.0)
    m = min(pred.shape[1], filled.shape[1])
    err = (pred[:, :m] - filled[:, :m]) * mask[:, :m]
    return ((err ** 2).sum(dim=1) / mask[:, :m].sum(dim=1).clamp(min=1.0)).numpy()


def main():
    ids = pd.read_csv(c.RESULTS_DIR / "sp4_cohort.csv")["caseid"].tolist()
    eligible = pd.read_csv(c.RESULTS_DIR / "eligible_cases.csv").set_index("caseid")
    candidates = vasopressor_caseids(ids, track_index())
    print(f"vasopressor cases inside the SP4 cohort: {len(candidates)}", flush=True)

    cases, kept = [], []
    for cid in candidates:
        if cid not in eligible.index:
            continue
        try:
            case = load_case(cid, eligible.loc[cid].to_dict())
        except Exception as exc:
            print(f"  skip {cid}: {exc}", flush=True)
            continue
        # the index only says a line existed; require actual delivered drug
        if case is None or float(case["norepi"].abs().sum()) == 0.0:
            continue
        cases.append(case)
        kept.append(cid)
    print(f"cases with a delivered vasopressor dose: {len(cases)}", flush=True)
    if not cases:
        raise SystemExit("no vasopressor cases available")

    zero = torch.zeros(len(cases), 6, dtype=torch.float64)
    pop = recon_mse(cases, zero)
    deltas, _ = fit_deltas_cohort(cases, n_iters=150, batch_size=len(cases))
    per = recon_mse(cases, deltas)

    df = pd.DataFrame({"caseid": kept,
                       "norepi_ug_min_max": [float(x["norepi"].max()) for x in cases],
                       "population_mse": pop, "personalized_mse": per,
                       "population_rmse": np.sqrt(pop),
                       "personalized_rmse": np.sqrt(per)})
    for i, name in enumerate(DELTA_NAMES):
        df[f"delta_{name}"] = deltas[:, i].numpy()
    df.to_csv(c.RESULTS_DIR / "sp4_norepi_series.csv", index=False)

    summary = {
        "n_cases": int(len(df)),
        "n_vasopressor_candidates": int(len(candidates)),
        "population_rmse_mean": float(np.sqrt(pop).mean()),
        "personalized_rmse_mean": float(np.sqrt(per).mean()),
        "improved_case_count": int((per < pop).sum()),
        "note": ("Case series, not a trained result: VitalDB provides too few "
                 "vasopressor cases to fit a calibration head. Deltas here come "
                 "from the per-case teacher fit through the differentiable twin."),
    }
    (c.RESULTS_DIR / "sp4_norepi_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
