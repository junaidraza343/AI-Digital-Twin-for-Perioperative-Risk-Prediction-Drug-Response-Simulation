"""Drive the real SP4 run end to end: cohort -> Stage 1 -> Stage 2 -> evaluation.

Writes results/coupled_stage1.pt, results/coupled_stage2.pt, results/sp4_deltas.csv
and results/sp4_results.json. Cases are length-sorted so each teacher chunk pads to
a similar T -- padding is wasted simulation, and record lengths vary ~5x.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import twin.config as c
from scripts.sp4_pipeline import (load_case, run_stage1_cohort, run_stage2_cohort,
                                  feature_columns)
from twin.data.vitaldb_loader import _tracks_tag
from twin.eval.personalization import delta_identifiability


def length_sorted_cohort(caseids, eligible):
    """Order cases by cached record length so batches pad to a similar T."""
    rows = []
    for cid in caseids:
        if cid not in eligible.index:
            continue
        path = c.DATA_CACHE / f"case_{cid}_{_tracks_tag()}.parquet"
        try:
            n = pd.read_parquet(path).shape[0] if path.exists() else 10 ** 9
        except Exception:
            n = 10 ** 9
        rows.append((n, cid))
    rows.sort()
    return [(cid, eligible.loc[cid].to_dict()) for _, cid in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=0, help="0 = whole cohort")
    ap.add_argument("--teacher-iters", type=int, default=100)
    ap.add_argument("--teacher-batch", type=int, default=128)
    ap.add_argument("--stage1-epochs", type=int, default=300)
    ap.add_argument("--stage2-epochs", type=int, default=20)
    ap.add_argument("--case-batch", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ids = pd.read_csv(c.RESULTS_DIR / "sp4_cohort.csv")["caseid"].tolist()
    eligible = pd.read_csv(c.RESULTS_DIR / "eligible_cases.csv").set_index("caseid")
    cohort = length_sorted_cohort(ids, eligible)
    if args.n_cases:
        cohort = cohort[:args.n_cases]
    print(f"SP4 cohort: {len(cohort)} cases | device={args.device}", flush=True)

    t0 = time.time()
    model, prep, cols, info1 = run_stage1_cohort(
        cohort, teacher_iters=args.teacher_iters, epochs=args.stage1_epochs,
        latent_dim=32, device=args.device, teacher_batch_size=args.teacher_batch,
        progress=True, save_path=c.RESULTS_DIR / "coupled_stage1.pt")
    print(f"stage1 {info1} in {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    model, info2 = run_stage2_cohort(
        cohort, model, prep, cols, epochs=args.stage2_epochs,
        device=args.device, case_batch_size=args.case_batch, progress=True,
        save_path=c.RESULTS_DIR / "coupled_stage2.pt")
    print(f"stage2 {info2} in {time.time()-t1:.0f}s", flush=True)

    # Head-predicted deltas per case + identifiability across the cohort
    model.eval()
    rows = []
    with torch.no_grad():
        for caseid, static_row in cohort:
            try:
                case = load_case(caseid, static_row, device=args.device)
            except Exception:
                continue
            if case is None:
                continue
            w = case["windows"]
            x = torch.tensor(prep.transform(w[cols]), dtype=torch.float32,
                             device=args.device)
            _, delta, _ = model(x)
            rows.append([caseid] + delta.mean(dim=0).cpu().numpy().tolist())
    deltas = pd.DataFrame(rows, columns=["caseid", "V1", "V2", "V3", "ke0",
                                         "EC50", "gamma"])
    deltas.to_csv(c.RESULTS_DIR / "sp4_deltas.csv", index=False)

    ident = delta_identifiability(
        torch.tensor(deltas[["V1", "V2", "V3", "ke0", "EC50", "gamma"]].to_numpy()))
    out = {"n_cases": int(len(deltas)), "stage1": info1, "stage2": info2,
           "delta_std": [float(v) for v in np.asarray(ident)],
           "teacher_iters": args.teacher_iters}
    (c.RESULTS_DIR / "sp4_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
