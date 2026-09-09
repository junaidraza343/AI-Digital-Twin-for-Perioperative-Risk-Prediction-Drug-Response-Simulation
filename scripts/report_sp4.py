"""Consolidate the SP4 run into one thesis-ready results summary.

Reads whatever exists in results/ and writes results/sp4_report.md. Missing inputs
are reported as missing rather than skipped, so the report never implies a result
that was not actually produced.
"""
import json

import numpy as np
import pandas as pd

import twin.config as c

DELTA_NAMES = ["V1", "V2", "V3", "ke0", "EC50", "gamma"]


def _load_json(name):
    path = c.RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def _pct(new, old):
    """Relative change, guarding the degenerate baseline."""
    return float("nan") if not old else (new - old) / old * 100.0


def build_report():
    res = _load_json("sp4_results.json")
    nore = _load_json("sp4_norepi_summary.json")
    lines = ["# SP4 — Coupled Twin: Results", ""]

    if res is None:
        lines += ["**results/sp4_results.json missing — the SP4 run has not "
                  "completed.**", ""]
    else:
        h = res.get("held_out", {})
        lines += [
            "## Cohort and split", "",
            f"- SP4 cohort: **{res.get('n_cases', '?')}** cases "
            f"(propofol infusion + continuous arterial MAP)",
            f"- Train **{res.get('n_train','?')}** / held-out test "
            f"**{res.get('n_test','?')}**, case-level stratified split",
            f"- Teacher iterations per case: {res.get('teacher_iters','?')}", "",
            "## Held-out IOH prediction", "",
            "| metric | value |", "|---|---|",
            f"| AUROC | {h.get('auroc', float('nan')):.3f} |",
            f"| AUPRC | {h.get('auprc', float('nan')):.3f} |",
            f"| PPV @ alarm rate | {h.get('ppv', float('nan')):.3f} |",
            f"| ECE | {h.get('ece', float('nan')):.3f} |",
            f"| Brier | {h.get('brier', float('nan')):.3f} |",
            f"| prevalence | {h.get('prevalence', float('nan')):.3f} |",
            f"| windows | {h.get('n_windows','?')} |", "",
            "## Held-out personalization (the SP4 claim)", "",
            "MAP reconstruction using the calibration head's delta, on cases the "
            "model never saw, against the population twin (delta = 0).", "",
            "| model | RMSE (mmHg) |", "|---|---|",
            f"| population (delta=0) | {h.get('population_rmse', float('nan')):.2f} |",
            f"| personalized (head delta) | {h.get('personalized_rmse', float('nan')):.2f} |",
            "",
            f"- Cases improved: **{h.get('improved_case_count','?')}** of "
            f"{h.get('n_cases','?')}",
        ]
        if "population_rmse" in h and "personalized_rmse" in h:
            change = _pct(h["personalized_rmse"], h["population_rmse"])
            verdict = ("personalization IMPROVES held-out reconstruction"
                       if change < 0 else
                       "personalization does NOT improve held-out reconstruction")
            lines += [f"- Change: **{change:+.1f}%** — {verdict}", ""]

        std = res.get("delta_std")
        if std:
            lines += ["## Delta identifiability", "",
                      "Spread of each personalization delta across the cohort; a "
                      "near-zero spread means the data does not move that "
                      "parameter.", "",
                      "| delta | std |", "|---|---|"]
            lines += [f"| {n} | {v:.4f} |" for n, v in zip(DELTA_NAMES, std)]
            lines += [""]

    lines += ["## Norepinephrine arm", ""]
    if nore is None:
        lines += ["**Not run** (results/sp4_norepi_summary.json missing).", ""]
    else:
        lines += [
            f"- Vasopressor cases in the SP4 cohort: "
            f"**{nore.get('n_cases','?')}** of {nore.get('n_vasopressor_candidates','?')} "
            "candidates",
            f"- Population RMSE {nore.get('population_rmse_mean', float('nan')):.2f} "
            f"-> personalized {nore.get('personalized_rmse_mean', float('nan')):.2f} mmHg",
            f"- Improved: {nore.get('improved_case_count','?')} cases", "",
            f"> {nore.get('note','')}", "",
        ]

    deltas_path = c.RESULTS_DIR / "sp4_deltas.csv"
    if deltas_path.exists():
        d = pd.read_csv(deltas_path)
        avail = [n for n in DELTA_NAMES if n in d.columns]
        if avail:
            sat = (d[avail].abs() > 0.65).mean().mean() * 100
            lines += ["## Caveat — delta saturation", "",
                      f"{sat:.1f}% of head-predicted delta components sit beyond "
                      "|0.65| against a +/-0.7 bound. With six free deltas the fit "
                      "is degenerate, so report reconstruction improvement, not "
                      "recovery of physiological parameters.", ""]

    out = c.RESULTS_DIR / "sp4_report.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(build_report())
