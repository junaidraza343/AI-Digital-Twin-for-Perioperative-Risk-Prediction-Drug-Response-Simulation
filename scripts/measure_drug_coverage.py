"""Measure SP4 cohort coverage: which eligible cases have propofol infusion +
continuous ART. Writes results/sp4_cohort.csv (qualifying caseids) and prints a
coverage summary. Re-pull is a manual VitalDB run (see plan Step 10)."""
import sys
import pandas as pd
import twin.config as c
from twin.data.vitaldb_loader import load_numeric_frame, _default_loader
from twin.data.drug_schedule import propofol_mg_min


def case_qualifies(frame: pd.DataFrame):
    """Check whether one case's frame has enough propofol + ART coverage for SP4."""
    prop = propofol_mg_min(frame)
    prop_minutes = float((prop > 0).sum()) / 60.0
    if prop_minutes < c.SP4_MIN_PROPOFOL_MINUTES:
        return False, f"insufficient propofol ({prop_minutes:.1f} min)"
    if c.MAP_TRACK not in frame:
        return False, "no ART_MBP column"
    art_frac = float(frame[c.MAP_TRACK].notna().mean())
    if art_frac < c.SP4_MIN_ART_FRACTION:
        return False, f"ART coverage {art_frac:.2f} < {c.SP4_MIN_ART_FRACTION}"
    return True, "ok"


def measure_coverage(eligible_cases: pd.DataFrame, loader_fn=_default_loader) -> dict:
    """Load each eligible case and tally how many qualify for the SP4 cohort."""
    caseids, reasons = [], []
    for _, row in eligible_cases.iterrows():
        cid = int(row["caseid"])
        frame = load_numeric_frame(cid, loader_fn=loader_fn)
        ok, reason = case_qualifies(frame)
        if ok:
            caseids.append(cid)
        reasons.append(reason)
    return {"n_total": len(eligible_cases), "n_qualifying": len(caseids),
            "caseids": caseids, "reasons": reasons}


def main():
    eligible = pd.read_csv(c.RESULTS_DIR / "eligible_cases.csv")
    summary = measure_coverage(eligible)
    out = c.RESULTS_DIR / "sp4_cohort.csv"
    pd.DataFrame({"caseid": summary["caseids"]}).to_csv(out, index=False)
    reasons_out = c.RESULTS_DIR / "sp4_cohort_reasons.csv"
    pd.DataFrame({"caseid": eligible["caseid"], "reason": summary["reasons"]}).to_csv(
        reasons_out, index=False)
    print(f"SP4 cohort: {summary['n_qualifying']}/{summary['n_total']} cases "
          f"qualify -> {out}")


if __name__ == "__main__":
    sys.exit(main())
