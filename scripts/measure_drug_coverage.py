"""Measure SP4 cohort coverage: which eligible cases have propofol infusion +
continuous ART. Writes results/sp4_cohort.csv (qualifying caseids) and prints a
coverage summary. Re-pull is a manual VitalDB run (see plan Step 10)."""
import sys
import pandas as pd
import twin.config as c
from twin.data.vitaldb_loader import load_numeric_frame, _default_loader
from twin.data.drug_schedule import propofol_mg_min

TRKS_INDEX_URL = "https://api.vitaldb.net/trks"


def track_index(url: str = TRKS_INDEX_URL) -> pd.DataFrame:
    """VitalDB's track index: one row per (caseid, tname)."""
    return pd.read_csv(url)


def candidates_with_tracks(eligible: pd.DataFrame, index: pd.DataFrame) -> list:
    """Eligible cases that the index says carry BOTH propofol and arterial MAP.

    Downloading a case only to discover it never had an arterial line is the
    dominant cost of the cohort build, and the index rules those out for free.
    """
    have = lambda name: set(index.loc[index["tname"] == name, "caseid"].astype(int))
    keep = have(c.PROPOFOL_TRACK) & have(c.MAP_TRACK)
    return sorted(set(eligible["caseid"].astype(int)) & keep)


def art_coverage(frame: pd.DataFrame) -> float:
    """Fraction of the record covered by a continuous, physiologic arterial MAP.

    Artifacts are dropped before measuring, then gaps up to SP4_ART_FFILL_LIMIT_S
    are bridged. Solar8000 samples ART_MBP every 2s, so counting raw non-NaN
    samples on the 1 Hz grid measures the recorder's sample rate rather than
    whether the line was actually running.
    """
    art = pd.to_numeric(frame[c.MAP_TRACK], errors="coerce")
    art = art.where((art >= c.MAP_ARTIFACT_LO) & (art <= c.MAP_ARTIFACT_HI))
    return float(art.ffill(limit=c.SP4_ART_FFILL_LIMIT_S).notna().mean())


def case_qualifies(frame: pd.DataFrame):
    """Check whether one case's frame has enough propofol + ART coverage for SP4."""
    prop = propofol_mg_min(frame)
    prop_minutes = float((prop > 0).sum()) / 60.0
    if prop_minutes < c.SP4_MIN_PROPOFOL_MINUTES:
        return False, f"insufficient propofol ({prop_minutes:.1f} min)"
    if c.MAP_TRACK not in frame:
        return False, "no ART_MBP column"
    art_frac = art_coverage(frame)
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
    n_eligible = len(eligible)
    try:
        cands = candidates_with_tracks(eligible, track_index())
        eligible = pd.DataFrame({"caseid": cands})
        print(f"track index: {len(eligible)}/{n_eligible} eligible cases carry "
              f"both {c.PROPOFOL_TRACK} and {c.MAP_TRACK}")
    except Exception as exc:  # index unreachable -> fall back to the full sweep
        print(f"track index unavailable ({exc}); scanning all {n_eligible} cases")
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
