"""Build the study cohort from the VitalDB clinical table and save it."""
import pandas as pd
import twin.config as c
from twin.data.cohort import filter_cohort
from twin.data.vitaldb_loader import load_cases_table


def build_cohort(cases_df=None):
    if cases_df is None:
        cases_df = load_cases_table()
    eligible, funnel = filter_cohort(cases_df)
    c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(funnel, columns=["step", "n_remaining"]).to_csv(
        c.RESULTS_DIR / "cohort_funnel.csv", index=False)
    eligible.to_csv(c.RESULTS_DIR / "eligible_cases.csv", index=False)
    return eligible, funnel


if __name__ == "__main__":  # pragma: no cover
    eligible, funnel = build_cohort()
    print("Cohort funnel:")
    for step, n in funnel:
        print(f"  {step:20s} {n}")
    print(f"Eligible cases: {len(eligible)}")
