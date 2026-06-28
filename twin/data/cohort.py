"""Filter the VitalDB clinical-information table to the study cohort."""
import pandas as pd
import twin.config as c


def _preop_map(df: pd.DataFrame) -> pd.Series:
    return df["preop_dbp"] + (df["preop_sbp"] - df["preop_dbp"]) / 3.0


def filter_cohort(cases_df: pd.DataFrame):
    """Return (eligible_df, funnel) where funnel is a list of (step, n_remaining)."""
    df = cases_df.copy()
    funnel = [("start", len(df))]

    df = df[df["age"] >= c.MIN_AGE]
    funnel.append(("age>=18", len(df)))

    df = df[~df["department"].isin(c.EXCLUDED_DEPARTMENTS)]
    funnel.append(("department", len(df)))

    # 'contains General' intentionally includes combined-technique cases (e.g. General/Regional).
    df = df[df["ane_type"].astype(str).str.contains("General", case=False, na=False)]
    funnel.append(("general_anesthesia", len(df)))

    df = df[(df["caseend"] - df["casestart"]) >= c.MIN_DURATION_SECONDS]
    funnel.append(("duration>=30min", len(df)))

    df = df[~df["asa"].isin(c.EXCLUDED_ASA)]
    funnel.append(("asa_not_5_6", len(df)))

    df = df[_preop_map(df) >= c.MAP_THRESHOLD]
    funnel.append(("preop_map>=65", len(df)))

    return df.reset_index(drop=True), funnel
