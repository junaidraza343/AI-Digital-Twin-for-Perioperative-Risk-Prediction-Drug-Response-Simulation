"""Turn cached VitalDB cases into a labeled feature table."""
import os
import sys
import pandas as pd
from twin.data.vitaldb_loader import load_numeric_frame, _default_loader
from twin.data.labeling import make_map_series, label_windows
from twin.data.features import extract_features

# When TWIN_PROGRESS=1, emit "PROGRESS k/N" lines so a UI can show a percentage.
_PROGRESS = os.environ.get("TWIN_PROGRESS") == "1"

# Leakage-safe static covariates. Outcome columns (death_inhosp, intraop_ebl,
# intraop_uo) and free-text/high-cardinality columns are deliberately excluded.
STATIC_COVARIATES = [
    "age", "sex", "weight", "height", "bmi", "asa",
    "preop_htn", "preop_dm",
    "preop_hb", "preop_cr", "preop_na", "preop_k",
]


def _safe_static(row_dict: dict) -> dict:
    """Select only allowlisted covariates; encode sex as binary (M=1, F=0)."""
    out = {}
    for key in STATIC_COVARIATES:
        if key in row_dict:
            value = row_dict[key]
            if key == "sex":
                value = 1 if str(value).upper().startswith("M") else 0
            out[key] = value
    return out


def build_windows_for_case(caseid, static_row, loader_fn=_default_loader) -> pd.DataFrame:
    """Load one case, label its windows, and attach leakage-safe features."""
    frame = load_numeric_frame(caseid, loader_fn=loader_fn)
    map_series = make_map_series(frame)
    windows = label_windows(map_series)
    if windows.empty:
        return windows
    safe_static = _safe_static(dict(static_row))
    rows = []
    for _, w in windows.iterrows():
        feats = extract_features(frame, int(w.t_end), dict(safe_static))
        feats["caseid"] = caseid
        feats["t_end"] = int(w.t_end)
        feats["y"] = int(w.y)
        feats["category"] = w.category
        rows.append(feats)
    return pd.DataFrame(rows)


def build_dataset(eligible_cases: pd.DataFrame, loader_fn=_default_loader) -> pd.DataFrame:
    """Build the full window table across all eligible cases."""
    parts = []
    total = len(eligible_cases)
    for k, (_, row) in enumerate(eligible_cases.iterrows(), start=1):
        try:
            part = build_windows_for_case(int(row["caseid"]), row.to_dict(), loader_fn=loader_fn)
            if not part.empty:
                parts.append(part)
        except Exception as exc:  # skip unreadable cases, keep going
            print(f"  skip case {row['caseid']}: {exc}")
        if _PROGRESS:
            print(f"PROGRESS {k}/{total}", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
