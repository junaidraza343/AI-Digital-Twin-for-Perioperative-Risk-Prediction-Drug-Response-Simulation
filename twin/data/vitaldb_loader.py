"""Lazy, cached access to VitalDB numeric tracks and the clinical table."""
import numpy as np
import pandas as pd
import twin.config as c

CASES_URL = "https://api.vitaldb.net/cases"


def _default_loader(caseid, tracks, interval):
    import vitaldb
    return vitaldb.load_case(caseid, tracks, interval)


def load_numeric_frame(caseid, loader_fn=_default_loader) -> pd.DataFrame:
    """Return a 1 Hz DataFrame of NUMERIC_TRACKS for one case, cached on disk."""
    c.DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = c.DATA_CACHE / f"case_{caseid}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    arr = loader_fn(caseid, c.NUMERIC_TRACKS, 1)
    arr = np.asarray(arr, dtype=float)
    frame = pd.DataFrame(arr, columns=c.NUMERIC_TRACKS)
    frame.to_parquet(cache_path)
    return frame


def load_cases_table() -> pd.DataFrame:
    """Download the VitalDB clinical-information table (cases.csv)."""
    return pd.read_csv(CASES_URL)
