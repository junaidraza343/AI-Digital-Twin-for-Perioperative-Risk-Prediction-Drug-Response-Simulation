"""Lazy, cached access to VitalDB numeric tracks and the clinical table."""
import hashlib
import re
import numpy as np
import pandas as pd
import twin.config as c

CASES_URL = "https://api.vitaldb.net/cases"


def _tracks_tag():
    h = hashlib.md5(",".join(c.NUMERIC_TRACKS).encode()).hexdigest()[:8]
    return h


def _default_loader(caseid, tracks, interval):
    import vitaldb
    return vitaldb.load_case(caseid, tracks, interval)


def cached_caseids() -> set:
    """Case ids already on disk FOR THE CURRENT TRACK SET.

    Files carry the track-set hash, so one written under a different NUMERIC_TRACKS
    can never be served and must not be reported as cached.
    """
    tag = _tracks_tag()
    out = set()
    if not c.DATA_CACHE.exists():
        return out
    for path in c.DATA_CACHE.glob(f"case_*_{tag}.parquet"):
        m = re.fullmatch(rf"case_(\d+)_{tag}\.parquet", path.name)
        if m:
            out.add(int(m.group(1)))
    return out


def load_numeric_frame(caseid, loader_fn=_default_loader) -> pd.DataFrame:
    """Return a 1 Hz DataFrame of NUMERIC_TRACKS for one case, cached on disk."""
    c.DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = c.DATA_CACHE / f"case_{caseid}_{_tracks_tag()}.parquet"
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
