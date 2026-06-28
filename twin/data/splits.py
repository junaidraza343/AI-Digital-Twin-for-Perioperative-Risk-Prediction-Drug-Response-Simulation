"""Case-level stratified train/val/test assignment (no segment leakage)."""
import numpy as np
import pandas as pd
import twin.config as c


def make_splits(meta_df: pd.DataFrame, ratios=c.SPLIT_RATIOS, seed=c.SEED) -> dict:
    """Assign each caseid to 'train'/'val'/'test', stratified by
    ASA x age-decile x antihypertensive-medication, deterministically."""
    rng = np.random.default_rng(seed)
    meta = meta_df.copy()
    asa = meta["asa"].fillna(-1).astype(int)
    decile = (meta["age"] // 10).clip(upper=9).astype(int)
    htn = meta["preop_htn"].fillna(False).astype(bool)
    meta["stratum"] = list(zip(asa, decile, htn))

    assign = {}
    for _, grp in meta.groupby("stratum"):
        ids = grp["caseid"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        for i, cid in enumerate(ids):
            if i < n_tr:
                assign[cid] = "train"
            elif i < n_tr + n_va:
                assign[cid] = "val"
            else:
                assign[cid] = "test"
    return assign
