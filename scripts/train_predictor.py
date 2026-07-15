"""Train the MAP-only IOH predictor on real windows and persist it for the twin API.

The MAP-only model (logistic on map_last + map_slope) is chosen because its two
features can be recomputed directly from the mechanistic twin's projected MAP,
so the same trained-on-VitalDB predictor runs live inside the what-if UI.
Outputs:
  results/predictor_map_only.joblib   -- fitted sklearn pipeline
  results/predictor_meta.json         -- unbiased test AUROC/AUPRC/PPV + alarm threshold
"""
import json
import joblib
import numpy as np
import pandas as pd

import twin.config as c
from twin.data.splits import make_splits
from twin.models.baselines import MapOnlyModel, MAP_FEATURES
from twin.eval.selection_bias import make_unbiased, evaluate

MODEL_PATH = c.RESULTS_DIR / "predictor_map_only.joblib"
META_PATH = c.RESULTS_DIR / "predictor_meta.json"


def train():
    windows = pd.read_parquet(c.RESULTS_DIR / "windows.parquet")
    case_meta = pd.read_csv(c.RESULTS_DIR / "eligible_cases.csv")
    assign = make_splits(case_meta)
    windows = windows.copy()
    windows["split"] = windows["caseid"].map(assign)
    train_df = windows[windows.split == "train"]
    test_df = windows[windows.split == "test"]

    model = MapOnlyModel()
    model.fit(train_df, train_df["y"].to_numpy())

    # Operating threshold: the probability at the configured alarm rate on the
    # unbiased test set (never on train), for the UI's binary alarm.
    unbiased = make_unbiased(test_df)
    p = model.predict_proba(unbiased)
    y = unbiased["y"].to_numpy()
    thr = float(np.quantile(p, 1.0 - c.ALARM_RATE))
    metrics = evaluate(y, p) if len(np.unique(y)) > 1 else {}

    joblib.dump(model, MODEL_PATH)
    meta = {
        "model": "map_only_logistic",
        "features": MAP_FEATURES,
        "trained_on": "VitalDB windows.parquet",
        "n_train": int(len(train_df)),
        "n_test_unbiased": int(len(unbiased)),
        "alarm_rate": c.ALARM_RATE,
        "alarm_threshold": thr,
        "unbiased": {k: round(float(v), 4) for k, v in metrics.items()},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    m = train()
    print(json.dumps(m, indent=2))
