"""Train the PyTorch tabular deep net and evaluate it in the selection-bias study.

Runs on GPU (CUDA on Colab, MPS on Apple Silicon) or CPU. Writes:
  results/deepnet.pt            -- trained weights + preprocessing
  results/deepnet_results.csv   -- model=deepnet x {biased, unbiased, gray_challenge}
which the admin console merges into the model x regime table.
"""
import numpy as np
import pandas as pd

import twin.config as c
from twin.data.splits import make_splits
from twin.models.deepnet import TorchTabularModel, pick_device
from twin.eval.selection_bias import make_biased, make_unbiased, evaluate

DROP = {"caseid", "t_end", "y", "category", "split"}
RESULTS = c.RESULTS_DIR / "deepnet_results.csv"
WEIGHTS = c.RESULTS_DIR / "deepnet.pt"


def _xy(df):
    return df[[col for col in df.columns if col not in DROP]], df["y"].to_numpy()


def train(epochs: int = 40):
    windows = pd.read_parquet(c.RESULTS_DIR / "windows.parquet")
    case_meta = pd.read_csv(c.RESULTS_DIR / "eligible_cases.csv")
    assign = make_splits(case_meta)
    windows = windows.copy()
    windows["split"] = windows["caseid"].map(assign)
    train_df = windows[windows.split == "train"]
    test = windows[windows.split == "test"]

    print(f"device: {pick_device()} | train rows: {len(train_df)}")
    Xtr, ytr = _xy(train_df)
    model = TorchTabularModel(epochs=epochs).fit(Xtr, ytr)

    regimes = [
        ("biased", make_biased(test)),
        ("unbiased", make_unbiased(test)),
        ("gray_challenge", test[test.category.isin(["overt", "gray"])].reset_index(drop=True)),
    ]
    rows = []
    for regime, subset in regimes:
        Xte, yte = _xy(subset)
        if len(yte) == 0 or len(np.unique(yte)) < 2:
            row = {"model": "deepnet", "regime": regime, "n": int(len(yte))}
        else:
            row = {"model": "deepnet", "regime": regime}
            row.update(evaluate(yte, model.predict_proba(Xte)))
        rows.append(row)

    out = pd.DataFrame(rows)
    c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS, index=False)
    model.save(WEIGHTS)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    print(train(ap.parse_args().epochs).to_string(index=False))
