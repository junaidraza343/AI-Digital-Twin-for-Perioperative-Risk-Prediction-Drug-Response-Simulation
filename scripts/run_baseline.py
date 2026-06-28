"""Train baselines and run the biased-vs-unbiased selection-bias study."""
import pandas as pd
import twin.config as c
from twin.data.splits import make_splits
from twin.models.baselines import MapOnlyModel, GbdtModel
from twin.eval.selection_bias import make_biased, make_unbiased, evaluate

FEATURE_DROP = {"caseid", "t_end", "y", "category", "split"}


def _xy(df):
    X = df[[col for col in df.columns if col not in FEATURE_DROP]]
    return X, df["y"].to_numpy()


def run(windows: pd.DataFrame, case_meta: pd.DataFrame) -> pd.DataFrame:
    assign = make_splits(case_meta)
    windows = windows.copy()
    windows["split"] = windows["caseid"].map(assign)
    train = windows[windows.split == "train"]
    test = windows[windows.split == "test"]

    Xtr, ytr = _xy(train)
    results = []
    for name, model in [("map_only", MapOnlyModel()), ("gbdt", GbdtModel())]:
        model.fit(Xtr, ytr)
        for regime, subset in [("biased", make_biased(test)), ("unbiased", make_unbiased(test))]:
            Xte, yte = _xy(subset)
            p = model.predict_proba(Xte)
            row = {"model": name, "regime": regime}
            row.update(evaluate(yte, p))
            results.append(row)
    out = pd.DataFrame(results)
    c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(c.RESULTS_DIR / "selection_bias_results.csv", index=False)
    return out


if __name__ == "__main__":  # pragma: no cover
    windows = pd.read_parquet(c.RESULTS_DIR / "windows.parquet")
    case_meta = pd.read_csv(c.RESULTS_DIR / "eligible_cases.csv")
    print(run(windows, case_meta).to_string(index=False))
