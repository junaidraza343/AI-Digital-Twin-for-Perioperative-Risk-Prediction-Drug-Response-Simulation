"""Train baselines and run the biased-vs-unbiased selection-bias study."""
import numpy as np
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

    # val split is reserved for SP2 threshold tuning; SP1 uses a fixed ALARM_RATE.
    Xtr, ytr = _xy(train)
    results = []
    trained = {}

    regimes = [
        ("biased", make_biased(test)),
        ("unbiased", make_unbiased(test)),
        ("gray_challenge", test[test.category.isin(["overt", "gray"])].reset_index(drop=True)),
    ]

    for name, model in [("map_only", MapOnlyModel()), ("gbdt", GbdtModel())]:
        model.fit(Xtr, ytr)
        trained[name] = model
        for regime, subset in regimes:
            Xte, yte = _xy(subset)
            if len(yte) == 0 or len(np.unique(yte)) < 2:
                row = {"model": name, "regime": regime, "auroc": float("nan"),
                       "auprc": float("nan"), "ppv": float("nan"), "ece": float("nan"),
                       "brier": float("nan"),
                       "n": int(len(yte)),
                       "prevalence": float(yte.mean()) if len(yte) else float("nan")}
            else:
                p = model.predict_proba(Xte)
                row = {"model": name, "regime": regime}
                row.update(evaluate(yte, p))
            results.append(row)

    out = pd.DataFrame(results)
    c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(c.RESULTS_DIR / "selection_bias_results.csv", index=False)

    from twin.eval.decision_curve import net_benefit_curve
    thresholds = np.linspace(0.01, 0.5, 25)
    dca_rows = []
    unbiased = make_unbiased(test)
    Xu, yu = _xy(unbiased)
    if len(yu) and len(np.unique(yu)) > 0:
        for name, model in [("map_only", trained["map_only"]), ("gbdt", trained["gbdt"])]:
            pu = model.predict_proba(Xu)
            nb = net_benefit_curve(yu, pu, thresholds)
            for t, v in zip(thresholds, nb):
                dca_rows.append({"model": name, "threshold": float(t), "net_benefit": float(v)})
    pd.DataFrame(dca_rows).to_csv(c.RESULTS_DIR / "decision_curve.csv", index=False)

    return out


if __name__ == "__main__":  # pragma: no cover
    windows = pd.read_parquet(c.RESULTS_DIR / "windows.parquet")
    case_meta = pd.read_csv(c.RESULTS_DIR / "eligible_cases.csv")
    print(run(windows, case_meta).to_string(index=False))
