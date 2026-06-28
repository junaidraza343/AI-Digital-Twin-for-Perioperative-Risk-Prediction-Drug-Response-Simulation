"""Biased vs unbiased evaluation harness for the selection-bias study."""
import pandas as pd
from twin.eval import metrics


def make_biased(windows_df: pd.DataFrame) -> pd.DataFrame:
    """HPI-style: keep only clearly-stable negatives and overt positives."""
    return windows_df[windows_df["category"] != "gray"].reset_index(drop=True)


def make_unbiased(windows_df: pd.DataFrame) -> pd.DataFrame:
    """Realistic: keep all windows, gray zone included."""
    return windows_df.reset_index(drop=True)


def evaluate(y, p) -> dict:
    """Compute the standard metric bundle for one (model, regime)."""
    return {
        "auroc": metrics.auroc(y, p),
        "auprc": metrics.auprc(y, p),
        "ppv": metrics.ppv_at_alarm_rate(y, p),
        "ece": metrics.expected_calibration_error(y, p),
        "brier": metrics.brier(y, p),
        "n": int(len(y)),
        "prevalence": float(sum(y) / len(y)),
    }
