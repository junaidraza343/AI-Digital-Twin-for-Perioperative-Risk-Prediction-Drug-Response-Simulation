"""Evaluation metrics for IOH prediction."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import twin.config as c


def auroc(y, p):
    return float(roc_auc_score(y, p))


def auprc(y, p):
    return float(average_precision_score(y, p))


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def ppv_at_alarm_rate(y, p, alarm_rate=c.ALARM_RATE):
    """Precision when the top `alarm_rate` fraction of scores fire an alarm."""
    y = np.asarray(y); p = np.asarray(p)
    n = len(p)
    k = max(1, int(round(alarm_rate * n)))
    thresh = np.sort(p)[::-1][k - 1]
    pred = p >= thresh
    denom = int(np.sum(pred))
    if denom == 0:
        return 0.0
    return float(np.sum((pred) & (y == 1)) / denom)


def expected_calibration_error(y, p, n_bins=c.ECE_BINS):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(p)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
