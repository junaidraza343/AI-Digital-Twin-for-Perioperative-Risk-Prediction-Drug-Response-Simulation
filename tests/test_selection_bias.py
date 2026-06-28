import numpy as np
import pandas as pd
from twin.eval.selection_bias import make_biased, make_unbiased, evaluate


def _windows():
    return pd.DataFrame({
        "caseid": [1, 1, 2, 2, 3],
        "t_end": [600, 630, 600, 630, 600],
        "y": [0, 1, 0, 0, 1],
        "category": ["stable", "overt", "gray", "stable", "overt"],
    })


def test_biased_drops_gray_zone():
    out = make_biased(_windows())
    assert "gray" not in set(out.category)
    assert len(out) == 4


def test_unbiased_keeps_everything():
    out = make_unbiased(_windows())
    assert len(out) == 5


def test_evaluate_returns_expected_metric_keys():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(y, p)
    for k in ["auroc", "auprc", "ppv", "ece", "brier"]:
        assert k in m
