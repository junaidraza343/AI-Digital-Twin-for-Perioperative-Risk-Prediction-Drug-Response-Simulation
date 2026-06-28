import numpy as np
from twin.eval.metrics import (auroc, auprc, ppv_at_alarm_rate,
                               expected_calibration_error, brier)


def test_auroc_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(y, p) == 1.0


def test_auprc_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auprc(y, p) == 1.0


def test_brier_zero_for_perfect():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    assert brier(y, p) == 0.0


def test_ppv_at_alarm_rate_top_decile():
    # 100 samples, top 10 by score are all positive -> PPV = 1.0
    y = np.zeros(100); y[:10] = 1
    p = np.linspace(1.0, 0.0, 100)  # first 10 highest scores are the positives
    assert ppv_at_alarm_rate(y, p, 0.10) == 1.0


def test_ece_zero_for_calibrated_extremes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p) == 0.0
