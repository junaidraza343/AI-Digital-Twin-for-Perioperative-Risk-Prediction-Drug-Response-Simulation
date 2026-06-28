import numpy as np
from twin.eval.decision_curve import net_benefit, net_benefit_curve


def test_net_benefit_treat_all_at_low_threshold():
    # At a very low threshold everyone is treated -> NB ~= prevalence
    y = np.array([0, 0, 1, 1])  # prevalence 0.5
    p = np.array([0.4, 0.4, 0.4, 0.4])
    nb = net_benefit(y, p, pt=0.001)
    assert abs(nb - 0.5) < 0.05


def test_net_benefit_zero_when_nothing_fires():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.1, 0.1, 0.1])
    assert net_benefit(y, p, pt=0.9) == 0.0


def test_curve_has_one_value_per_threshold():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    ts = np.array([0.1, 0.5, 0.9])
    curve = net_benefit_curve(y, p, ts)
    assert len(curve) == 3
