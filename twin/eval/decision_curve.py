"""Decision-curve analysis: clinical net benefit across thresholds."""
import numpy as np


def net_benefit(y, p, pt):
    """Net benefit at threshold probability pt (Vickers & Elkin 2006)."""
    if pt >= 1.0:
        return float("-inf")
    y = np.asarray(y); p = np.asarray(p)
    n = len(y)
    pred = p >= pt
    tp = int(np.sum((pred) & (y == 1)))
    fp = int(np.sum((pred) & (y == 0)))
    return float(tp / n - (fp / n) * (pt / (1 - pt)))


def net_benefit_curve(y, p, thresholds):
    return np.array([net_benefit(y, p, t) for t in thresholds])
