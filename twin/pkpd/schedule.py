"""Convert a dosing schedule to a per-timestep rate vector.

A schedule is a list of (t_start_s, t_end_s, rate) segments. A bolus is a
1-second segment. Overlapping segments sum.
"""
import numpy as np


def rate_vector(schedule, dt: float, duration: float) -> np.ndarray:
    n = int(round(duration / dt)) + 1
    v = np.zeros(n)
    for t_start, t_end, rate in schedule:
        i0 = int(round(t_start / dt))
        i1 = int(round(t_end / dt))
        i0 = max(0, min(i0, n))
        i1 = max(0, min(i1, n))
        v[i0:i1] += rate
    return v
