"""Data-driven calibration: fit the personalization deltas to an observed MAP.

This is a CPU stand-in for the learned SP4 calibration head. Instead of a neural
network emitting the deltas, we fit them by minimizing the error between the
twin's projected MAP and the patient's observed MAP, via a coarse-to-fine grid
search over the two most influential perturbations (EC50 sensitivity, ke0 onset).
Deterministic.
"""
import numpy as np

from twin.pkpd.engine import project_map


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def fit_deltas(patient, prop_schedule, norepi_schedule, observed_map,
               duration=900.0, dt=1.0, bound=0.7, keys=("EC50", "ke0")):
    """Return (deltas, rmse) fitting `keys` so the twin matches observed_map.

    Coarse grid over [-bound, bound] per key, then one refinement pass around the
    best point. Only the overlapping length of observed_map is scored.
    """
    observed = np.asarray(observed_map, dtype=float)

    def score(d):
        proj = project_map(patient, prop_schedule, norepi_schedule,
                           duration=duration, dt=dt, deltas=d)
        m = min(len(proj.map), len(observed))
        return _rmse(proj.map[:m], observed[:m])

    def search(centers, span, n):
        grids = {k: np.linspace(centers[k] - span, centers[k] + span, n) for k in keys}
        grids = {k: np.clip(v, -bound, bound) for k, v in grids.items()}
        best, best_e = dict(centers), float("inf")
        # two keys only -> nested loop (kept simple and explicit)
        for a in grids[keys[0]]:
            for b in grids[keys[1]]:
                d = {keys[0]: float(a), keys[1]: float(b)}
                e = score(d)
                if e < best_e:
                    best_e, best = e, d
        return best, best_e

    centers = {k: 0.0 for k in keys}
    best, _ = search(centers, bound, 9)          # coarse
    best, best_e = search(best, bound / 4, 9)     # refine
    return best, best_e
