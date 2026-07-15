"""Sigmoid Emax pharmacodynamics and MAP combination.

Propofol lowers MAP by a fractional reduction (0..emax). Norepinephrine raises
MAP by a fractional rise. Combined: MAP(t) = MAP0*(1 - reduction) + MAP0*rise,
clipped to a physiological range.
"""
import numpy as np
from twin.pkpd.params import PDParams

MAP_MIN, MAP_MAX = 20.0, 200.0


def emax_reduction(ce: np.ndarray, pd: PDParams) -> np.ndarray:
    ce = np.asarray(ce, dtype=float)
    ce_g = np.power(np.clip(ce, 0, None), pd.gamma)
    return pd.emax * ce_g / (pd.ec50 ** pd.gamma + ce_g)


def combine_map(map0: float, prop_reduction: np.ndarray, norepi_rise: np.ndarray) -> np.ndarray:
    traj = map0 * (1.0 - prop_reduction) + map0 * norepi_rise
    return np.clip(traj, MAP_MIN, MAP_MAX)
