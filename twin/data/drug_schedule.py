"""Convert cached Orchestra drug-rate tracks (mL/h) into engine input vectors.

Propofol -> mg/min (drives the PK-PD twin's central compartment).
Vasopressors (phenylephrine + norepinephrine) -> combined ug/min norepi-equivalent.
NaN (pump off / not charted) -> 0. Documented concentration approximations in config.
"""
import numpy as np
import pandas as pd
import twin.config as c


def _ml_per_hour_to_per_min(frame: pd.DataFrame, track: str) -> np.ndarray:
    if track not in frame:
        return np.zeros(len(frame))
    v = frame[track].to_numpy(dtype=float)
    v = np.nan_to_num(v, nan=0.0)
    return v / 60.0  # mL/h -> mL/min


def propofol_mg_min(frame: pd.DataFrame) -> np.ndarray:
    return _ml_per_hour_to_per_min(frame, c.PROPOFOL_TRACK) * c.PPF20_MG_PER_ML


def vasopressor_ug_min(frame: pd.DataFrame) -> np.ndarray:
    phen = _ml_per_hour_to_per_min(frame, c.PHENYLEPHRINE_TRACK) * c.PHEN_UG_PER_ML
    nepi = _ml_per_hour_to_per_min(frame, c.NOREPI_TRACK) * c.NEPI_UG_PER_ML
    return phen + nepi
