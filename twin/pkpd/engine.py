"""Top-level PK-PD projection: patient + dosing + deltas -> MAP trajectory."""
from dataclasses import dataclass
import numpy as np

import twin.config as c
from twin.pkpd.params import POP_PROPOFOL, POP_PD, POP_NOREPI
from twin.pkpd.covariates import scale_propofol
from twin.pkpd.calibration import apply_deltas
from twin.pkpd.schedule import rate_vector
from twin.pkpd.propofol import simulate_effect_site
from twin.pkpd.norepi import simulate_map_rise
from twin.pkpd.pd_model import emax_reduction, combine_map


@dataclass(frozen=True)
class Patient:
    age: float
    weight: float
    height: float
    sex: str
    map0: float = 90.0


@dataclass(frozen=True)
class MapProjection:
    t: np.ndarray
    map: np.ndarray
    ce: np.ndarray
    minutes_below_65: float


def project_map(patient: Patient, prop_schedule, norepi_schedule,
                duration: float = 900.0, dt: float = 1.0, deltas=None) -> MapProjection:
    prop_params = scale_propofol(POP_PROPOFOL, patient.age, patient.weight,
                                 patient.height, patient.sex)
    pd_params = POP_PD
    if deltas:
        prop_params, pd_params = apply_deltas(prop_params, pd_params, deltas)

    prop_rate = rate_vector(prop_schedule, dt, duration)
    norepi_rate = rate_vector(norepi_schedule, dt, duration)

    ce = simulate_effect_site(prop_params, prop_rate, dt)
    reduction = emax_reduction(ce, pd_params)
    rise = simulate_map_rise(POP_NOREPI, norepi_rate, dt)
    map_traj = combine_map(patient.map0, reduction, rise)

    t = np.arange(len(map_traj)) * dt
    below = float(np.sum(map_traj < c.MAP_THRESHOLD) * dt / 60.0)
    return MapProjection(t=t, map=map_traj, ce=ce, minutes_below_65=below)


def ioh_risk(proj: MapProjection) -> str:
    """Categorical IOH risk from the projected MAP nadir.

    High: MAP breaches the 65 mmHg threshold. Medium: dips into the 65-75 gray
    zone. Low: stays >= 75. Mirrors the stable/gray/overt window categories.
    """
    m = float(proj.map.min())
    if m < c.MAP_THRESHOLD:
        return "High"
    if m < c.GRAY_HIGH:
        return "Medium"
    return "Low"


def project_band(patient: Patient, prop_schedule, norepi_schedule,
                 duration: float = 900.0, dt: float = 1.0, deltas=None,
                 spread: float = 0.2):
    """Confidence band from PK-PD parameter uncertainty.

    Perturbs the two most influential parameters (EC50 sensitivity, ke0 onset)
    by +/- spread around the chosen deltas and returns (lo, hi, central) MAP
    envelopes. Deterministic.
    """
    base = dict(deltas or {})
    central = project_map(patient, prop_schedule, norepi_schedule,
                          duration, dt, base or None).map
    runs = [central]
    for ec in (-spread, spread):
        for ke in (-spread, spread):
            d = dict(base)
            d["EC50"] = d.get("EC50", 0.0) + ec
            d["ke0"] = d.get("ke0", 0.0) + ke
            runs.append(project_map(patient, prop_schedule, norepi_schedule,
                                    duration, dt, d).map)
    stack = np.vstack(runs)
    return stack.min(axis=0), stack.max(axis=0), central
