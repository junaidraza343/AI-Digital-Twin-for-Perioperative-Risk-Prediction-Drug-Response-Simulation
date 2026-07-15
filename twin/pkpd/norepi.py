"""Norepinephrine -> fractional MAP rise.

SIMPLIFIED from Joachim 2024 to its summary dynamics: a two-stage cascade
(plasma -> effect) so a bolus produces a delayed rise peaking ~tpeak_s after
administration, driving a sigmoid Emax MAP rise capped at dmap_max. Documented
approximation for the prototype.

For a unit impulse with equal rate constants k = 1/tpeak, the effect response is
k*t*exp(-k*t), which peaks at t = tpeak. Steady infusion drives plasma (and thus
effect) toward input/k, so the sigmoid saturates at dmap_max under heavy dosing.
"""
import numpy as np
from twin.pkpd.params import NorepiParams


def simulate_map_rise(params: NorepiParams, rate_ug_min: np.ndarray, dt: float) -> np.ndarray:
    p = params
    k = 1.0 / p.tpeak_s  # per second; equal plasma/effect rate -> peak at tpeak
    n = len(rate_ug_min)
    plasma = 0.0
    effect = 0.0
    conc = np.zeros(n)
    for i in range(n):
        inp = rate_ug_min[i] / 60.0  # ug/s
        plasma = plasma + dt * (inp - k * plasma)
        effect = effect + dt * k * (plasma - effect)
        conc[i] = effect
    conc_g = np.power(np.clip(conc, 0, None), p.gamma)
    return p.dmap_max * conc_g / (p.ec50 ** p.gamma + conc_g)
