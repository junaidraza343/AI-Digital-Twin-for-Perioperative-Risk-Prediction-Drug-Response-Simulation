"""Propofol 3-compartment PK + effect-site compartment.

State: amounts A1,A2,A3 (mg) in central/shallow/deep; effect-site conc Ce (ug/mL).
Micro-rate constants from clearances and volumes. Explicit Euler at dt seconds;
rates are per-minute so we convert with dt/60. Cp = A1/V1 (mg/L = ug/mL).
"""
import numpy as np
from twin.pkpd.params import PropofolParams


def simulate_effect_site(params: PropofolParams, rate_mg_min: np.ndarray, dt: float) -> np.ndarray:
    p = params
    k10 = p.CL / p.V1
    k12 = p.Q2 / p.V1
    k21 = p.Q2 / p.V2
    k13 = p.Q3 / p.V1
    k31 = p.Q3 / p.V3
    step = dt / 60.0  # per-minute rate constants -> per-step

    n = len(rate_mg_min)
    a1 = a2 = a3 = 0.0
    ce = np.zeros(n)
    ce_prev = 0.0
    for i in range(n):
        infusion = rate_mg_min[i] * step  # mg entering central this step
        da1 = (-(k10 + k12 + k13) * a1 + k21 * a2 + k31 * a3) * step + infusion
        da2 = (k12 * a1 - k21 * a2) * step
        da3 = (k13 * a1 - k31 * a3) * step
        a1 += da1
        a2 += da2
        a3 += da3
        cp = a1 / p.V1
        ce_prev = ce_prev + p.ke0 * step * (cp - ce_prev)
        ce[i] = ce_prev
    return ce
