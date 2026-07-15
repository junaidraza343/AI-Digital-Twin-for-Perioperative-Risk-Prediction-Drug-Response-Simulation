"""Apply the calibration head's 6 multiplicative perturbations: theta = theta_pop * exp(delta).

In this prototype deltas are set manually to demonstrate the personalization
interface; the learned head is SP4.
"""
import math
from dataclasses import replace
from twin.pkpd.params import PropofolParams, PDParams

DELTA_KEYS = ("V1", "V2", "V3", "ke0", "EC50", "gamma")


def apply_deltas(prop: PropofolParams, pd: PDParams, deltas: dict):
    d = {k: float(deltas.get(k, 0.0)) for k in DELTA_KEYS}
    prop2 = replace(
        prop,
        V1=prop.V1 * math.exp(d["V1"]),
        V2=prop.V2 * math.exp(d["V2"]),
        V3=prop.V3 * math.exp(d["V3"]),
        ke0=prop.ke0 * math.exp(d["ke0"]),
    )
    pd2 = replace(
        pd,
        ec50=pd.ec50 * math.exp(d["EC50"]),
        gamma=pd.gamma * math.exp(d["gamma"]),
    )
    return prop2, pd2
