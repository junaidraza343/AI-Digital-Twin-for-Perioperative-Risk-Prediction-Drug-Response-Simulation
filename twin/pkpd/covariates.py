"""Covariate scaling of propofol PK parameters.

REDUCED FORM of Eleveld 2018 for the prototype: volumes scale linearly with
weight vs a 70 kg reference; clearances scale by weight^0.75 (allometric);
CL and ke0 carry a mild linear age effect (2.5% per decade below/above 35 yr,
clamped). Documented as an approximation; not the full maturation/sigmoid model.
"""
from dataclasses import replace
from twin.pkpd.params import PropofolParams

_REF_WEIGHT = 70.0
_REF_AGE = 35.0


def scale_propofol(params: PropofolParams, age: float, weight: float,
                   height: float, sex: str) -> PropofolParams:
    w_lin = weight / _REF_WEIGHT
    w_allo = (weight / _REF_WEIGHT) ** 0.75
    age_factor = max(0.5, 1.0 - 0.0025 * (age - _REF_AGE))  # ~2.5%/decade, clamped
    return replace(
        params,
        V1=params.V1 * w_lin,
        V2=params.V2 * w_lin,
        V3=params.V3 * w_lin,
        CL=params.CL * w_allo * age_factor,
        Q2=params.Q2 * w_allo,
        Q3=params.Q3 * w_allo,
        ke0=params.ke0 * age_factor,
    )
