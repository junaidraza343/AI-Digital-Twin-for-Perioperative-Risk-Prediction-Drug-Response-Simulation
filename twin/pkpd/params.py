"""Population PK-PD parameters from the literature. Constants only; no logic.

Propofol PK: Eleveld 2018 (BJA 120(5):942-959), representative 50 yo / 70 kg adult.
Propofol->MAP PD: sigmoid Emax (fractional MAP reduction).
Norepinephrine: Joachim 2024 (BJCP 90(11):2861-2869), summarized as tpeak / dMAP_max.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PropofolParams:
    V1: float   # L, central volume
    V2: float   # L, shallow peripheral
    V3: float   # L, deep peripheral
    CL: float   # L/min, metabolic clearance
    Q2: float   # L/min, central<->shallow
    Q3: float   # L/min, central<->deep
    ke0: float  # 1/min, effect-site equilibration


@dataclass(frozen=True)
class PDParams:
    emax: float   # max fractional MAP reduction (0-1)
    ec50: float   # ug/mL, half-effect concentration
    gamma: float  # Hill slope


@dataclass(frozen=True)
class NorepiParams:
    tpeak_s: float    # s, bolus time-to-peak MAP rise
    dmap_max: float   # max fractional MAP rise (0-1)
    ec50: float       # ug/min-equivalent, half-effect infusion level
    gamma: float      # Hill slope


POP_PROPOFOL = PropofolParams(V1=6.3, V2=25.0, V3=270.0, CL=1.8, Q2=1.7, Q3=0.84, ke0=0.146)
POP_PD = PDParams(emax=0.30, ec50=4.0, gamma=2.5)
POP_NOREPI = NorepiParams(tpeak_s=74.0, dmap_max=0.24, ec50=5.0, gamma=2.0)
