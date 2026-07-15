import math
from twin.pkpd.params import POP_PROPOFOL, POP_PD
from twin.pkpd.calibration import apply_deltas, DELTA_KEYS


def test_zero_deltas_reproduce_population():
    prop, pd = apply_deltas(POP_PROPOFOL, POP_PD, {k: 0.0 for k in DELTA_KEYS})
    assert prop == POP_PROPOFOL and pd == POP_PD


def test_positive_delta_v1_scales_v1():
    prop, _ = apply_deltas(POP_PROPOFOL, POP_PD, {"V1": 0.2})
    assert math.isclose(prop.V1, POP_PROPOFOL.V1 * math.exp(0.2), rel_tol=1e-9)


def test_delta_ec50_scales_pd():
    _, pd = apply_deltas(POP_PROPOFOL, POP_PD, {"EC50": -0.3})
    assert math.isclose(pd.ec50, POP_PD.ec50 * math.exp(-0.3), rel_tol=1e-9)


def test_delta_keys_are_the_six():
    assert set(DELTA_KEYS) == {"V1", "V2", "V3", "ke0", "EC50", "gamma"}
