import numpy as np
from twin.pkpd.params import POP_PD
from twin.pkpd.pd_model import emax_reduction, combine_map


def test_zero_concentration_no_effect():
    assert emax_reduction(np.array([0.0]), POP_PD)[0] == 0.0


def test_half_effect_at_ec50():
    r = emax_reduction(np.array([POP_PD.ec50]), POP_PD)[0]
    assert abs(r - POP_PD.emax / 2) < 1e-9


def test_saturates_at_emax():
    r = emax_reduction(np.array([1e6]), POP_PD)[0]
    assert abs(r - POP_PD.emax) < 1e-6


def test_monotone_increasing_in_ce():
    ce = np.linspace(0, 20, 50)
    r = emax_reduction(ce, POP_PD)
    assert np.all(np.diff(r) >= 0)


def test_combine_map_applies_reduction_and_rise():
    traj = combine_map(map0=90.0, prop_reduction=np.array([0.0, 0.2]),
                       norepi_rise=np.array([0.0, 0.1]))
    assert traj[0] == 90.0
    assert abs(traj[1] - (90.0 * (1 - 0.2) + 90.0 * 0.1)) < 1e-9
