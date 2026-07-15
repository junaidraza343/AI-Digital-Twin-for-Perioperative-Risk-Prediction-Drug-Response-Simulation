import numpy as np
from twin.pkpd.params import POP_PROPOFOL
from twin.pkpd.schedule import rate_vector
from twin.pkpd.propofol import simulate_effect_site


def test_zero_dose_keeps_ce_zero():
    ce = simulate_effect_site(POP_PROPOFOL, rate_vector([], 1.0, 300.0), dt=1.0)
    assert np.allclose(ce, 0.0)


def test_infusion_raises_ce_monotonically_during_dosing():
    r = rate_vector([(0.0, 600.0, 20.0)], dt=1.0, duration=600.0)  # 20 mg/min, 10 min
    ce = simulate_effect_site(POP_PROPOFOL, r, dt=1.0)
    # effect-site concentration rises over the infusion
    assert ce[-1] > ce[60] > ce[1] > 0.0


def test_effect_site_lags_plasma():
    # after stopping a bolus-like infusion, Ce keeps rising briefly (lag) then falls
    r = rate_vector([(0.0, 30.0, 200.0)], dt=1.0, duration=600.0)
    ce = simulate_effect_site(POP_PROPOFOL, r, dt=1.0)
    peak_idx = int(np.argmax(ce))
    assert peak_idx > 30  # peak effect-site occurs after infusion ends at t=30s


def test_deterministic():
    r = rate_vector([(0.0, 60.0, 10.0)], dt=1.0, duration=300.0)
    a = simulate_effect_site(POP_PROPOFOL, r, dt=1.0)
    b = simulate_effect_site(POP_PROPOFOL, r, dt=1.0)
    assert np.array_equal(a, b)
