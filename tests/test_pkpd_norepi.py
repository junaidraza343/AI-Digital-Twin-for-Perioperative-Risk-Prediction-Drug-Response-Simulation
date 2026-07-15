import numpy as np
from twin.pkpd.params import POP_NOREPI
from twin.pkpd.schedule import rate_vector
from twin.pkpd.norepi import simulate_map_rise


def test_zero_dose_no_rise():
    rise = simulate_map_rise(POP_NOREPI, rate_vector([], 1.0, 300.0), dt=1.0)
    assert np.allclose(rise, 0.0)


def test_bolus_produces_transient_peak():
    r = rate_vector([(10.0, 11.0, 500.0)], dt=1.0, duration=300.0)  # ~bolus at t=10s
    rise = simulate_map_rise(POP_NOREPI, r, dt=1.0)
    peak_idx = int(np.argmax(rise))
    assert rise[peak_idx] > 0.0
    assert peak_idx > 10  # peak after the bolus


def test_rise_capped_at_dmap_max():
    r = rate_vector([(0.0, 300.0, 1e5)], dt=1.0, duration=300.0)
    rise = simulate_map_rise(POP_NOREPI, r, dt=1.0)
    assert rise.max() <= POP_NOREPI.dmap_max + 1e-6
