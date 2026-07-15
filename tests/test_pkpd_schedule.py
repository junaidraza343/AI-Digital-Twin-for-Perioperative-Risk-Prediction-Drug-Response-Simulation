import numpy as np
from twin.pkpd.schedule import rate_vector


def test_constant_infusion_fills_window():
    # 10 mg/min from t=0 to t=60s, dt=1s, duration=120s
    v = rate_vector([(0.0, 60.0, 10.0)], dt=1.0, duration=120.0)
    assert v.shape == (121,)
    assert np.allclose(v[:60], 10.0)
    assert np.allclose(v[61:], 0.0)


def test_empty_schedule_is_zeros():
    v = rate_vector([], dt=1.0, duration=10.0)
    assert v.shape == (11,) and np.all(v == 0.0)


def test_overlapping_segments_sum():
    v = rate_vector([(0.0, 10.0, 5.0), (0.0, 10.0, 3.0)], dt=1.0, duration=10.0)
    assert np.allclose(v[:10], 8.0)
