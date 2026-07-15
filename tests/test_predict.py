import numpy as np
import pandas as pd

from twin.models.baselines import MapOnlyModel
from twin.predict import ioh_probability


def _toy_model():
    """A MAP-only model where low MAP / falling slope -> high IOH probability."""
    rng = np.random.default_rng(0)
    map_last = rng.uniform(50, 100, 400)
    map_slope = rng.uniform(-1, 1, 400)
    # label: hypotensive if low MAP or steeply falling
    y = ((map_last < 68) | (map_slope < -0.4)).astype(int)
    X = pd.DataFrame({"map_last": map_last, "map_slope": map_slope})
    return MapOnlyModel().fit(X, y)


def test_probability_length_matches_trajectory():
    m = _toy_model()
    traj = np.full(300, 85.0)
    p = ioh_probability(traj, dt=1.0, model=m)
    assert p.shape == (300,)
    assert np.all((p >= 0) & (p <= 1))


def test_lower_map_gives_higher_probability():
    m = _toy_model()
    high = ioh_probability(np.full(300, 90.0), dt=1.0, model=m)
    low = ioh_probability(np.full(300, 58.0), dt=1.0, model=m)
    assert low.mean() > high.mean()


def test_falling_trajectory_probability_rises():
    m = _toy_model()
    traj = np.linspace(95, 60, 600)  # steadily falling MAP
    p = ioh_probability(traj, dt=1.0, model=m)
    assert p[-1] > p[0]
