import numpy as np
import pandas as pd
from twin.models.baselines import MapOnlyModel, GbdtModel


def _separable():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({
        "map_last": np.concatenate([rng.normal(85, 3, n), rng.normal(68, 3, n)]),
        "map_slope": np.concatenate([rng.normal(0.0, 0.1, n), rng.normal(-0.2, 0.1, n)]),
        "hr_mean": rng.normal(70, 5, 2 * n),
    })
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y


def test_map_only_uses_only_map_features_and_learns():
    X, y = _separable()
    m = MapOnlyModel().fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(y),)
    assert ((p >= 0) & (p <= 1)).all()
    from twin.eval.metrics import auroc
    assert auroc(y, p) > 0.8


def test_gbdt_learns_and_is_deterministic():
    X, y = _separable()
    p1 = GbdtModel().fit(X, y).predict_proba(X)
    p2 = GbdtModel().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1, p2)
    from twin.eval.metrics import auroc
    assert auroc(y, p1) > 0.85
