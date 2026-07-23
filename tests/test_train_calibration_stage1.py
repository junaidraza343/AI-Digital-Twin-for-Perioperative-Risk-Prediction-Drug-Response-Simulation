import numpy as np
import pytest
pytest.importorskip("torch")
import torch
from scripts.train_calibration import FeaturePrep, distill_step


def test_distill_step_reduces_delta_mse():
    torch.manual_seed(0)
    from twin.models.coupled import CoupledTwin
    model = CoupledTwin(n_features=6, latent_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(32, 6)
    target_delta = torch.zeros(32, 6)
    target_delta[:, 4] = 0.3  # constant EC50 target

    first = distill_step(model, opt, x, target_delta)
    for _ in range(200):
        last = distill_step(model, opt, x, target_delta)
    assert last < first
    assert last < 0.05


def test_feature_prep_imputes_and_standardizes():
    import pandas as pd
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 20.0, 30.0]})
    prep = FeaturePrep().fit(df)
    A = prep.transform(df)
    assert not np.isnan(A).any()
    assert abs(A[:, 1].mean()) < 1e-6  # standardized column ~zero mean
