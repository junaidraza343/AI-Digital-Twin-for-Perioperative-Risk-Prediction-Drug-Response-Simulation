import pytest
pytest.importorskip("torch")
import torch
from twin.models.coupled import CoupledTwin


def test_forward_shapes_and_delta_bounds():
    model = CoupledTwin(n_features=12, latent_dim=32, delta_bound=0.7)
    x = torch.randn(8, 12)
    logit, delta, z = model(x)
    assert logit.shape == (8,)
    assert delta.shape == (8, 6)
    assert z.shape == (8, 32)
    assert torch.all(delta.abs() <= 0.7 + 1e-5)


def test_encoder_is_swappable_module():
    # The encoder must be an attribute we can replace (SP2 hand-off).
    model = CoupledTwin(n_features=12, latent_dim=32)
    assert hasattr(model, "encoder")
    assert isinstance(model.encoder, torch.nn.Module)
