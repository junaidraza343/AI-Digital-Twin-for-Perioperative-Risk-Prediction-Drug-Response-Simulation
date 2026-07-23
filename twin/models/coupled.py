"""Coupled twin: shared encoder E -> latent z; prediction head P -> IOH logit;
calibration head C -> 6 personalization deltas. The shared z is the coupling.

Encoder is a plain nn.Module attribute so SP2 can swap its body (CNN-GRU waveform
stack) without touching the heads, losses, or training loop."""
import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    """MLP over standardized window features -> latent z. Replaced in SP2."""

    def __init__(self, n_features, latent_dim=32, hidden=64, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, latent_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CoupledTwin(nn.Module):
    def __init__(self, n_features, latent_dim=32, delta_bound=0.7, encoder=None):
        super().__init__()
        self.delta_bound = delta_bound
        self.encoder = encoder or TabularEncoder(n_features, latent_dim)
        self.pred_head = nn.Linear(latent_dim, 1)
        self.calib_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 6),
        )

    def forward(self, x):
        z = self.encoder(x)
        logit = self.pred_head(z).squeeze(-1)
        delta = self.delta_bound * torch.tanh(self.calib_head(z))
        return logit, delta, z
