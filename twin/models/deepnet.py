"""Device-agnostic PyTorch tabular deep net for IOH prediction.

A small MLP over the same window features as the tree models. Picks CUDA (Colab /
cloud) > MPS (Apple Silicon) > CPU automatically, so the identical code trains on a
GPU when one is present and falls back to CPU otherwise. Same fit / predict_proba
interface as the sklearn baselines, so it drops into the selection-bias study.
"""
import numpy as np

import twin.config as c

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def pick_device():
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _numeric_cols(X):
    return [col for col in X.columns if X[col].dtype != object]


if _HAS_TORCH:
    class _MLP(nn.Module):
        def __init__(self, d, hidden=(128, 64), p=0.2):
            super().__init__()
            layers, prev = [], d
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
                prev = h
            layers += [nn.Linear(prev, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


class TorchTabularModel:
    """MLP classifier with median-impute + standardize; imbalance-aware BCE."""

    def __init__(self, hidden=(128, 64), epochs=40, lr=1e-3, batch=512, seed=None):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch not installed; `pip install torch`.")
        self.hidden, self.epochs, self.lr, self.batch = hidden, epochs, lr, batch
        self.seed = c.SEED if seed is None else seed
        self.device = pick_device()
        self.columns = self.med = self.mu = self.sd = self.net = None

    def _prep(self, X, fit=False):
        A = X[self.columns].to_numpy(dtype=float)
        if fit:
            self.med = np.nanmedian(A, axis=0)
        inds = np.where(np.isnan(A))
        A[inds] = np.take(self.med, inds[1])
        if fit:
            self.mu = A.mean(axis=0)
            self.sd = A.std(axis=0) + 1e-6
        return (A - self.mu) / self.sd

    def fit(self, X, y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.columns = _numeric_cols(X)
        A = self._prep(X, fit=True).astype(np.float32)
        y = np.asarray(y, dtype=np.float32)
        dev = self.device
        Xt = torch.tensor(A, device=dev); yt = torch.tensor(y, device=dev)
        self.net = _MLP(A.shape[1], self.hidden).to(dev)
        pos = float(max(y.sum(), 1)); neg = float(max(len(y) - y.sum(), 1))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=dev))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        n = len(y)
        self.net.train(True)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward(); opt.step()
        return self

    def predict_proba(self, X):
        A = self._prep(X, fit=False).astype(np.float32)
        self.net.train(False)   # inference mode (equivalent to .eval())
        with torch.no_grad():
            logits = self.net(torch.tensor(A, device=self.device))
            return torch.sigmoid(logits).cpu().numpy()

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "columns": self.columns,
                    "med": self.med, "mu": self.mu, "sd": self.sd,
                    "hidden": self.hidden}, path)
