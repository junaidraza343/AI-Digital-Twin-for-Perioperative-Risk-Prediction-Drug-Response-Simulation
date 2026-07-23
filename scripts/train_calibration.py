"""Train the SP4 calibration head. Stage 1 distills from the per-case teacher;
Stage 2 fine-tunes end-to-end (added in Task 6). Device-agnostic (pick_device).

Usage:
  python -m scripts.train_calibration --stage 1 [--smoke]
  python -m scripts.train_calibration --stage 2 [--smoke]
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from twin.models.deepnet import pick_device
from twin.models.coupled import CoupledTwin


class FeaturePrep:
    """Median-impute + standardize numeric feature columns (fit on train)."""

    def __init__(self):
        self.columns = self.med = self.mu = self.sd = None

    def fit(self, df):
        self.columns = [c for c in df.columns if df[c].dtype != object]
        A = df[self.columns].to_numpy(dtype=float)
        self.med = np.nanmedian(A, axis=0)
        A = self._impute(A)
        self.mu = A.mean(axis=0); self.sd = A.std(axis=0) + 1e-6
        return self

    def _impute(self, A):
        idx = np.where(np.isnan(A))
        A = A.copy(); A[idx] = np.take(self.med, idx[1])
        return A

    def transform(self, df):
        A = self._impute(df[self.columns].to_numpy(dtype=float))
        return (A - self.mu) / self.sd


def distill_step(model, opt, x, target_delta):
    """One optimization step regressing calib head -> target_delta. Returns MSE."""
    model.train()
    opt.zero_grad()
    _, delta, _ = model(x)
    loss = nn.functional.mse_loss(delta, target_delta)
    loss.backward(); opt.step()
    return float(loss.detach())


# --- orchestration (Stage 1) -------------------------------------------------

def run_stage1(smoke=False):
    """Build features + per-case teacher deltas, train the calib head to match.
    In --smoke mode uses a tiny synthetic dataset so it runs on CPU in seconds."""
    device = pick_device()
    if smoke:
        n, f = 64, 6
        x = torch.randn(n, f, device=device)
        target = torch.zeros(n, 6, device=device); target[:, 4] = 0.3
        model = CoupledTwin(n_features=f, latent_dim=16).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(100):
            loss = distill_step(model, opt, x, target)
        print(f"[stage1-smoke] final distill MSE={loss:.4f}")
        return model
    raise NotImplementedError(
        "Full Stage 1 needs the SP4 cohort (Task 1 Step 10) + teacher deltas; "
        "run with --smoke on laptop, full run on the GPU harness (Task 8).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=(1, 2))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.stage == 1:
        run_stage1(smoke=args.smoke)
    else:
        from scripts.train_calibration_stage2 import run_stage2  # Task 6
        run_stage2(smoke=args.smoke)


if __name__ == "__main__":
    main()
