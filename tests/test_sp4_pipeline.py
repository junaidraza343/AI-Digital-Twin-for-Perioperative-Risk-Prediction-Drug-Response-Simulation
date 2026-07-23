import numpy as np
import pytest
pytest.importorskip("torch")
import twin.config as c
from scripts.sp4_pipeline import run_stage1_cohort, run_stage2_cohort


def _fake_array(T=360, dip=(200, 290)):
    cols = c.NUMERIC_TRACKS
    a = np.full((T, len(cols)), np.nan)
    map_ = np.full(T, 85.0)
    map_[dip[0]:dip[1]] = 55.0            # hypotensive episode -> some y=1 windows
    a[:, cols.index(c.MAP_TRACK)] = map_
    a[:, cols.index(c.PROPOFOL_TRACK)] = 60.0
    a[:, cols.index("Solar8000/HR")] = 70.0
    return a


@pytest.fixture
def small_windows(monkeypatch, tmp_path):
    # shrink window sizing so short synthetic cases produce prediction windows
    monkeypatch.setattr(c, "OBS_WINDOW_SECONDS", 60)
    monkeypatch.setattr(c, "HORIZON_SECONDS", 60)
    monkeypatch.setattr(c, "STRIDE_SECONDS", 30)
    monkeypatch.setattr(c, "EVENT_MIN_SECONDS", 10)
    monkeypatch.setattr(c, "DATA_CACHE", tmp_path)   # don't touch the real cache


def _cohort_and_loader():
    statics = {1: {"age": 55, "weight": 72, "height": 170, "sex": "M", "asa": 2},
               2: {"age": 40, "weight": 60, "height": 165, "sex": "F", "asa": 1},
               3: {"age": 66, "weight": 88, "height": 180, "sex": "M", "asa": 3}}
    cohort = [(cid, statics[cid]) for cid in statics]

    def loader(caseid, tracks, interval):
        shift = {1: 0, 2: 10, 3: -10}[caseid]
        return _fake_array(dip=(200 + shift, 290 + shift))

    return cohort, loader


def test_stage1_then_stage2_pipeline(small_windows):
    cohort, loader = _cohort_and_loader()
    model, prep, cols, info = run_stage1_cohort(
        cohort, loader_fn=loader, teacher_iters=30, epochs=60, lr=1e-2,
        latent_dim=16, device="cpu")
    assert info["n_cases"] == 3 and info["n_windows"] > 0
    assert len(cols) > 0
    assert info["distill_loss_last"] < info["distill_loss_first"]  # head learns targets

    model, s2 = run_stage2_cohort(
        cohort, model, prep, cols, loader_fn=loader, epochs=20, lr=5e-3, device="cpu")
    assert np.isfinite(s2["recon_first"]) and np.isfinite(s2["recon_last"])
    # fine-tuning from the distilled init must not blow up reconstruction
    assert s2["recon_last"] <= s2["recon_first"] + 1e-3
