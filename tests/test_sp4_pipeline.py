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


def test_stage1_skips_unloadable_case_and_trains_on_the_rest(small_windows):
    """A single bad case must not abort a batched cohort fit.

    Teacher fitting is batched across cases, so a case that fails to load has to
    be dropped before the batch is assembled rather than poisoning its chunk.
    """
    cohort, good_loader = _cohort_and_loader()

    def flaky_loader(caseid, tracks, interval):
        if caseid == 2:
            raise RuntimeError("simulated VitalDB pull failure")
        return good_loader(caseid, tracks, interval)

    _, _, _, info = run_stage1_cohort(
        cohort, loader_fn=flaky_loader, teacher_iters=10, epochs=5, lr=1e-2,
        device="cpu", teacher_batch_size=8)
    assert info["n_cases"] == 2, info


def test_stage1_teacher_batch_size_does_not_change_case_count(small_windows):
    """Chunking is a throughput knob, not a modelling one."""
    cohort, loader = _cohort_and_loader()
    kw = dict(loader_fn=loader, teacher_iters=10, epochs=5, lr=1e-2, device="cpu")
    _, _, _, big = run_stage1_cohort(cohort, teacher_batch_size=8, **kw)
    _, _, _, small = run_stage1_cohort(cohort, teacher_batch_size=1, **kw)
    assert big["n_cases"] == small["n_cases"] == 3


def test_stage2_case_batching_preserves_case_count_and_reduces_recon(small_windows):
    """Stage 2 steps over minibatches of cases, not one case at a time.

    Reconstruction runs the twin over the full record, so a per-case step means
    epochs x cases simulations. Batching cases makes the cohort tractable; it must
    still fit every case and still drive reconstruction down.
    """
    cohort, loader = _cohort_and_loader()
    model, prep, cols, _ = run_stage1_cohort(
        cohort, loader_fn=loader, teacher_iters=10, epochs=10, lr=1e-2, device="cpu")
    _, info = run_stage2_cohort(
        cohort, model, prep, cols, loader_fn=loader, epochs=15, lr=5e-3,
        device="cpu", case_batch_size=2)
    assert info["n_cases"] == 3
    assert info["recon_last"] < info["recon_first"], info


def test_load_case_rejects_artifact_baseline_and_cleans_observed(small_windows):
    """An arterial line zeroed at the start must not become the patient's baseline.

    Real VitalDB records open with flush/zeroing artifacts (case 16 starts at
    12 mmHg). Seeding map0 from the first sample drives the twin into its MAP_MIN
    clamp, where the gradient is identically zero and the teacher cannot fit.
    """
    from scripts.sp4_pipeline import load_case
    cols = c.NUMERIC_TRACKS

    def artifact_loader(caseid, tracks, interval):
        a = _fake_array()
        map_ = a[:, cols.index(c.MAP_TRACK)].copy()
        map_[:5] = [12.0, 36.0, 249.0, 10.0, 39.0]   # zeroing + flush transients
        a[:, cols.index(c.MAP_TRACK)] = map_
        return a

    case = load_case(1, {"age": 55, "weight": 72, "height": 170, "sex": "M"},
                     loader_fn=artifact_loader)
    assert 40.0 <= case["patient"].map0 <= 200.0, case["patient"].map0
    obs = case["observed"].numpy()[0]
    valid = obs[~np.isnan(obs)]
    assert valid.min() >= 40.0 and valid.max() <= 200.0
