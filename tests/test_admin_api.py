"""Smoke tests for the admin / ML-ops endpoints."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from api.main import app
import twin.config as c

client = TestClient(app)
_HAS_DATA = (c.RESULTS_DIR / "windows.parquet").exists()


def test_status_shape():
    d = client.get("/api/admin/status").json()
    assert {"cached_cases", "n_windows", "model_trained", "demo_cases"} <= set(d)


def test_admin_page_served():
    r = client.get("/admin")
    assert r.status_code == 200 and "ML OPS" in r.text.upper()


def test_metrics_endpoint():
    d = client.get("/api/admin/metrics").json()
    assert "available" in d and "rows" in d


@pytest.mark.skipif(not _HAS_DATA, reason="no windows.parquet")
def test_curves_has_both_models():
    d = client.get("/api/admin/curves").json()
    assert d["available"] is True
    assert {"map_only", "gbdt"} <= set(d["models"])
    assert 0.5 <= d["models"]["map_only"]["auroc"] <= 1.0


@pytest.mark.skipif(not _HAS_DATA, reason="no windows.parquet")
@pytest.mark.slow
def test_retrain_predictor(monkeypatch):
    """Retrain is a polled background job, so the POST only hands back a job id.

    Marked slow: it runs a REAL retrain over the full windows.parquet.
    """
    import time
    import api.admin as admin

    monkeypatch.setenv("ADMIN_TOKEN", "test-token")
    hdr = {"X-Admin-Token": "test-token"}
    job_id = client.post("/api/admin/train-predictor", headers=hdr).json()["job_id"]
    assert admin._JOBS[job_id]["kind"] == "train"

    deadline = time.time() + 1800
    while client.get(f"/api/admin/job/{job_id}").json()["status"] == "running":
        assert time.time() < deadline, "retrain job did not finish in 30 min"
        time.sleep(5)

    d = client.get(f"/api/admin/job/{job_id}").json()
    assert d["status"] == "done", d.get("error")
    assert len(d["rows"]) == 12          # 4 models x 3 regimes
    assert d["predictor_meta"]["unbiased"]["auroc"] > 0.5


def test_build_job_lifecycle(monkeypatch):
    # Patch the subprocess launcher so the test NEVER rebuilds the real dataset;
    # just confirms the job registry + polling machinery works.
    import subprocess, sys
    import api.admin as admin

    def fake_popen(n_cases, f, env):
        return subprocess.Popen([sys.executable, "-c", "print('fake build ok')"],
                                stdout=f, stderr=subprocess.STDOUT)

    monkeypatch.setattr(admin, "subprocess_popen", fake_popen)
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")
    b = client.post("/api/admin/build?n_cases=5",
                    headers={"X-Admin-Token": "test-token"}).json()
    assert "job_id" in b
    j = client.get(f"/api/admin/job/{b['job_id']}").json()
    assert j["status"] in {"running", "done", "failed"}


# ------------------------------------------------------ production hardening
def test_mutating_admin_endpoints_are_disabled_without_a_token(monkeypatch):
    """Fail closed: with no ADMIN_TOKEN configured, compute-spawning endpoints
    must refuse. /api/admin/build starts a VitalDB subprocess and train-predictor
    starts a training thread; unauthenticated they are a remote compute-exhaustion
    trigger on a public deployment."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    for path in ("/api/admin/train-predictor", "/api/admin/rescan-cases",
                 "/api/admin/build"):
        r = client.post(path)
        assert r.status_code == 503, (path, r.status_code)


def test_mutating_admin_endpoints_reject_a_wrong_token(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "correct-horse")
    r = client.post("/api/admin/rescan-cases", headers={"X-Admin-Token": "nope"})
    assert r.status_code == 401


def test_mutating_admin_endpoints_accept_the_configured_token(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "correct-horse")
    r = client.post("/api/admin/rescan-cases",
                    headers={"X-Admin-Token": "correct-horse"})
    assert r.status_code == 200


def test_build_rejects_an_absurd_case_count(monkeypatch):
    """n_cases is caller-controlled and drives a download loop; bound it."""
    monkeypatch.setenv("ADMIN_TOKEN", "correct-horse")
    r = client.post("/api/admin/build?n_cases=100000",
                    headers={"X-Admin-Token": "correct-horse"})
    assert r.status_code == 422


def test_readonly_admin_endpoints_stay_open(monkeypatch):
    """Observability must not require the token; it spawns nothing."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert client.get("/api/admin/status").status_code == 200
