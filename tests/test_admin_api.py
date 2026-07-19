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
def test_retrain_predictor():
    """Retrain is a polled background job, so the POST only hands back a job id."""
    import time
    import api.admin as admin

    job_id = client.post("/api/admin/train-predictor").json()["job_id"]
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
    b = client.post("/api/admin/build?n_cases=5").json()
    assert "job_id" in b
    j = client.get(f"/api/admin/job/{b['job_id']}").json()
    assert j["status"] in {"running", "done", "failed"}
