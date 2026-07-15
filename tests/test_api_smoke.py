"""Smoke test for the FastAPI backend."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "PERIOPERATIVE" in r.text.upper()


def test_simulate_shapes_and_risk():
    r = client.post("/api/simulate", json={
        "age": 60, "weight": 70, "height": 170, "sex": "M", "map0": 90,
        "duration_min": 15, "propofol": 30, "norepi": 0, "norepi_start_min": 0,
        "personalize": False, "show_band": True,
    })
    assert r.status_code == 200
    d = r.json()
    n = len(d["t_min"])
    assert len(d["map_pop"]) == n == len(d["ce"]) == len(d["band_lo"]) == len(d["band_hi"])
    assert d["risk"] in {"Low", "Medium", "High"}
    assert d["threshold"] == 65.0


def test_heavier_propofol_lowers_min_map():
    def minmap(rate):
        return client.post("/api/simulate", json={
            "age": 80, "weight": 55, "height": 165, "sex": "F", "map0": 85,
            "duration_min": 15, "propofol": rate, "show_band": False,
        }).json()["min_map"]
    assert minmap(50) < minmap(10)


def test_validation_rejects_out_of_range():
    r = client.post("/api/simulate", json={"age": 5, "propofol": 30})
    assert r.status_code == 422
