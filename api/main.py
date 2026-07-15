"""FastAPI backend for the perioperative digital-twin dashboard.

Exposes the mechanistic PK-PD engine, the trained IOH predictor, data-driven
calibration, and real VitalDB cases as a JSON API, and serves the static
clinical-instrument frontend. Stateless; every request is a fresh projection.
"""
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from twin.pkpd.engine import Patient, project_map, project_band, ioh_risk
from twin.pkpd.fitting import fit_deltas
from twin.predict import load_predictor, ioh_probability

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="Perioperative Digital Twin", version="2.0")

# Load the trained predictor once at startup (None if not trained yet).
PREDICTOR, PREDICTOR_META = load_predictor()


class SimRequest(BaseModel):
    age: float = Field(60, ge=18, le=90)
    weight: float = Field(70, ge=40, le=140)
    height: float = Field(170, ge=140, le=200)
    sex: str = Field("M", pattern="^[MF]$")
    map0: float = Field(90, ge=60, le=120)
    duration_min: float = Field(15, ge=5, le=30)
    propofol: float = Field(30, ge=0, le=80)      # mg/min infusion
    norepi: float = Field(0, ge=0, le=30)         # ug/min infusion
    norepi_start_min: float = Field(0, ge=0, le=30)
    personalize: bool = False
    delta_ec50: float = Field(0.0, ge=-0.7, le=0.7)
    delta_ke0: float = Field(0.0, ge=-0.7, le=0.7)
    show_band: bool = True


class CalibrateRequest(BaseModel):
    age: float = Field(60, ge=18, le=90)
    weight: float = Field(70, ge=40, le=140)
    height: float = Field(170, ge=140, le=200)
    sex: str = Field("M", pattern="^[MF]$")
    map0: float = Field(90, ge=60, le=120)
    duration_min: float = Field(15, ge=5, le=30)
    propofol: float = Field(30, ge=0, le=80)
    norepi: float = Field(0, ge=0, le=30)
    norepi_start_min: float = Field(0, ge=0, le=30)
    observed_map: list[float] = Field(..., min_length=2)


def _schedules(propofol, norepi, norepi_start_min, duration_s):
    prop = [(0.0, duration_s, propofol)] if propofol > 0 else []
    ne = ([(norepi_start_min * 60.0, duration_s, norepi)] if norepi > 0 else [])
    return prop, ne


def _predict_prob(map_traj: np.ndarray, dt: float = 1.0):
    """(peak probability, downsampled curve) or (None, None) if no model."""
    if PREDICTOR is None:
        return None, None
    prob = ioh_probability(map_traj, dt=dt, model=PREDICTOR)
    stride = max(1, len(prob) // 300)
    return round(float(np.max(prob)), 4), prob[::stride].round(4).tolist()


@app.post("/api/simulate")
def simulate(req: SimRequest):
    duration_s = req.duration_min * 60.0
    patient = Patient(age=req.age, weight=req.weight, height=req.height,
                      sex=req.sex, map0=req.map0)
    prop, ne = _schedules(req.propofol, req.norepi, req.norepi_start_min, duration_s)
    deltas = ({"EC50": req.delta_ec50, "ke0": req.delta_ke0}
              if req.personalize else None)

    pop = project_map(patient, prop, ne, duration=duration_s)
    shown = project_map(patient, prop, ne, duration=duration_s, deltas=deltas)

    peak_prob, prob_curve = _predict_prob(shown.map)
    resp = {
        "t_min": (shown.t / 60.0).round(4).tolist(),
        "map_pop": pop.map.round(3).tolist(),
        "map_shown": shown.map.round(3).tolist(),
        "ce": shown.ce.round(4).tolist(),
        "personalized": bool(req.personalize),
        "risk": ioh_risk(shown),
        "min_map": round(float(shown.map.min()), 1),
        "minutes_below_65": round(float(shown.minutes_below_65), 1),
        "ce_end": round(float(shown.ce[-1]), 2),
        "threshold": 65.0,
        "ioh_prob": peak_prob,
        "prob_curve": prob_curve,
    }
    if req.show_band:
        lo, hi, _ = project_band(patient, prop, ne, duration=duration_s, deltas=deltas)
        resp["band_lo"] = lo.round(3).tolist()
        resp["band_hi"] = hi.round(3).tolist()
    return resp


@app.post("/api/calibrate")
def calibrate(req: CalibrateRequest):
    """Fit the calibration deltas to an observed MAP (CPU stand-in for SP4 head)."""
    duration_s = req.duration_min * 60.0
    patient = Patient(age=req.age, weight=req.weight, height=req.height,
                      sex=req.sex, map0=req.map0)
    prop, ne = _schedules(req.propofol, req.norepi, req.norepi_start_min, duration_s)
    # Resample the (possibly downsampled) observed MAP onto the per-second grid.
    n = int(duration_s) + 1
    obs = np.interp(np.linspace(0, 1, n),
                    np.linspace(0, 1, len(req.observed_map)), req.observed_map)
    # Fit over the induction window only: there propofol PK-PD drives MAP and the
    # twin can reproduce the response, so the deltas are identifiable. (Later
    # intra-op hypotension has surgical causes the propofol-only twin can't model.)
    fit_s = min(duration_s, 360.0)  # first 6 minutes
    n_fit = int(fit_s) + 1
    deltas, rmse = fit_deltas(patient, prop, ne, obs[:n_fit], duration=fit_s)
    return {
        "delta_ec50": round(deltas["EC50"], 3),
        "delta_ke0": round(deltas["ke0"], 3),
        "rmse": round(rmse, 2),
    }


@app.get("/api/model")
def model_card():
    """Trained predictor metadata for the UI model card."""
    return {"available": PREDICTOR is not None, "meta": PREDICTOR_META or {}}


@app.get("/api/cases")
def cases():
    from twin.data.case_api import demo_case_catalog
    return {"cases": demo_case_catalog()}


@app.get("/api/case/{caseid}")
def case(caseid: int):
    from twin.data.case_api import available_cases, load_case_bundle
    if caseid not in available_cases():
        raise HTTPException(status_code=404, detail="case not available")
    return load_case_bundle(caseid, model=PREDICTOR)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")
