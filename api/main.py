"""FastAPI backend for the perioperative digital-twin dashboard.

Exposes the mechanistic PK-PD engine as a JSON API and serves the static
clinical-instrument frontend. Stateless; every request is a fresh projection.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from twin.pkpd.engine import Patient, project_map, project_band, ioh_risk

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="Perioperative Digital Twin", version="1.0")


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


def _schedules(req: SimRequest, duration_s: float):
    prop = [(0.0, duration_s, req.propofol)] if req.propofol > 0 else []
    ne = ([(req.norepi_start_min * 60.0, duration_s, req.norepi)]
          if req.norepi > 0 else [])
    return prop, ne


@app.post("/api/simulate")
def simulate(req: SimRequest):
    duration_s = req.duration_min * 60.0
    patient = Patient(age=req.age, weight=req.weight, height=req.height,
                      sex=req.sex, map0=req.map0)
    prop, ne = _schedules(req, duration_s)
    deltas = ({"EC50": req.delta_ec50, "ke0": req.delta_ke0}
              if req.personalize else None)

    pop = project_map(patient, prop, ne, duration=duration_s)
    shown = project_map(patient, prop, ne, duration=duration_s, deltas=deltas)

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
    }
    if req.show_band:
        lo, hi, _ = project_band(patient, prop, ne, duration=duration_s, deltas=deltas)
        resp["band_lo"] = lo.round(3).tolist()
        resp["band_hi"] = hi.round(3).tolist()
    return resp


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")
