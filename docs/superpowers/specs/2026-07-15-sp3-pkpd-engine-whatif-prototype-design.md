# SP3 — Mechanistic PK-PD Engine + What-If Dashboard (Monday Prototype)

**Date:** 2026-07-15
**Project:** AI Digital Twin for Personalized Intraoperative Risk Prediction and Drug Response Simulation (FYP)
**Sub-project:** SP3 of 5 (mechanistic engine) + a thin SP5 slice (what-if dashboard)
**Status:** Design approved; pending implementation plan
**Deadline:** Demo to evaluator Monday 2026-07-20

---

## 1. Context & Goal

The FYP couples a deep-learning IOH predictor (SP1/SP2) with a personalized mechanistic PK-PD engine (SP3), joined by a learned calibration head (SP4) and exposed via a what-if dashboard (SP5). SP2/SP4 need a GPU the developer does not have; SP1 is already built (CPU).

**This prototype builds the CPU-only pieces that make the "digital twin" concept tangible for a live demo:** the SP3 mechanistic engine, plus a minimal Streamlit what-if dashboard on top of it. Slides, written report, and the real VitalDB data run are **out of scope for this deliverable** (tracked separately).

**Success = a bulletproof live demo:** enter a patient, set propofol/norepinephrine dosing, and watch a projected MAP trajectory update live against the 65 mmHg hypotension line, with a population-vs-personalized toggle. The demo must run offline with zero external-data dependency.

## 2. Scope & Success Criteria

### Delivers
- `twin/pkpd/` — a tested, deterministic mechanistic engine:
  - Eleveld-2018 propofol three-compartment PK + effect-site compartment, with covariate scaling.
  - Sigmoid Emax propofol→MAP pharmacodynamics.
  - Joachim-2024 norepinephrine PK-PD (raises MAP).
  - Combined MAP projection from a baseline MAP plus drug effects.
  - Calibration hook: 6 multiplicative perturbations δ applied as `θ = θ_pop · exp(δ)`.
- `dashboard/app.py` — a Streamlit what-if app driving the engine interactively.
- Unit tests covering PK dynamics, PD saturation, dose-response monotonicity, and calibration behavior.

### Success = all true
1. Engine reproduces qualitatively correct dynamics: a propofol infusion lowers MAP with the expected ~5-min effect-site lag; a norepinephrine bolus raises it with a ~74 s time-to-peak; steady-state effect-site concentration tracks the sigmoid Emax curve.
2. Engine is deterministic and unit-tested; a fixed input yields a fixed trajectory.
3. Dashboard runs locally on the Intel MacBook (CPU), updates the MAP plot on control changes in well under a second, and shows population-vs-personalized curves diverging under non-zero δ.

### Explicitly deferred (not this deliverable)
- The learned calibration head itself (SP4) — the engine only exposes the δ *interface*; δ values are set manually/illustratively in the demo.
- The deep waveform predictor (SP2), the SP1 real-data run, slides, and the written report.
- Real-time coupling to the SP1 risk score (a documented stretch; see §7).

### Prototype-then-scale
Population parameter values come from the literature (see §4); exact covariate-scaling equations are implemented from Eleveld 2018 / Joachim 2024 as far as time allows, with any simplification documented in-code and in the spec, never hidden.

## 3. Architecture

Extends the existing `twin/` package; adds a sibling `dashboard/` for the UI. Follows the existing style: small single-purpose modules, constants in a config module, pure functions with tested interfaces.

```
twin/
  pkpd/
    __init__.py
    params.py        # population parameter dataclasses + literature values (propofol, norepi, PD)
    covariates.py    # Eleveld covariate scaling: (age,weight,height,sex) -> per-patient PK params
    propofol.py      # 3-compartment PK + effect-site ODE -> Ce(t); infusion/bolus schedule input
    norepi.py        # Joachim norepinephrine PK-PD -> MAP-increase effect(t)
    pd_model.py      # sigmoid Emax: Ce -> fractional MAP change; combine drug effects
    engine.py        # top-level: covariates + dose schedule + delta -> projected MAP trajectory
    calibration.py   # apply 6 multiplicative deltas: theta_pop -> theta_personalized
dashboard/
  app.py             # Streamlit: inputs -> engine -> live MAP plot + what-if compare
tests/
  test_pkpd_propofol.py
  test_pkpd_norepi.py
  test_pkpd_pd_model.py
  test_pkpd_engine.py
  test_pkpd_calibration.py
```

**Module contracts (each independently testable):**
- `params.py`: `PropofolParams`, `NorepiParams`, `PDParams` dataclasses holding population values; no logic.
- `covariates.py`: `scale_propofol(params, age, weight, height, sex) -> PropofolParams`. Pure.
- `propofol.py`: `simulate_effect_site(params, schedule, dt, duration) -> np.ndarray[Ce]`. Solves the 3-compartment + effect-site linear ODE.
- `norepi.py`: `simulate_map_effect(params, schedule, dt, duration) -> np.ndarray[delta_map_up]`.
- `pd_model.py`: `emax(Ce, pd_params) -> fractional_reduction`; `combine(map0, prop_reduction, norepi_increase) -> map_traj`.
- `calibration.py`: `apply_deltas(params, deltas: dict) -> params` where `deltas` has keys V1,V2,V3,ke0,EC50,gamma.
- `engine.py`: `project_map(patient, prop_schedule, norepi_schedule, deltas=None, map0=None, dt, duration) -> MapProjection` (trajectory + time axis + metadata). This is the single entry point the dashboard calls.

## 4. Parameters (population values, from the literature)

Sourced from the FYP attributes/parameters spec and the cited papers; encoded in `params.py`.

**Propofol PK (Eleveld 2018), representative 50 yo / 70 kg adult:**
V1 ≈ 6.3 L, V2 ≈ 25 L, V3 ≈ 270 L, CL ≈ 1.8 L/min, Q2 ≈ 1.7 L/min, Q3 ≈ 0.84 L/min, ke0 ≈ 0.146 /min.

**Propofol → MAP PD (sigmoid Emax):**
MAP0 = patient-specific (from first 5 min in real data; a slider default in pure-sim), Emax ≈ 30% max fractional reduction, EC50 ≈ 4 µg/mL, Hill γ ≈ 2.5.

**Norepinephrine PK-PD (Joachim 2024):**
tpeak ≈ 74 s, ΔMAPmax ≈ 24% rise, Emax sigmoid (direction inverted vs propofol).

**Calibration perturbations (δ, illustrative in this prototype):**
δ_V1, δ_V2, δ_V3, δ_ke0, δ_EC50, δ_γ — multiplicative, `θ = θ_pop · exp(δ)`.

## 5. Numerics

- Fixed-step integration of the linear compartment ODEs (`dt = 1 s`, default `duration = 15 min`), matching the 1 Hz cadence of SP1. A closed-form/matrix-exponential or simple explicit Euler/RK solution is acceptable given the linear, stiff-but-mild system; choice documented in-code. Determinism is required.
- Dose input is a **schedule**: a list of `(t_start, t_end, rate)` infusion segments plus optional `(t, amount)` boluses, converted to a per-timestep input vector.
- MAP trajectory = `MAP0 · (1 − propofol_fractional_reduction(t)) + norepi_increase(t)`, clipped to a physiological range.

## 6. Dashboard (Streamlit)

- **Inputs (sidebar):** age, weight, height, sex; propofol infusion rate (and optional bolus); norepinephrine infusion/bolus; simulation duration; population-vs-personalized toggle with 6 δ sliders (default 0 = population).
- **Main panel:** projected MAP trajectory vs time, with a horizontal 65 mmHg hypotension line and shaded < 65 region; a second "what-if B" trajectory overlaid for side-by-side intervention comparison; small readouts (min projected MAP, minutes below 65, propofol Ce at end).
- **Performance:** each recompute is a sub-second CPU call to `engine.project_map`; cache patient-covariate scaling with `st.cache_data`.
- **Robustness:** input ranges clamped; no external network calls; the app imports only `twin.pkpd` + plotting.

## 7. Stretch (only if core is done and stable)

Real-VitalDB overlay: load one cached case's actual `ART_MBP` (via the existing `twin/data/vitaldb_loader.py`) and its `Orchestra/PPF20_RATE` / norepinephrine tracks, feed the real dose schedule into the engine, and overlay projected-vs-actual MAP. This needs a small data pull and is gated behind the pure-sim core being finished; it must never block the guaranteed demo.

## 8. Testing & Reproducibility

- **Propofol PK:** zero dose → Ce stays 0; a step infusion → Ce rises monotonically toward a plateau; effect-site lags plasma (peak Ce after infusion stop is delayed).
- **PD:** Emax is monotone increasing in Ce, saturates at Emax, equals Emax/2 at EC50; γ controls steepness.
- **Norepi:** a bolus produces a transient MAP rise peaking near tpeak.
- **Calibration:** δ = 0 reproduces population params exactly; positive δ_V1 increases V1 by `exp(δ)`; personalized trajectory differs from population when any δ ≠ 0.
- **Engine:** fixed input → byte-identical trajectory (determinism); a heavier propofol infusion yields a lower minimum MAP (dose-response monotonicity end-to-end).
- All randomness (none expected) flows from the existing single seed in `twin/config.py`.

## 9. Open Items / Assumptions

- Covariate-scaling fidelity: full Eleveld allometric/maturation equations if time permits; otherwise a documented reduced form (weight allometry + age effect on CL/ke0). Any reduction is stated in-code and here — never presented as the full model.
- MAP0 in pure-sim mode is an operator input (default ~85 mmHg); in the stretch overlay it is estimated from the case's first 5 min.
- Norepinephrine parameterization from Joachim 2024 is approximated to the ΔMAPmax/tpeak summary values where the full model is unavailable; documented.
- `streamlit` and `plotly` (or matplotlib) added to `requirements.txt`.
```
