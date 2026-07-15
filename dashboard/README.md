# What-If Digital-Twin Demo

Interactive what-if simulator for the perioperative digital twin. A mechanistic
PK-PD engine (Eleveld-2018 propofol + Joachim-2024 norepinephrine) projects a
patient's mean arterial pressure (MAP) under chosen dosing, against the 65 mmHg
intraoperative-hypotension (IOH) threshold.

## Setup (once)
```bash
pip install -r requirements.txt
```
If `python`/`streamlit` are not on your PATH, use `python3 -m streamlit ...`.

## Run
```bash
python3 -m streamlit run dashboard/app.py
```
Opens at http://localhost:8501.

## Demo script (~2 min)

1. **Set a frail patient** (sidebar): Age **80**, Weight **55**, Sex **F**,
   Baseline MAP **85**. This is a patient with low propofol clearance.
2. **Induce.** Raise **Propofol** to **50 mg/min**. The MAP curve crosses the red
   65 mmHg line — "Min projected MAP" drops to ~61 mmHg and "Minutes < 65" climbs
   to ~9 min. *This is a predicted hypotensive event.*
3. **Rescue.** Set **Norepinephrine** to **12 µg/min** starting at **3 min**. MAP
   recovers above 65 (min ~75 mmHg, 0 min below). *The twin lets you rehearse the
   intervention before giving it.*
4. **Personalize.** Enable **"Enable personalized twin"** and set **δ EC50 = −0.40**.
   The orange (personalized) curve dips below the blue (population) one — this
   patient is more propofol-sensitive than the population average. **This δ is
   exactly what the SP4 calibration head will learn from the patient's own data;
   here we set it by hand to show the personalization interface.**

## What this demonstrates
- A working mechanistic twin of drug → MAP dynamics, on CPU, in real time.
- The what-if decision-support loop: try an intervention, see the projected MAP
  before acting. Decision support, **not** closed-loop control.
- The personalization hook (6 multiplicative δ perturbations) that couples this
  engine to the learned calibration head — the project's novel contribution.

## Scope / honesty notes
- Covariate scaling and the norepinephrine model are **documented reduced forms**
  of Eleveld 2018 / Joachim 2024 (see `twin/pkpd/covariates.py`,
  `twin/pkpd/norepi.py`) — adequate for a prototype, not the full published models.
- δ values are set manually in this prototype; the learned calibration head is SP4.
- MAP0 is an operator input here; in the real system it is estimated from the
  patient's first 5 minutes.

## Tests
```bash
python3 -m pytest tests/test_pkpd_*.py tests/test_dashboard_smoke.py -q
```
