# SP4 — Coupled Twin: Results

## Cohort and split

- SP4 cohort: **840** cases (propofol infusion + continuous arterial MAP)
- Train **624** / held-out test **138**, case-level stratified split
- Teacher iterations per case: 100

## Held-out IOH prediction

| metric | value |
|---|---|
| AUROC | 0.825 |
| AUPRC | 0.177 |
| PPV @ alarm rate | 0.186 |
| ECE | 0.007 |
| Brier | 0.035 |
| prevalence | 0.039 |
| windows | 27982 |

## Held-out personalization (the SP4 claim)

MAP reconstruction using the calibration head's delta, on cases the model never saw, against the population twin (delta = 0).

| model | RMSE (mmHg) |
|---|---|
| population (delta=0) | 21.67 |
| personalized (head delta) | 20.92 |

- Cases improved: **75** of 127
- Change: **-3.5%** — personalization IMPROVES held-out reconstruction

## Delta identifiability

Spread of each personalization delta across the cohort; a near-zero spread means the data does not move that parameter.

| delta | std |
|---|---|
| V1 | 0.3767 |
| V2 | 0.4971 |
| V3 | 0.5075 |
| ke0 | 0.4186 |
| EC50 | 0.4985 |
| gamma | 0.3398 |

## Norepinephrine arm

- Vasopressor cases in the SP4 cohort: **34** of 38 candidates
- Population RMSE 22.34 -> personalized 18.56 mmHg
- Improved: 34 cases

> Case series, not a trained result: VitalDB provides too few vasopressor cases to fit a calibration head. Deltas here come from the per-case teacher fit through the differentiable twin.

## Caveat — delta saturation

41.0% of head-predicted delta components sit beyond |0.65| against a +/-0.7 bound. With six free deltas the fit is degenerate, so report reconstruction improvement, not recovery of physiological parameters.
