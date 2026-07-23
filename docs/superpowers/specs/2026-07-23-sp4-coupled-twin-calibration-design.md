# SP4 — Learned Calibration Head + Coupled Digital Twin (Design)

**Date:** 2026-07-23
**Status:** Approved design, ready for implementation planning
**Depends on:** SP1 (data foundation, IOH windows/labels), SP3 (numpy PK-PD engine)
**Blocks / hands off to:** SP2 (waveform encoder swaps into `E`)

## 1. Purpose

SP4 is the project's **novel contribution**: the *integration* of a data-driven IOH
predictor and a mechanistic PK-PD engine through a **shared latent state** and a
**learned calibration head** that personalizes the twin online.

A shared encoder `E` maps a per-window feature vector `x_t` to a latent `z_t`. Two
heads read `z_t`:

- **Prediction head `P`** — emits the IOH logit (the SP1 task).
- **Calibration head `C`** — emits 6 multiplicative perturbations
  `δ_t = (V1, V2, V3, ke0, EC50, γ)` that personalize the PK-PD twin via
  `θ = θ_pop · exp(δ)` (the interface already defined in
  `twin/pkpd/calibration.py::apply_deltas`).

`z_t` feeding **both** heads is the coupling — the load-bearing "shared latent state
`z_i(t)`" claim. Personalization is **online / time-varying**: the same time-agnostic
head, called on a rolling window, yields `δ_t` over time; called once on an early
window, it yields a static per-case personalization (the first validation milestone).

Decision support, not closed-loop control.

## 2. Scope decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Training approach | **Hybrid**: distill from teacher, then fine-tune end-to-end | Distillation gives a working head fast (checkpoint-safe); fine-tune lands the novel end-to-end result; distilled init de-risks identifiability. |
| δ dimensionality | **All 6**, with an identifiability prior `λ·‖δ‖²` + per-δ monitoring | User choice; prior keeps unidentifiable δ (V3, γ) near 0 unless data moves them — the viva defense. |
| Temporal scope | **Online / time-varying `δ(t)`**; static is the nested first milestone | Matches the stated project vision ("personalized online"). Time-agnostic head nests the static case. |
| Compute | **GPU-targeted** (Kaggle primary, Colab fallback), device-agnostic CPU fallback | User requirement. Same code runs laptop-CPU → MPS → CUDA via `pick_device()`. |
| Encoder now | **Tabular** trunk (exists today), behind a swappable interface | SP4 runs before SP2; SP2 later substitutes the CNN-GRU waveform encoder into `E` unchanged. |

## 3. Data prerequisite (why Task 1 exists)

The current cache (`data_cache/*.parquet`, 4,701 cases) contains **only Solar8000
vital signs** — `NUMERIC_TRACKS` in `twin/config.py` requested no drug tracks. Both
SP4 stages need the **actually-administered drug schedule**: the teacher and the
reconstruction loss both project MAP *given* propofol/vasopressor input.

Therefore SP4 requires a re-pull that adds:

- `Orchestra/PPF20_RATE` — propofol infusion (primary drive of the twin).
- `Orchestra/PHEN_RATE`, `Orchestra/NEPI_RATE` — vasopressors (norepi/phenylephrine rise term).
- Prefer continuous `Solar8000/ART_MBP` over intermittent `NIBP_MBP` as the
  reconstruction target.

Not all VitalDB cases have Orchestra pump data. **The SP4 cohort = cases with
propofol infusion + continuous ART.** Task 1 measures this coverage first; that
number sizes SP4. The re-pull is scoped to the SP4 cohort only. DUA already accepted.

## 4. Architecture

```
window features x_t ──► Encoder E ──► z_t  (shared latent, ~32-d)
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                              ▼
              Prediction head P              Calibration head C
                 → IOH logit                    → δ_t (6)
                 (SP1 task)                         │
                                                    ▼
                                differentiable PK-PD twin(δ_t, drugs)
                                                    │
                                                    ▼
                                              MAP̂ trajectory
```

- **Encoder `E`** — a module mapping window features → `z_t`. Today wraps the tabular
  MLP trunk (reuses the `deepnet.py` pattern). **Swappable**: SP2 replaces its body
  with the CNN-GRU waveform stack; the `features → z` interface is fixed and strict.
- **Prediction head `P`** — linear/MLP on `z_t` → IOH logit; imbalance-aware BCE.
- **Calibration head `C`** — MLP on `z_t` → 6 δ (optionally bounded via `tanh·bound`).
- **Differentiable twin** — see §5; consumes `δ_t` + administered drug schedule → MAP̂.

## 5. Differentiable PK-PD twin

`twin/pkpd/torch_engine.py`: a PyTorch reimplementation of the numpy engine
(`twin/pkpd/engine.py`) — propofol 3-compartment PK, effect-site first-order
equilibration (linear recurrence), sigmoid Emax PD, norepinephrine MAP-rise, and
`combine_map`. All operations differentiable, so gradients flow `δ → θ → MAP̂`. Runs
on GPU; batched over cases and over a δ-grid (used by the teacher).

**Correctness gates (tests):**
- At `δ = 0`, `torch_engine` matches numpy `project_map` within tight tolerance.
- Finite-difference gradient check of `∂MAP̂/∂δ`.

## 6. Training (GPU-targeted Hybrid)

Built on the differentiable engine up front so both stages share it and run on GPU.

**Stage 1 — Distillation (checkpoint-safe deliverable):**
- Extend the grid-fit teacher (`twin/pkpd/fitting.py` → `twin/pkpd/teacher.py`) to all
  6 δ with the identifiability prior. Projections **batched through `torch_engine` on
  GPU**, so the grid/optimizer search over δ is a batched tensor evaluation, not a CPU
  bottleneck.
- Train `C(z)` to regress the teacher's δ (plain amortized regression). Yields a
  working amortized calibration head.

**Stage 2 — End-to-end fine-tune (the novel result):**
- Joint loss `L = BCE(IOH) + λ_recon·‖MAP̂ − MAP_obs‖ + λ_prior·‖δ‖²`, backprop
  through the differentiable twin, starting from the distilled init.
- Online: temporal smoothness `‖δ_t − δ_{t-1}‖` optional regularizer on rolling `δ(t)`.

All training uses `pick_device()` (CUDA → MPS → CPU); identical code on laptop and
Kaggle/Colab.

## 7. Evaluation / success criteria

`twin/eval/personalization.py`:

- **Primary — personalization works:** held-out **per-patient MAP-trajectory RMSE**,
  personalized twin vs population twin (paired test). Personalized should win.
- **Identifiability:** which δ actually move (expect EC50, ke0 to move; V3, γ to stay
  ~0). Reported per-δ — the empirical viva defense.
- **Coupling payoff (secondary):** does the shared trunk improve IOH AUROC/AUPRC vs the
  SP1 predictor, in the unbiased + gray-zone regimes.
- **What-if sanity (Hypothesis 4):** monotonicity (more propofol → higher projected
  risk) and sensible intervention ranking on personalized twins.

## 8. GPU training harness

A Kaggle-primary / Colab-fallback notebook (extends `notebooks/train_deepnet_colab.ipynb`)
that:
- Uses the device-agnostic model/engine code unchanged.
- **Pulls cases from VitalDB inside the GPU session** rather than uploading (waveform
  data is large; matters most for SP2).
- Reused by both SP4 (Stage 1/2) and SP2 (waveform training).

## 9. File layout & testing

New / changed files (each TDD, matching the SP1 pattern):

| File | Purpose |
|---|---|
| `twin/config.py` | Add drug + ART tracks to `NUMERIC_TRACKS` |
| `twin/pkpd/torch_engine.py` | Differentiable PK-PD twin (GPU) |
| `twin/pkpd/teacher.py` | 6-δ teacher (batched through torch_engine) |
| `twin/models/coupled.py` | Encoder `E` + prediction head `P` + calibration head `C` |
| `scripts/train_calibration.py` | Stage 1 + Stage 2 training entrypoint |
| `twin/eval/personalization.py` | RMSE / identifiability / coupling / what-if eval |
| `notebooks/train_calibration_gpu.ipynb` | Kaggle/Colab GPU harness |

Tests: numpy-parity at δ=0, gradient finite-difference check, head output shapes,
distillation overfits a toy batch, Stage-2 reconstruction loss decreases, eval metrics
computed on a held-out split.

## 10. Hand-off to SP2 (next sub-project)

SP2 replaces the body of `Encoder E` with a CNN-GRU stack over 4 waveforms @100Hz,
trained on GPU via the Task 8 harness. Because the coupling and both heads only touch
`z`, SP2 is an **encoder substitution** plus GPU training — the calibration head,
differentiable twin, losses, and evaluation are unchanged.

## 11. Risks

- **Drug-track coverage (Task 1)** may shrink the usable cohort — measured first.
- **Identifiability** of 6 δ from MAP alone — mitigated by distilled init + prior +
  per-δ monitoring; report honestly.
- **Differentiable-engine numerical stability** — guarded by the numpy-parity and
  gradient-check gates before any training.
