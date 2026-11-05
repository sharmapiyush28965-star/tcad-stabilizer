# Fab-grade roadmap — tcad-stabilizer v0.7+

Target (his order, 2026-11-05): **semiconductor fabs**, not teachers/students.
A fab evaluates a TCAD tool one way first: "reproduce OUR measured data."
If it cannot match their measured IV, nothing else matters. That sets the
order below. Honest status: v0.6 is a correct, fast drift-diffusion
solver — it is NOT yet fab-grade. Each phase below is built + tested
before the next starts. Nothing is claimed until a test runs it.

## Phase 1 — TRUST (v0.7, in progress)
The fab can check us against their own silicon.
- Doping-profile import (CSV: position vs net doping) for 1D/2D diodes
  — real SIMS/process profiles instead of ideal step junctions.
  (`set_doping_profile` on the solvers; boundaries recomputed.)
- Measured-IV import (CSV: V, I) + least-squares calibration fit over
  (SRH lifetimes taun/taup, doping scale, series resistance Rs).
  Rs is applied at the terminal level (V_term = Vj + I*Rs) — no solver
  change. Fit quality reported as RMS log-error; a bad fit is reported
  as a bad fit, never hidden.
- Compact-model parameter extraction (pure post-processing):
  diode ideality n, Is, Rs; MOSFET Vt (constant-current + max-gm),
  subthreshold slope, DIBL, Ion/Ioff, gm_max. The numbers a fab
  modelling team actually consumes.
- Tests: synthetic measured data with KNOWN parameters must be
  recovered by the fit within tolerance; extraction checked against
  hand-computed values.

## Phase 2 — REAL-DEVICE PHYSICS (v0.8)
What real silicon has that ideal devices do not.
- Interface traps (Dit) + fixed oxide charge at Si/SiO2.
- Self-heating (coupled lattice-temperature equation) — power devices.
- Density-gradient quantum correction — nanoscale MOSFETs.
- Field-dependent mobility (Caughey-Thomas) — high-field accuracy.
Each item lands only with a validation test against a known limit
(e.g. quantum correction -> classical result as tox grows).

## Phase 3 — FAB DELIVERABLES (v0.9)
- Statistical variability: random-dopant Monte Carlo (Vt spread).
- TCAD-to-SPICE export: extracted parameters as a table/CSV for the
  modelling team + a validation report (sim vs measured, per device).
- Fab pricing + support SLA (his call): fab licence is NOT ₹60k/seat
  (Sentaurus charges $100k+/yr/licence). Price is set when Phase 1
  proves the fit works on real data.

## Explicitly NOT in scope
- Full process simulation (implant/diffusion/oxidation/etch) — a
  separate product. Fabs bring their own profiles (Phase 1 import).
- Reliability (HCI/BTI) degradation models — empirical, need fab
  calibration data we do not have. Revisit after a fab engagement.
- Replacing Sentaurus sign-off flows. We sell exploratory/R&D studies
  and fast concept checks first; sign-off is earned, not claimed.
