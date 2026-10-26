# 3D MOSFET Solver

Real 3D n-channel MOSFET: drift-diffusion in 3 dimensions with the
Newton-Raphson convergence stabilizer built in. This completes the
device line: 1D diode, 2D diode, 3D diode, 2D MOSFET, **3D MOSFET**.

## What it does

- Full 3D drift-diffusion: potential + electron/hole quasi-Fermi levels
  on a 3D mesh (x = source-to-drain, y = vertical, z = width)
- Gate oxide stack (SiO2, 8 nm) with Si/SiO2 interface physics
  (displacement-field continuity, no carrier flux into oxide,
  oxide solves Poisson only)
- Three real terminals: source, drain (ohmic contacts), gate (metal)
- Sparse Newton solver: colored finite-difference Jacobian
  (27 colors x 3 vars = 81 residual evals per step, independent of
  mesh size), assembled in scipy CSR and solved with `spsolve`.
  The colored Jacobian is verified **exact** against direct FD
  (0 bad columns out of 504 on the check mesh).

## Physical validation (8/8 tests pass, `test_mosfet_3d.py`)

| # | Test | Result |
|---|------|--------|
| 1 | Equilibrium converges | PASS (7 Newton iters) |
| 2 | Threshold is physical: Id/W jumps 1.7e-11 -> 4.3 A/m for Vg 0->1 V (long-channel analytic Vt = 0.97 V; the short-channel device turns on earlier, as expected) | PASS |
| 3 | Id-Vgs: 1.7e-11, 4.3, 37 A/m at Vg = 0, 1, 2 V | PASS |
| 4 | Id-Vds at Vg=2: 71, 220, 260 A/m at Vd = 0.2, 1, 2 V (linear -> saturation) | PASS |
| 5 | Subthreshold slope 80 mV/dec (ideal = 60) | PASS |
| 6 | On-state physical: max psi = 2.00 V, no NaN | PASS |
| 7 | **Matches the 2D MOSFET per unit width to ~1e-10 relative** at both checked bias points | PASS |
| 8 | Stabilizer rescues a brutal one-step gate jump (eq -> Vd=0.1, Vg=2): plain Newton diverges in 4 iters, stabilized Newton converges in 17 | PASS |

Test mesh: Nx=14, Ny_si=8, Ny_ox=3, Nz=4 (616 nodes, 1848 unknowns).
This is the same x/y mesh as the proven 2D MOSFET suite — a coarser
y mesh (Ny_si=6, Ny_ox=2) was verified to stall Newton at strong
inversion in *both* the 2D and 3D solvers, so the proven mesh is used.
Full suite runtime is ~35 min (pure-Python residual evaluation).

## Files

- `mosfet_3d.py` — the solver (`MOSFET3D` class)
- `test_mosfet_3d.py` — the 8-test validation suite
- `plot_mosfet_3d.py` — mid-width x-y cross-section figure
  (`mosfet_3d_xsec.png`: potential at equilibrium and on-state,
  electron density showing the inversion channel, transfer curve)

## Usage

```python
from tcad_stabilizer import MOSFET3D, make_stabilizer

m = MOSFET3D(Nx=14, Ny_si=8, Ny_ox=3, Nz=4)   # W = Lz = 1 um
hook = make_stabilizer("classical")

m.set_bias(Vs=0.0, Vd=0.0, Vg=0.0)
x = m.newton_solve(m.initial_guess(), hook=hook)["x"]   # equilibrium

# Continuation to the on-state (standard TCAD practice)
for Vd, Vg in [(0.1, 0.0), (0.1, 1.0), (0.1, 2.0), (0.5, 2.0), (1.0, 2.0)]:
    m.set_bias(0.0, Vd, Vg)
    x = m.newton_solve(x, hook=hook)["x"]

Id = m.drain_current(x)              # total current (A)
Idw = m.drain_current_per_width(x)   # A/m, comparable to MOSFET2D
psi_xy = m.cross_section(x)          # (Ny, Nx) potential at mid-width
```

## Honest limitations

- n-channel only (no pMOSFET, no BJT); Boltzmann statistics;
  steady-state only; uniform structured mesh.
- Bias points are reached by continuation (small sequential steps) —
  large *combined* Vd+Vg jumps can diverge even with the stabilizer;
  the gate-only jump of test 8 is the demonstrated rescue case.
- Vt is lower than the long-channel analytic value because of
  short-channel effects at Lch = 1.4 um on this mesh — physical
  behavior, also seen in the 2D solver.
- Pure-Python residual loops: ~5 s per Newton iteration on the test
  mesh. A vectorized/analytical Jacobian would be the next step for
  large production meshes.

## Product coverage

The package now spans: 1D diode, 2D diode, 3D diode, 2D MOSFET,
3D MOSFET, plus the convergence stabilizer (classical + ML-experimental).
This is the full device set for an industry-standard drift-diffusion
TCAD core.
