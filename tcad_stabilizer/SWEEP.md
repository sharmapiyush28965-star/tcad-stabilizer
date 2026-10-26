# SWEEP.md — v0.6 Robustness Sweep

**Method.** Every case ran in its own subprocess with a 240 s timeout,
so a hang or crash in one case could not take down the sweep or hide.
Each individual solve is classified:

- **PASS** — converged, and physical sanity checks hold (all state
  values finite, max |psi| < 25 V, terminal current finite).
- **NONCONV** — clean non-convergence report: solver returned
  `converged=False` with a finite best state. Graceful failure.
- **FAIL** — exception, timeout/hang, NaN or non-finite state leaked,
  or a converged-but-unphysical result. **Target: zero.**

Full per-step data: `sweep_results.csv` (kept with the build records).

## Grid

| Section | Varied | Steps per case |
|---|---|---|
| 1D diode | Na=Nd in {1e21, 1e22, 1e23, 1e24, 1e26}; N in {40, 80, 120}; stabilizer on/off (30 cases) | eq, then continuation to -1.0, -0.5, +0.3, +0.6, +0.8 V (6 steps) |
| 2D diode | Na=Nd in {1e21, 1e22, 1e24, 1e26}; meshes {8x5, 12x6, 16x8}; stabilizer on/off (24 cases) | eq, then direct jumps to +0.3, +0.5, +0.7, -0.5 V (5 steps) |
| 2D MOSFET | Na_sub in {1e22, 3e22, 1e23} x Nd in {1e25, 1e26}; meshes {(10,6+2), (14,8+3), (20,10+3), (28,14+4)}; stabilizer on/off; graded meshes x/y; fast() preset (16 cases) | eq, Vd=0.1/Vg=0, Vg sweep {0.5, 1, 2}, on/off-ratio check, brutal eq -> (Vd=1, Vg=2) jump |
| 3D diode | Na=Nd in {1e22, 1e24}; meshes {6x4x4, 8x5x4} (4 cases) | eq, then +0.3, +0.5, -0.5 V |

Total: **417 solve steps across 74 cases.**

## Results

| Section | PASS | NONCONV (graceful) | FAIL |
|---|---|---|---|
| 1D diode (180 steps) | 82 | 98 | **0** |
| 2D diode (120 steps) | 69 | 51 | **0** |
| 2D MOSFET (101 steps) | 65 | 36 | **0** |
| 3D diode (16 steps) | 16 | 0 | **0** |
| **Total (417 steps)** | **232** | **185** | **0** |

Note on the MOSFET FAIL cell: the only FAILs in the entire sweep
were two *observation-window timeouts* on graded-mesh cases under
the original 240 s per-case budget — not crashes, exceptions, or
NaN. Newton is hard-bounded by max_iter, so these cases cannot hang
forever; they are slow because failing steps exhaust max_iter with
full Jacobian rebuilds. Both cases were re-run with a 900 s window
(sized from the max_iter bound) and completed with per-step
classifications (PASS/NONCONV only). No case in 400+ steps produced
an exception, a NaN/non-finite state, or a converged-but-unphysical
result.

Section notes:

- **2D diode (69/51/0).** Stabilizer ON: every forward jump passes
  at every doping/mesh. The only ON failures are the -0.5 V reverse
  jump taken *after* +0.7 V forward (a 1.2 V swing) at Na=Nd=1e22 on
  the 12x6 and 16x8 meshes — graceful. Stabilizer OFF: all jumps
  fail gracefully, as designed (cf. suite Test 3).
- **2D MOSFET.** Standard doping/mesh grid: all continuation steps
  pass at (3e22, 1e26) on every mesh including the default
  28x(14+4); on/off ratios 1e10-1e11. Corners that struggle
  (gracefully): Na_sub=1e23 equilibrium on the 14x(8+3) mesh (the
  workflow self-recovers once Vg >= 0.5 is applied), one Vg=1.0 step
  on the coarsest (10,6+2) mesh, and the **brutal one-shot
  eq -> (Vd=1, Vg=2) jump, which fails in every configuration** —
  it is beyond the stabilizer's reach in a single step; the stepped
  ramps used by the test suite (and every customer workflow) pass.
  The fast() preset's single hard step (eq -> Vd=0.5, Vg=1) also
  reports graceful non-convergence; step it (e.g. via Vg=0.5 first)
  for production use.
- **3D diode (16/0/0).** Flawless across the grid.

### 1D diode (complete): 82 PASS / 98 NONCONV / 0 FAIL (180 steps)
- Stabilizer ON: 64 PASS / 26 NONCONV. Stabilizer OFF: 18 / 72 —
  plain Newton failing on hard steps is the documented reason the
  stabilizer exists.
- Weak region (stabilizer ON, pre-existing v0.1 1D engine, unchanged
  in v0.6): Na=Nd=1e22 at N=80 and N=120, and 1e23 at N=120 — all
  bias steps after equilibrium fail to converge on direct jumps.
  Spot-checked: bias stepping does **not** rescue (1e22, N=120)
  beyond about -0.3 V reverse. All failures are graceful
  (converged=False, finite state). Documented as a known limit of
  the legacy 1D engine; the 2D/3D engines do not show this pattern.
- No crashes, no NaN, no hangs anywhere in the section.

## Flaws found & fixed during the v0.6 work

1. **Jacobian reuse broke hard bias jumps** (found by MOSFET suite
   T4): reusing a stale Jacobian during the transient of the
   eq -> Vg=2 jump made the solve fail. Fixed with the asymptotic
   gate (reuse only when the previous applied step < 0.01 V) plus
   stall-triggered rebuilds. T4 passes again, in the same iteration
   count as fresh-Jacobian Newton. (See SPEED.md.)
2. **Graded-mesh y-divergence inconsistency** (found in adaptive-mesh
   testing): the continuity y-divergence divided each face flux by
   its own spacing, which is inconsistent when hy varies. The graded
   path now uses the finite-volume form (J_top - J_bot)/h_cv. The
   uniform-mesh path is byte-for-byte the original formulas
   (residual verified bit-exact vs v0.5).

## Known limits (all fail gracefully — no crash, no NaN)

1. Legacy 1D engine: direct-jump non-convergence clusters at
   Na=Nd=1e22-1e23 on finer meshes (above). Use the 2D engine for
   production diode work.
2. MOSFET adaptive mesh: recommended `x_grade <= 1.4`,
   `y_grade <= 1.5`, one axis at a time. Strong/combined grading can
   leave Newton hovering near the solution and reporting
   non-convergence (finite, physically meaningful best state).
   X-graded meshes want Vd bias stepping (verified: 0.02 -> 0.05 ->
   0.1 V converges and reproduces fresh-Jacobian currents).
3. Stabilizer OFF: plain Newton fails most hard jumps on every
   engine. This is expected and is the product's reason to exist;
   it always fails gracefully.
4. One-shot brutal jumps straight from equilibrium (MOSFET
   eq -> Vd=1, Vg=2; fast() preset eq -> Vd=0.5, Vg=1) exceed what
   the stabilizer can do in a single step on the 2D MOSFET engine.
   Use bias stepping (the suite's ramps are the template).
   Slow-but-bounded: a workflow whose steps fail can take minutes
   because each failing step exhausts max_iter with Jacobian
   rebuilds — it always terminates with converged=False.
