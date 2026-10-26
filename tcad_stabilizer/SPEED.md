# SPEED.md — v0.6 Performance Release

**What changed in v0.6 (no physics changes):**

1. **Sparse colored FD Jacobian** in the 2D solvers (`dd_solver_2d.py`,
   `mosfet_2d.py`), ported from the proven 3D diode approach. The 5-point
   stencil needs only 9 colors x 3 variables = **27 residual evaluations
   per Jacobian build, independent of mesh size** — the old dense FD
   needed 3*N evaluations (462 on the test MOSFET mesh, 1512 on the
   default mesh) plus a dense LU solve. The Jacobian is assembled into a
   scipy CSR matrix and factored once with `splu`.
   *Verified:* the colored Jacobian is **bit-exact** vs the old dense FD
   Jacobian (max diff 0.0), and the default-mesh residual is bit-exact vs
   v0.5. Same per-unknown FD step as before.
2. **Jacobian reuse (quasi-Newton)** — new `jac_reuse=3` parameter on
   `newton_solve` (both 2D solvers). The LU factorization is reused for
   up to `jac_reuse` iterations, rebuilt early if the residual stalls.
   Reuse is *gated*: it is only allowed once the previous applied step
   was < 0.01 V (asymptotic regime). This gate matters — naive reuse
   broke the hard Vg=2 jump in test T4 (stale Jacobian steps during the
   transient poison the trajectory); with the gate, T4 converges in the
   same 16 iterations as fresh-Jacobian Newton while easy solves still
   save builds (equilibrium: 6 builds / 9 iterations). Set
   `jac_reuse=0` to force a fresh Jacobian every iteration.
   The iteration matrix never changes the converged solution: the fixed
   point is F(x)=0 and the convergence test is unchanged.
3. **Adaptive mesh (MOSFET2D)** — new optional `x_grade` / `y_grade`
   constructor parameters (default 1.0 = uniform, bit-identical to v0.5).
   `x_grade > 1` clusters x points geometrically toward the two
   junctions; `y_grade > 1` clusters silicon y points toward the
   Si/SiO2 interface. The graded discretization uses finite-volume
   (control-width) divergence forms; the default path is untouched.
4. **Fast mode** — `MOSFET2D.fast()` (14 x (8+3), the mesh the full test
   suite is validated on) and `DDSolver2D.fast()` (12 x 6) classmethod
   presets: same physics and solver, coarser-but-valid meshes for quick
   exploration. Use the default constructors for production accuracy.
5. `setup.py` now declares the `scipy` dependency (the 3D solvers
   already needed it; v0.5 under-declared it as numpy-only).

## Benchmarks (same script, same machine, before vs after)

Machine note: timings were taken on a shared VM under background load;
absolute times are pessimistic, but before/after ran under the same
conditions, so ratios are representative. An idle customer machine
will be faster across the board.

| Case | Before (v0.5) | After (v0.6) | Speedup |
|---|---|---|---|
| 2D MOSFET 14x(8+3): equilibrium + Vd warm-up + Vgs sweep {0, 0.5, 1} | 355.4 s | 15.6 s | **23x** |
| 2D diode 16x8: equilibrium + brutal 0->0.5 V jump | 158.5 s | 8.4 s | **19x** |
| 2D MOSFET default mesh 28x(14+4): cost per Newton iteration | 106.2 s | 1.4 s | **74x** |

The default-mesh MOSFET went from ~20+ minutes for one equilibrium
solve to ~15-20 seconds. Full test suites on this (busy) VM: the 2D
MOSFET suite now completes in 71 s and the 2D diode suite in 26 s.

**Result identity (before vs after, same benchmark script):**
- Diode jump current: 2.6453606061663e-03 vs 2.6453606061509e-03 A/m
  (rel. diff 6e-12); jump converges in 15 iterations in both.
- MOSFET Id at Vg=1.0: 4.2749166623 vs 4.2749166621 A/m (rel. diff 5e-11).
- MOSFET Id at Vg=0.5 (subthreshold): rel. diff ~6e-6 — expected:
  subthreshold currents are exponentially sensitive to the last
  ~1e-7 V of the converged potential; every test-suite criterion
  (threshold, on/off ratio > 1e4, subthreshold slope) is unchanged.
- Off-state leakage (~1e-10 A/m, a cancellation-dominated quantity):
  agrees to < 3e-10 A/m absolute.

**Test suites (package code, v0.6):** 2D MOSFET 6/6 PASS, 2D diode
5/5 PASS. Robustness sweep: see SWEEP.md.

## Adaptive mesh — honest accuracy note

Grading redistributes resolution; it is not a free accuracy win for
every quantity. Measured drain current Id(Vg=1, Vd=0.1) at 260 nodes
vs a 504-node uniform reference (Id = 3.306 A/m):

| Mesh (260 nodes) | Id (A/m) | Error vs reference |
|---|---|---|
| uniform 20x(10+3) | 3.939 | +19% |
| y_grade=1.5 | 2.505 | -24% |
| x_grade=1.4 (bias-stepped) | 4.550 | +38% |

Drain current at this bias is strongly mesh-dependent in *all*
variants (the 14x(8+3) suite mesh gives 4.27 A/m). Use grading when
you need resolution in a specific region (interface fields, junction
profiles); validate the quantity you care about against a finer mesh,
as with any TCAD tool.

**Graded meshes want bias stepping.** On an x-graded mesh the direct
equilibrium -> Vd=0.1 step may not converge (the drain-junction
potential drop concentrates in the finest cells). Ramping Vd in
2-3 increments (0.02 -> 0.05 -> 0.1) converges cleanly and reproduces
the same currents as fresh-Jacobian solves (Id = 1.03e-04 / 4.55 /
42.4 A/m at Vg = 0.5 / 1 / 2). This is standard TCAD continuation
practice; the uniform mesh does not need it for these biases.

## Known limits (fail gracefully, never crash)

- **Strong/combined grading**: `y_grade = 1.8`, or combined
  `x_grade >= 1.4` with `y_grade = 1.5`, can leave Newton hovering
  near the solution (scaled residual ~0.2-0.5, updates above the
  strict 1e-10 gate) and reporting `converged=False`. The returned
  best state is finite and physically meaningful. This happens with
  the hook on or off and with fresh Jacobians every iteration — it is
  a property of the stiff discrete system, not of the speed code.
  Recommended: `x_grade <= 1.4`, `y_grade <= 1.5`, one axis at a time.
- Physics limits are unchanged from v0.5 (Boltzmann statistics,
  steady state, structured meshes; 2D solvers use FD Jacobians —
  now sparse/colored instead of dense).
