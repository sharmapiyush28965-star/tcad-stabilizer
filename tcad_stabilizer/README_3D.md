# TCAD Stabilizer — 3D Diode Solver

Real 3D drift-diffusion simulation with Newton stabilization.

## Quick start

```python
from tcad_stabilizer.diode_3d import DDSolver3D
from tcad_stabilizer import make_stabilizer

# 3D diode: 12 x 8 x 6 mesh
s = DDSolver3D(Nx=12, Ny=8, Nz=6)
s.set_bias(0, 0)
r = s.newton_solve(s.initial_guess(), hook=make_stabilizer("classical"))
print("converged:", r["converged"], "iters:", r["iterations"])

# Forward bias with stabilizer
s.set_bias(0.5, 0)
r = s.newton_solve(r["x"], hook=make_stabilizer("classical"))
print("I =", s.terminal_current(r["x"]), "A")
```

## What's inside

- **Physics**: 3D Poisson + electron/hole continuity, steady state,
  Scharfetter–Gummel discretization (industry-standard method).
- **Mesh**: structured Nx×Ny×Nz grid. Dirichlet contacts on x-faces,
  zero-flux (Neumann) on y/z faces.
- **Jacobian**: sparse, assembled by colored finite differences
  (27 colors × 3 vars = 81 residual evals per Newton step, independent
  of mesh size), solved with `scipy.sparse.linalg.spsolve`.
- **Stabilizer**: the same hook interface as 1D/2D — `make_stabilizer("classical")`
  rescues Newton on brutal bias jumps where plain Newton stalls.
- **Cross-sections**: `s.cross_section(x, plane="xy"|"xz"|"yz")` returns a
  2D array of the potential for plotting.

## Tests

`test_3d.py` — 5 tests, all must PASS:
1. 3D equilibrium converges from scratch
2. 3D matches 2D for z-uniform problems (rel err < 1e-6)
3. 3D IV curve is exponential (matches ideal diode law)
4. Stabilizer rescues a hard 3D case (plain Newton fails)
5. Solution is physical (no NaN, reasonable values)

Run: `python3 test_3d.py`

## Performance notes

- Small mesh (8×5×4 = 160 nodes): ~10 s per equilibrium solve.
- The sparse Jacobian has <4% nonzero entries; assembly is the main
  cost (81 residual evaluations). For larger meshes, consider an
  analytical Jacobian or matrix-free Newton-Krylov.

## Limitations (honest)

- Diode only (no 3D MOSFET yet).
- Boltzmann statistics (non-degenerate).
- Dense-in-time? No — steady state only.
- Uniform structured mesh (no adaptive refinement).
