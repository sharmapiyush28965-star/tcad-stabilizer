# TCAD Stabilizer 2D — Simple Guide

## What is this?
A 2D chip simulator. You design a diode (2D), it tells you how the chip behaves.

## Quick start
```python
from tcad_stabilizer.dd_solver_2d import DDSolver2D
from tcad_stabilizer import make_stabilizer

# Make a 2D diode (16 x 8 grid)
s = DDSolver2D(Nx=16, Ny=8)

# Solve (finds the physics solution)
r = s.newton_solve(s.initial_guess(), hook=make_stabilizer("classical"))
print("Converged:", r["converged"])  # True = success

# Apply voltage and solve again
s.set_bias(0.5, 0)  # 0.5V on left contact
r = s.newton_solve(r["x"], hook=make_stabilizer("classical"))

# Get current
I = s.terminal_current(r["x"])
print(f"Current: {I} A/m")
```

## What the stabilizer does
Normal solver sometimes gets stuck (numbers blow up). The stabilizer watches
each step and gently reduces the step size when it detects trouble. The final
answer is unchanged — only the path to get there is smoother.

## Tests
Run `python test_2d_suite.py` in `phase2_2d/` — all 5 tests must pass:
1. 2D equilibrium converges from scratch
2. 2D matches 1D (for uniform problems)
3. Stabilizer rescues hard cases where plain solver fails
4. Solution is physical (no NaN, reasonable values)
5. IV curve has correct exponential shape
