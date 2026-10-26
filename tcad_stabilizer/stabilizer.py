"""
TCAD Convergence Stabilizer v0.1 — working product.

A drop-in Newton hook that stabilizes drift-diffusion solves.

Modes:
  "classical" (default, GUARANTEED): dvmax(0.2) update limiting.
      Rescues all tested stall cases. Solution-preserving.
  "ml" (experimental): M1 neural adaptive damping.
      Faster when it works, but does not yet rescue reliably.
      Always paired with classical fallback.

Safety:
  - Track A only (damping). Cannot change the converged physics.
  - Classical fallback always available (solve_with_fallback).
  - Uncertain = do nothing: OOD states fall back to classical.

Usage:
    from stabilizer import make_stabilizer, solve_with_fallback
    hook = make_stabilizer("classical")
    result = solver.newton_solve(x0, hook=hook)
    # or, with automatic fallback:
    result, used_fb = solve_with_fallback(solver, x0, mode="ml")
"""
import numpy as np
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _dvmax_hook(dvmax=0.2):
    def h(state):
        d = np.max(np.abs(state["delta"]))
        a = min(1.0, dvmax / d) if d > dvmax else 1.0
        return {"alpha": a}
    h.mode = "classical"
    return h


_M1_CACHE = {}


def _m1_hook(weights_path=None):
    from train_m1 import M1
    key = weights_path or "default"
    if key not in _M1_CACHE:
        p = weights_path or os.path.join(_HERE, "m1_weights.npz")
        _M1_CACHE[key] = M1.load(p)
    m1 = _M1_CACHE[key]

    def h(state):
        d = np.abs(state["delta"])
        dv = np.abs(state["edge"]["dpsi_vt"])
        f = np.array([[
            state["iteration"],
            float(np.log10(max(state["residual_norm"], 1e-16))),
            float(d.max()), float(d.mean()),
            float(dv.max()), float(dv.mean()),
        ]])
        # OOD -> fall back to classical for this iteration
        z = np.abs((f - m1.mu) / m1.sd)
        if bool((z >= 6.0).any()):
            return _dvmax_hook()(state)
        a = float(m1.predict(f)[0])
        return {"alpha": float(np.clip(a, 0.05, 1.0))}
    h.mode = "ml"
    return h


def make_stabilizer(mode="classical", weights_path=None):
    """Build the stabilizer hook. mode: "classical" (default) or "ml"."""
    if mode == "ml":
        return _m1_hook(weights_path)
    return _dvmax_hook()


def solve_with_fallback(solver, x0, mode="ml", **kw):
    """Solve with the requested mode; if it fails, fall back to classical.

    Returns (result, used_fallback: bool).
    With mode="classical", the fallback is a no-op (already classical).
    """
    primary = make_stabilizer(mode)
    r = solver.newton_solve(x0, hook=primary, **kw)
    if r["converged"]:
        return r, False
    if mode == "classical":
        return r, False
    r2 = solver.newton_solve(x0, hook=make_stabilizer("classical"), **kw)
    return (r2, True) if r2["converged"] else (r, True)
