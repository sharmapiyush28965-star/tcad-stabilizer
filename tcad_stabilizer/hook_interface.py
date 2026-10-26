"""
Hook interface specification v1 — solver-agnostic contract.

A "solver" is anything that runs a Newton(-like) iteration and agrees to:
  1. build the `state` dict below (pure data: numpy arrays + scalars, no
     solver objects), once per iteration, after assembling residual/Jacobian
     and solving for the Newton step;
  2. call `correction = hook(state)`;
  3. apply the correction as specified below.

The interface is deliberately solver-agnostic: a DEVSIM driver, a commercial
tool wrapper, or this reference solver can all implement it. Versioned so
future extensions (new state fields / correction keys) don't break hooks.

--- state (dict, pure data) ---
  "version": "1"
  "iteration": int — Newton iteration count (0-based)
  "x": (3N,) float — current iterate, interleaved [psi, phin, phip] per node (V)
  "delta": (3N,) float — Newton step for this iteration (V)
  "residual": (3N,) float — F(x)
  "residual_norm": float — honest worst fractional imbalance (see Phase 1 memo)
  "row_scale": (3N,) float — per-row normalization used for the linear solve
  "mesh": {"x": (N,), "h": float, "N": int}
  "fields": {"psi","phin","phip","n","p","doping"} — (N,) arrays, SI units
  "edge": {"dpsi_vt": (N-1,),            # (psi[i+1]-psi[i])/Vt per edge
           "bernoulli_p": (N-1,),        # B(+dpsi/Vt)
           "bernoulli_m": (N-1,)}        # B(-dpsi/Vt)

--- correction (dict) ---
  Track A — damping (SOLUTION-PRESERVING, primary):
    "alpha": float or (3N,) array. Applied as x_new = x + alpha*delta.
             alpha=1 (or missing key) = plain Newton.
             Because it only scales the step, it cannot change the converged
             fixed point: F(x*)=0 stays F(x*)=0.

  Track B — flux modification (EXPERIMENTAL, changes the discrete solution):
    "flux_scale": (N-1,) float array. Per-edge multiplier applied to the
             Scharfetter-Gummel electron AND hole fluxes during residual
             assembly, starting from the NEXT iteration (one-iteration lag,
             documented). flux_scale=1 (or missing) = unmodified physics.
             Because it changes the discrete equations, the converged solution
             DIFFERS from the unmodified one. Requires separate validation.

  Forward compatibility: unknown keys are ignored; missing keys = no correction.
  A hook may return {} (no-op).
"""

VERSION = "1"

STATE_KEYS = ("version", "iteration", "x", "delta", "residual",
              "residual_norm", "row_scale", "mesh", "fields", "edge")

# Correction keys the solver understands. Anything else is ignored.
TRACK_A_KEYS = ("alpha",)
TRACK_B_KEYS = ("flux_scale",)


def validate_state(state):
    """Lightweight check that a state dict honors the v1 contract."""
    for k in STATE_KEYS:
        if k not in state:
            raise KeyError(f"hook state missing required key: {k!r}")
    if state["version"] != VERSION:
        raise ValueError(f"unsupported hook version: {state['version']!r}")
    return True


def no_op(state):
    """The default hook: plain Newton."""
    return {}
