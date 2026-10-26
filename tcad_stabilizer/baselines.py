"""
Classical damping baselines, implemented as Newton-hook functions.

Each hook has the product interface: hook(state) -> {"alpha": ...}.
`alpha` is a scalar or per-element array applied as x_new = x + alpha*delta.
These are the fallback policies the product must never perform worse than.
"""

import numpy as np


def hook_none(state):
    """Plain Newton (alpha = 1)."""
    return {}


def hook_fixed(alpha):
    """Constant damping."""
    def h(state):
        return {"alpha": float(alpha)}
    h.__name__ = f"fixed({alpha})"
    return h


def hook_armijo(c=1e-4, max_backtrack=12):
    """
    Backtracking line search on merit m = ||S*F||_2^2.
    For the exact Newton direction the directional derivative is -2*m,
    so the Armijo condition is m_new <= m * (1 - 2*c*alpha).
    """
    def h(state):
        solver = state["solver"]
        s = state["row_scale"]
        F, d = state["residual"], state["delta"]
        m = float(np.sum((s * F) ** 2))
        alpha = 1.0
        for _ in range(max_backtrack):
            Fn = solver.residual(state["x"] + alpha * d)
            mn = float(np.sum((s * Fn) ** 2))
            if np.isfinite(mn) and mn <= m * (1.0 - 2.0 * c * alpha):
                break
            alpha *= 0.5
        return {"alpha": alpha}
    h.__name__ = "armijo"
    return h


def hook_dvmax(dvmax=1.0):
    """
    Potential-update limiting (Atlas-style DVMAX): cap the largest
    Newton update component at `dvmax` volts by scaling the whole step.
    """
    def h(state):
        mx = float(np.max(np.abs(state["delta"])))
        alpha = min(1.0, dvmax / mx) if mx > dvmax else 1.0
        return {"alpha": alpha}
    h.__name__ = f"dvmax({dvmax})"
    return h
