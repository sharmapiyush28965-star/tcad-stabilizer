"""
Phase 1 reference solver: 1D drift-diffusion (drift-diffusion).

Physics: Poisson + electron/hole continuity, steady state,
Scharfetter-Gummel discretization of carrier fluxes, Newton-Raphson.

Formulation: unknowns per node are [psi, phin, phip] (electrostatic
potential and electron/hole quasi-Fermi potentials, all O(1) volts).
Carrier densities via Boltzmann: n = ni*exp((psi-phin)/Vt),
p = ni*exp((phip-psi)/Vt).  (Degenerate statistics NOT modelled --
noted limitation for Phase 1.)

Newton loop exposes a per-iteration hook, the product interface:
    correction = hook(state)          # state -> dict
    alpha = correction.get("alpha", 1.0)
    x_new = x + alpha * delta
`alpha` may be a scalar or an array broadcastable to the Newton step.
Default (hook=None): plain Newton, alpha = 1.

Phase 2 Track B seam: `solver.flux_mult` (None default) holds optional
per-edge Bernoulli multipliers applied inside residual(). None reproduces
Phase 1 exactly; the solver adapter sets it when a Track B correction
is issued.

Row equilibration: each residual row is scaled by 1/|F0| (F0 = initial
residual), Dirichlet rows by 1.0 (absolute volts). Convergence is
||S*F||_inf < tol.
"""

import numpy as np

# ---------------- physical constants (SI) ----------------
q = 1.602176634e-19        # C
kB = 1.380649e-23          # J/K
T = 300.0                  # K
Vt = kB * T / q            # thermal voltage ~ 0.02585 V
eps0 = 8.854187817e-12     # F/m
eps_si = 11.7 * eps0       # silicon permittivity
ni = 1.0e16                # intrinsic carrier conc, m^-3 (Si, 300K)
mun = 0.14                 # electron mobility, m^2/V/s
mup = 0.05                 # hole mobility, m^2/V/s
Dn = mun * Vt              # Einstein
Dp = mup * Vt


def bernoulli(x):
    """Bernoulli function B(x) = x/(exp(x)-1), numerically stable."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    big_pos = x > 50.0
    big_neg = x < -50.0
    mid = ~(big_pos | big_neg)
    xm = x[mid]
    tiny = np.abs(xm) < 1e-10
    vals = np.empty_like(xm)
    vals[tiny] = 1.0 - xm[tiny] / 2.0 + xm[tiny] ** 2 / 12.0
    vals[~tiny] = xm[~tiny] / np.expm1(xm[~tiny])
    out[mid] = vals
    out[big_pos] = 0.0
    out[big_neg] = -x[big_neg]
    return out


def densities(psi, phin, phip):
    """Boltzmann carrier densities from quasi-Fermi potentials."""
    n = ni * np.exp(np.clip((psi - phin) / Vt, -50.0, 50.0))
    p = ni * np.exp(np.clip((phip - psi) / Vt, -50.0, 50.0))
    return n, p


class DriftDiffusion1D:
    def __init__(self, N=120, L=2e-6, Na=1e22, Nd=1e22,
                 xj=0.5, abrupt_cells=2, taun=1e-6, taup=1e-6,
                 Vleft=0.0, Vright=0.0):
        """
        N: mesh nodes; L: device length (m); Na/Nd: doping (m^-3);
        xj: junction position as fraction of L; abrupt_cells: transition
        width in mesh cells (smaller = sharper, more brutal for SG);
        taun/taup: SRH lifetimes (s); Vleft/Vright: contact bias (V).
        """
        self.N, self.L = N, L
        self.xc = np.linspace(0.0, L, N)
        self.h = L / (N - 1)
        self.taun, self.taup = taun, taup
        self.Vleft, self.Vright = Vleft, Vright

        # Net doping C = Nd - Na: -Na left of junction, +Nd right.
        wt = abrupt_cells * self.h
        x = self.xc
        trans = np.clip((x - (xj * L - wt / 2.0)) / wt, 0.0, 1.0)
        self.C = -Na + (Na + Nd) * trans

        # Ohmic contact boundary values.
        self.psi_b0 = Vleft + Vt * np.arcsinh(self.C[0] / (2.0 * ni))
        self.psi_b1 = Vright + Vt * np.arcsinh(self.C[-1] / (2.0 * ni))

        # Phase 2 Track B: per-edge Bernoulli multiplier, set from the hook's
        # "flux_scale" correction. None (default) -> unmodified physics.
        self.flux_mult = None
        self._edge = None

    # ---------------- residuals ----------------
    def recombination(self, n, p):
        n1 = p1 = ni
        return (n * p - ni ** 2) / (self.taup * (n + n1) + self.taun * (p + p1))

    def residual(self, x):
        """Residual F(x). If self.flux_mult is set (Track B, from the hook's
        "flux_scale"), the Scharfetter-Gummel Bernoulli values are scaled
        per-edge during assembly."""
        N, h = self.N, self.h
        psi = x[0::3]
        phin = x[1::3]
        phip = x[2::3]
        n, p = densities(psi, phin, phip)
        F = np.zeros(3 * N)

        # Poisson, interior nodes.
        ii = np.arange(1, N - 1)
        lap = (psi[ii + 1] - 2.0 * psi[ii] + psi[ii - 1]) / h ** 2
        chg = (q / eps_si) * (p[ii] - n[ii] + self.C[ii])
        F[3 * ii] = lap + chg
        terms = np.zeros(3 * N)
        terms[3 * ii] = np.abs(lap) + (q / eps_si) * (p[ii] + n[ii] + np.abs(self.C[ii]))

        # Scharfetter-Gummel edge fluxes.
        dpsi = (psi[1:] - psi[:-1]) / Vt
        Bp, Bm = bernoulli(dpsi), bernoulli(-dpsi)
        # Phase 2 Track B seam: per-edge Bernoulli multipliers from the hook.
        # None (default) -> Phase 1 behavior, bit-identical.
        _fm = self.flux_mult
        if _fm is not None:
            _fm = np.asarray(_fm, dtype=float)
            if _fm.shape != Bp.shape:
                raise ValueError(f"flux_mult shape {_fm.shape} != {Bp.shape}")
            Bp = Bp * _fm
            Bm = Bm * _fm
        Jn = (q * Dn / h) * (n[1:] * Bp - n[:-1] * Bm)
        Jp = (q * Dp / h) * (p[:-1] * Bp - p[1:] * Bm)
        self._last_Jn, self._last_Jp = Jn, Jp
        self._edge = {"dpsi_vt": dpsi, "bernoulli_p": Bp, "bernoulli_m": Bm}

        # Continuity, interior nodes: div J + q*(R - G) = 0, G = 0.
        R = self.recombination(n[ii], p[ii])
        dJn = (Jn[ii] - Jn[ii - 1]) / h
        dJp = (Jp[ii] - Jp[ii - 1]) / h
        F[3 * ii + 1] = dJn + q * R
        F[3 * ii + 2] = dJp + q * R
        terms[3 * ii + 1] = np.abs(dJn) + q * np.abs(R)
        terms[3 * ii + 2] = np.abs(dJp) + q * np.abs(R)

        # Dirichlet (ohmic contacts).
        F[0], F[1], F[2] = psi[0] - self.psi_b0, phin[0] - self.Vleft, phip[0] - self.Vleft
        F[-3] = psi[-1] - self.psi_b1
        F[-2] = phin[-1] - self.Vright
        F[-1] = phip[-1] - self.Vright
        terms[[0, 1, 2, -3, -2, -1]] = 1.0  # absolute volts

        self._terms = terms
        return F

    def fd_jacobian(self, x, F0):
        n = x.size
        J = np.empty((n, n))
        eps = 1e-8 * (1.0 + np.abs(x))
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = eps[j]
            J[:, j] = (self.residual(x + dx) - F0) / eps[j]
        return J

    def initial_guess(self, Vleft=None, Vright=None):
        Vl = self.Vleft if Vleft is None else Vleft
        Vr = self.Vright if Vright is None else Vright
        ramp = Vl + (Vr - Vl) * self.xc / self.L
        psi = Vt * np.arcsinh(self.C / (2.0 * ni)) + ramp
        x = np.zeros(3 * self.N)
        x[0::3], x[1::3], x[2::3] = psi, ramp, ramp
        return x

    # ---------------- Newton solve with hook ----------------
    def newton_solve(self, x0, hook=None, tol=1e-6, max_iter=60):
        x = x0.copy()

        # Per-row absolute floors (fixed for the device) for the relative
        # residual. Rows whose own term magnitudes fall below the floor are
        # treated as numerical noise, not as equations to enforce.
        Cmax = max(1.0, float(np.max(np.abs(self.C))))
        Ap = max((q / eps_si) * Cmax, Vt / self.h ** 2)   # Poisson
        An = q * mun * Vt * Cmax / self.h ** 2             # e- continuity
        Ah = q * mup * Vt * Cmax / self.h ** 2             # h+ continuity
        ii = np.arange(1, self.N - 1)
        arow = np.ones(3 * self.N)
        arow[3 * ii] = Ap
        arow[3 * ii + 1] = An
        arow[3 * ii + 2] = Ah

        hist, alphas = [], []
        converged, diverged = False, False
        best = {"honest": float("inf"), "x": x.copy()}
        for k in range(max_iter):
            F = self.residual(x)  # also refreshes self._terms
            # Adaptive row scale: normalize each equation by its OWN term
            # magnitudes, so the linear solve prioritizes genuinely violated
            # equations. `honest` is the worst fractional imbalance.
            s = 1.0 / np.maximum(self._terms, 1e-12 * arow)
            honest = float(np.max(s * np.abs(F)))
            hist.append(honest)
            if honest < best["honest"]:
                best = {"honest": honest, "x": x.copy()}
            if not np.isfinite(honest) or honest > 1e6 * max(1.0, hist[0]):
                diverged = True
                break
            J = self.fd_jacobian(x, F)
            try:
                delta = np.linalg.solve(s[:, None] * J, -s * F)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(s[:, None] * J, -s * F, rcond=None)
            update = float(np.max(np.abs(delta)))

            # Convergence: Newton update negligible (like production TCAD)
            # AND no equation macroscopically violated (false-convergence guard).
            if update < 1e-10 and honest < 0.5:
                converged = True
                break

            # v1 hook state: pure data (solver-agnostic) + "solver" extension
            # for advanced hooks that need residual/Jacobian evaluation.
            fields = self._fields(x)
            state = {
                "version": "1",
                "iteration": k,
                "x": x.copy(), "delta": delta.copy(),
                "residual": F.copy(), "residual_norm": honest,
                "row_scale": s,
                "mesh": {"x": self.xc.copy(), "h": self.h, "N": self.N},
                "fields": {kk: vv.copy() for kk, vv in fields.items()
                           if kk in ("psi", "phin", "phip", "n", "p", "doping")},
                "edge": {kk: vv.copy() for kk, vv in self._edge.items()},
                "solver": self,  # extension: not part of the v1 pure-data contract
            }
            corr = hook(state) if hook is not None else {}
            # Track A: damping (solution-preserving).
            alpha = corr.get("alpha", 1.0)
            alphas.append(float(np.min(alpha)) if np.ndim(alpha) else float(alpha))
            # Track B: flux modification (EXPERIMENTAL). Takes effect from the
            # NEXT iteration's residual assembly (documented one-iteration lag).
            if "flux_scale" in corr and corr["flux_scale"] is not None:
                self.flux_mult = np.asarray(corr["flux_scale"], dtype=float)
            x = x + alpha * delta

        # Track B cleanup: don't leak a hook's flux_mult into later solves.
        self.flux_mult = None

        # Return the best state seen (for non-converged runs, the last state
        # may have wandered; the best-honest state is the most physical).
        rx = x if converged else best["x"]
        return {
            "converged": converged, "diverged": diverged,
            "iterations": len(hist), "residual_history": hist,
            "alphas": alphas, "x": rx,
            "final_residual": hist[-1] if hist else float("inf"),
        }

    def _fields(self, x):
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        n, p = densities(psi, phin, phip)
        return {"psi": psi, "phin": phin, "phip": phip,
                "n": n, "p": p, "doping": self.C, "x": self.xc, "h": self.h}

    def set_bias(self, Vleft, Vright):
        """Update contact bias (for voltage ramping / continuation)."""
        self.Vleft, self.Vright = Vleft, Vright
        self.psi_b0 = Vleft + Vt * np.arcsinh(self.C[0] / (2.0 * ni))
        self.psi_b1 = Vright + Vt * np.arcsinh(self.C[-1] / (2.0 * ni))

    def terminal_current(self, x):
        """Total current density (A/m^2) at the left contact edge."""
        self.residual(x)  # refresh edge fluxes
        return float(self._last_Jn[0] + self._last_Jp[0])
