"""Fab-grade helpers (v0.7 Phase 1 — TRUST): measured-data import,
calibration fitting, and compact-model parameter extraction.

These are pure post-processing around the existing solvers — no physics
changes inside the solvers. A fab's first check of any TCAD tool is
"reproduce OUR measured data": import their IV + doping profile, fit the
model parameters, and report the fit error honestly. A bad fit is
reported as a bad fit, never hidden.

Units (SI throughout): V in volts, I in amps (or A/m^2 for diode
current density — see each function), position in metres, doping in
m^-3, series resistance in ohm*m^2 when current is a density.
"""

import csv

import numpy as np
from scipy.optimize import least_squares

from .dd_solver import DriftDiffusion1D, Vt
from .stabilizer import make_stabilizer

__all__ = [
    "load_iv_csv",
    "load_profile_csv",
    "diode_sweep",
    "calibrate_diode",
    "extract_diode",
    "extract_mosfet",
]


# ---------------- CSV import ----------------

def _find_column(fieldnames, *keys):
    """First header whose lowercase name contains any of keys."""
    low = [f.strip().lower() for f in fieldnames]
    for k in keys:
        for i, f in enumerate(low):
            if k in f:
                return i
    return None


def load_iv_csv(path):
    """Measured I-V data from CSV. Headers must name a voltage column
    (contains 'volt' or exactly 'v') and a current column (contains
    'curr', 'amp', or exactly 'i'). Returns (V, I) sorted by V, both
    float arrays, V in volts, I in amps. Rows with non-positive current
    are dropped (log-scale fitting needs I > 0)."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError("IV file needs a header row plus data rows")
    header = rows[0]
    iv = _find_column(header, "volt", "v")
    ii = _find_column(header, "curr", "amp", "i")
    if iv is None or ii is None:
        raise ValueError("IV file needs voltage and current columns")
    V, I = [], []
    for r in rows[1:]:
        try:
            v, i = float(r[iv]), float(r[ii])
        except (ValueError, IndexError):
            continue
        if i > 0:
            V.append(v)
            I.append(i)
    V = np.asarray(V)
    I = np.asarray(I)
    order = np.argsort(V)
    return V[order], I[order]


def load_profile_csv(path, x_unit="m", c_unit="m^-3"):
    """Doping profile from CSV: position vs net doping Nd-Na. Headers
    must name a position column ('x', 'position', 'depth') and a doping
    column ('doping', 'conc', 'net'). Units: x_unit 'm' or 'cm',
    c_unit 'm^-3' or 'cm^-3' (SIMS data is usually per cm). Returns
    (x, C) sorted by x, SI units (m, m^-3)."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError("profile file needs a header row plus data rows")
    header = rows[0]
    ix = _find_column(header, "x", "position", "depth")
    ic = _find_column(header, "doping", "conc", "net")
    if ix is None or ic is None:
        raise ValueError("profile file needs position and doping columns")
    x, c = [], []
    for r in rows[1:]:
        try:
            x.append(float(r[ix]))
            c.append(float(r[ic]))
        except (ValueError, IndexError):
            continue
    x = np.asarray(x)
    c = np.asarray(c)
    order = np.argsort(x)
    x, c = x[order], c[order]
    if x_unit == "cm":
        x = x * 1e-2
    elif x_unit != "m":
        raise ValueError("x_unit must be 'm' or 'cm'")
    if c_unit == "cm^-3":
        c = c * 1e6
    elif c_unit != "m^-3":
        raise ValueError("c_unit must be 'm^-3' or 'cm^-3'")
    return x, c


# ---------------- sweeps ----------------

def diode_sweep(solver, voltages, stabilizer=True):
    """Forward IV sweep on a diode solver (1D or 2D): equilibrium, then
    ramp up in small steps, warm-started, stabilizer hook on. Returns
    (V, J) with J the terminal current density (A/m^2). The solver's
    bias is left at the last voltage."""
    hook = make_stabilizer("classical") if stabilizer else None
    solver.set_bias(0, 0)
    r = solver.newton_solve(solver.initial_guess(), hook=hook, max_iter=30)
    x = r["x"]
    V, J = [], []
    for v in voltages:
        solver.set_bias(float(v), 0)
        r = solver.newton_solve(x, hook=hook, max_iter=30)
        x = r["x"]
        V.append(float(v))
        J.append(solver.terminal_current(x))
    return np.asarray(V), np.asarray(J)


# ---------------- calibration ----------------

def calibrate_diode(solver_factory, V_meas, I_meas,
                    p0=(5e-7, 1.0, 0.0), max_nfev=60,
                    separate_lifetimes=False):
    """Fit a diode model to measured forward IV.

    solver_factory(taun, taup, cscale) -> a fresh diode solver with
    those SRH lifetimes (s) and doping scale (C multiplied by cscale).
    V_meas/I_meas: measured forward voltages (V, > 0) and currents
    (same units as the solver's terminal current — A/m^2 for the
    built-in solvers).

    Fit parameters: an effective SRH lifetime tau (taun = taup = tau),
    doping scale cscale, and series resistance Rs (ohm*m^2 when
    current is a density). Rs is applied at the terminal:
    V_term = Vj + J*Rs. Residuals are in log10(I) — diode currents
    span decades, so a linear residual would let the largest current
    dominate the fit.

    Why one lifetime, not two: a single forward IV curve cannot tell
    taun from taup apart — in a symmetric diode they enter only as a
    combination (depletion recombination ~ 1/(taun+taup), diffusion
    symmetric in the two sides). Fitting both separately from one
    curve returns an arbitrary split that still fits the curve.
    Pass separate_lifetimes=True to fit them independently anyway
    (only meaningful with asymmetric devices or extra measurements);
    p0 is then (taun, taup, cscale, Rs).

    Returns a dict: fitted taun/taup/cscale/Rs, rms_log_err (dex —
    0.05 dex is ~12% average error), n_points, and the fitted (V, I)
    curve on the measured voltage grid.
    """
    V_meas = np.asarray(V_meas, dtype=float)
    I_meas = np.asarray(I_meas, dtype=float)
    if V_meas.size < 4 or np.any(V_meas <= 0) or np.any(I_meas <= 0):
        raise ValueError("need >= 4 measured points with V > 0, I > 0")
    logI_meas = np.log10(I_meas)

    # Junction-voltage grid WIDER than the measured terminal range, so
    # the residual interpolation never clamps at the ends (a clamped
    # end makes the fit blind to Rs — the failure mode that hides a
    # wrong series resistance).
    lo = max(0.02, V_meas.min() * 0.5)
    hi = V_meas.max() + 0.2
    n_grid = max(12, 2 * V_meas.size)
    Vj_grid = np.linspace(lo, hi, n_grid)

    def _sweep_logJ(taun, taup, cscale):
        solver = solver_factory(taun, taup, cscale)
        _, J = diode_sweep(solver, Vj_grid)
        if np.any(J <= 0):
            return None
        return np.log10(J)

    if separate_lifetimes:
        # [log10 taun, log10 taup, log10 cscale, Rs]
        def _unpack(p):
            return 10 ** p[0], 10 ** p[1], 10 ** p[2], p[3]
        p0v = np.array([np.log10(p0[0]), np.log10(p0[1]),
                        np.log10(p0[2]), p0[3]], dtype=float)
        bounds = ([-9.0, -9.0, -1.0, 0.0], [-3.0, -3.0, 1.0, 1e-3])
    else:
        # [log10 tau, log10 cscale, Rs]; taun = taup = tau.
        def _unpack(p):
            return 10 ** p[0], 10 ** p[0], 10 ** p[1], p[2]
        p0v = np.array([np.log10(p0[0]), np.log10(p0[1]),
                        p0[2]], dtype=float)
        bounds = ([-9.0, -1.0, 0.0], [-3.0, 1.0, 1e-3])

    def resid(p):
        taun, taup, cscale, Rs = _unpack(p)
        logJ = _sweep_logJ(taun, taup, cscale)
        if logJ is None:
            return np.full(V_meas.size, 10.0)
        # Invert the Rs drop on the measured side: the junction voltage
        # behind each measured point is Vj = V_term - I*Rs. Interpolating
        # the simulated log-current at those junction voltages gives a
        # residual with a real gradient in Rs.
        Vj_meas = np.clip(V_meas - I_meas * Rs, Vj_grid[0], Vj_grid[-1])
        return np.interp(Vj_meas, Vj_grid, logJ) - logI_meas

    sol = least_squares(resid, p0v, bounds=bounds, max_nfev=max_nfev)
    taun, taup, cscale, Rs = _unpack(sol.x)
    # Final fitted curve: the model's predicted TERMINAL IV
    # (V_term = Vj + J*Rs), on the measured voltage grid for the error.
    logJ = _sweep_logJ(taun, taup, cscale)
    J_grid = 10 ** logJ
    V_term = Vj_grid + J_grid * Rs
    order = np.argsort(V_term)
    I_fit = 10 ** np.interp(V_meas, V_term[order], logJ[order],
                            left=logJ[order][0], right=logJ[order][-1])
    rms = float(np.sqrt(np.mean((np.log10(I_fit) - logI_meas) ** 2)))
    out = {"taun": float(taun), "taup": float(taup),
           "cscale": float(cscale), "Rs": float(Rs),
           "rms_log_err": rms, "n_points": int(V_meas.size),
           "V_fit": V_meas, "I_fit": I_fit}
    if not separate_lifetimes:
        out["tau"] = float(taun)
    return out


# ---------------- extraction ----------------

def extract_diode(V, I):
    """Diode parameters from a forward IV curve: ideality factor n,
    saturation current Is, and series resistance Rs. n and Is come
    from a log-linear fit over the decade starting one decade above
    the minimum current (low-bias ideality); Rs from the slope of
    dV/d(ln I) vs I over the highest decade (derivative method).
    Currents must be > 0. Rs is in ohm*m^2 when I is a density."""
    # Low-bias ideality: fit one decade starting a decade ABOVE the
    # minimum current. The very bottom decade bends because of the
    # "-1" in I = Is*(exp(V/nVt) - 1) — fitting there corrupts the
    # slope. A decade up, I >> Is, the curve is a clean exponential.
    # This reports the LOW-bias ideality (recombination-dominated in
    # real devices); the high-bias region is used for Rs below.
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    if V.size < 5 or np.any(I <= 0):
        raise ValueError("need >= 5 points with I > 0")
    logI = np.log10(I)
    lo = (logI >= logI.min() + 1.0) & (logI <= logI.min() + 2.0)
    if lo.sum() < 2:
        lo = (logI >= logI.min() + 0.5) & (logI <= logI.min() + 1.5)
    if lo.sum() < 2:
        lo = np.argsort(logI)[:3]
    slope, intercept = np.polyfit(V[lo], logI[lo], 1)
    # d(log10 I)/dV = q / (n kT ln 10)  ->  n = 1 / (slope * Vt * ln 10)
    n = 1.0 / (slope * Vt * np.log(10))
    Is = 10 ** intercept
    # High decade: series resistance via the derivative method. At high
    # current V = n*Vt*ln(I/Is) + I*Rs, so dV/d(ln I) = n*Vt + Rs*I:
    # the slope of dV/d(ln I) vs I is Rs. This never uses the fitted
    # n/Is, so their small errors cannot fake a series resistance
    # (the old V_dev method did exactly that).
    lnI = np.log(I)
    dVdlnI = np.gradient(V, lnI)
    hi = logI >= logI.max() - 1.0
    if hi.sum() < 2:
        hi = np.argsort(logI)[-2:]
    Rs = float(np.polyfit(I[hi], dVdlnI[hi], 1)[0])
    return {"n": float(n), "Is": float(Is), "Rs": float(max(Rs, 0.0))}


def extract_mosfet(Vg, Id, Vd, Vg2=None, Id2=None, Vd2=None):
    """MOSFET parameters from transfer curve(s). Vg/Id: gate voltage
    and drain current (any consistent current unit — results are
    ratios/slopes except Ion/Ioff; Id is used as a magnitude, so the
    solver's negative drain-current sign convention is fine). Vt by
    max-transconductance extrapolation with the linear-region Vd/2
    correction (tangent hits Id = 0 at Vt + Vd/2); subthreshold slope
    SS in mV/dec from the steepest log10(Id) vs Vg region; Ion = Id at
    max Vg, Ioff = min Id. With a second curve (Vg2, Id2 at Vd2),
    DIBL = (Vt at low Vd - Vt at high Vd) / (Vd_high - Vd_low),
    in mV/V (positive when Vt falls as Vd rises)."""
    Vg = np.asarray(Vg, dtype=float)
    Id = np.abs(np.asarray(Id, dtype=float))
    if Vg.size < 5 or np.any(Id <= 0):
        raise ValueError("need >= 5 points with Id != 0")

    def _vt_ss(vg, idc, vd):
        # Max-transconductance tangent, LINEAR-region corrected:
        # in the linear region Id = K*[(Vg - Vt)*Vd - Vd^2/2], so the
        # tangent to Id vs Vg hits Id = 0 at Vg = Vt + Vd/2. The
        # reported Vt subtracts that Vd/2. Use low Vd (0.1 V) for the
        # textbook value; the correction is what makes DIBL come out
        # with the right sign (raw tangents rise with Vd).
        gm = np.gradient(idc, vg)
        k = int(np.argmax(gm))
        if k == 0:
            slope = (idc[1] - idc[0]) / (vg[1] - vg[0])
        elif k == len(vg) - 1:
            slope = (idc[-1] - idc[-2]) / (vg[-1] - vg[-2])
        else:
            slope = (idc[k + 1] - idc[k - 1]) / (vg[k + 1] - vg[k - 1])
        vt_lin = float(vg[k] - idc[k] / slope) if slope != 0 else float(vg[k])
        vt = vt_lin - vd / 2.0
        # SS: steepest decade of log10(Id) vs Vg, in mV/dec.
        logId = np.log10(idc)
        dlog = np.gradient(logId, vg)
        ss = float(1000.0 / np.max(dlog)) if np.max(dlog) > 0 else float("inf")
        return vt, ss, float(np.max(gm))

    Vt_, ss, gm_max = _vt_ss(Vg, Id, Vd)
    out = {"Vt": Vt_, "SS_mV_dec": ss, "gm_max": gm_max,
           "Ion": float(Id[-1]), "Ioff": float(Id.min()),
           "Ion_Ioff": float(Id[-1] / Id.min())}
    if Vg2 is not None and Id2 is not None and Vd2 is not None:
        Vt2, _, _ = _vt_ss(np.asarray(Vg2, dtype=float),
                           np.abs(np.asarray(Id2, dtype=float)), Vd2)
        # DIBL is positive when Vt falls as Vd rises:
        # (Vt at low Vd - Vt at high Vd) / (Vd_high - Vd_low).
        out["DIBL_mV_V"] = float((Vt2 - Vt_) / (Vd - Vd2) * 1000.0)
        out["Vt_Vd2"] = float(Vt2)
    return out
