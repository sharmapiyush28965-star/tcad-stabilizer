"""Tests for the fab-grade Phase 1 module (fab.py).

Rule: synthetic "measured" data is generated from KNOWN parameters, and
the fit must recover them. Extraction is checked against hand-computed
values. No tolerance is loosened to hide a bad fit.
"""
import csv
import os
import tempfile

import numpy as np

from tcad_stabilizer.dd_solver import DriftDiffusion1D, Vt
from tcad_stabilizer.dd_solver_2d import DDSolver2D
from tcad_stabilizer.mosfet_2d import MOSFET2D
from tcad_stabilizer.stabilizer import make_stabilizer
from tcad_stabilizer.fab import (
    load_iv_csv,
    load_profile_csv,
    diode_sweep,
    calibrate_diode,
    extract_diode,
    extract_mosfet,
)


def test_profile_import_1d():
    s = DriftDiffusion1D(N=60, L=2e-6, Na=1e22, Nd=1e22)
    # Nonuniform profile: junction at 30% instead of 50%.
    x = np.linspace(0, 2e-6, 25)
    trans = np.clip((x - 0.3 * 2e-6) / (0.2e-6), 0, 1)
    C = -1e22 + 2e22 * trans
    out = s.set_doping_profile(x, C)
    assert np.allclose(out, np.interp(s.xc, x, C))
    # Boundary values recomputed from the new end doping.
    assert abs(s.psi_b0 - Vt * np.arcsinh(C[0] / 2e16)) < 1e-12
    # Still solves, current positive and sane.
    s.set_bias(0, 0)
    r = s.newton_solve(s.initial_guess(), hook=make_stabilizer("classical"))
    assert r["converged"]
    s.set_bias(0.5, 0)
    r = s.newton_solve(r["x"], hook=make_stabilizer("classical"))
    J = s.terminal_current(r["x"])
    assert J > 0 and J < 1e6
    print("PASS profile import 1D — J@0.5V =", round(J, 1))


def test_profile_import_2d():
    s = DDSolver2D.fast()
    x = np.linspace(0, 2e-6, 25)
    trans = np.clip((x - 0.3 * 2e-6) / (0.2e-6), 0, 1)
    C = -1e22 + 2e22 * trans
    out = s.set_doping_profile(x, C)
    C1d = np.interp(np.linspace(0, 2e-6, s.Nx), x, C)
    assert np.allclose(out[: s.Nx], C1d)
    assert np.allclose(out[s.Nx : 2 * s.Nx], C1d)  # tiled over y
    V, J = diode_sweep(s, [0.2, 0.4, 0.6])
    assert np.all(J > 0) and np.all(np.diff(J) > 0)
    print("PASS profile import 2D — J@0.6V =", round(J[-1], 1))


def test_csv_load():
    with tempfile.TemporaryDirectory() as d:
        iv_path = os.path.join(d, "iv.csv")
        with open(iv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Voltage (V)", "Current (A)"])
            for v in [0.1, 0.2, 0.3]:
                w.writerow([v, 1e-9 * v])
        V, I = load_iv_csv(iv_path)
        assert np.allclose(V, [0.1, 0.2, 0.3])
        assert np.allclose(I, [1e-10, 2e-10, 3e-10])
        prof_path = os.path.join(d, "prof.csv")
        with open(prof_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["depth (cm)", "doping (cm^-3)"])
            w.writerow([0.0, -1e16])
            w.writerow([1e-4, 1e16])
        x, C = load_profile_csv(prof_path, x_unit="cm", c_unit="cm^-3")
        assert np.allclose(x, [0.0, 1e-6])
        assert np.allclose(C, [-1e22, 1e22])
    print("PASS csv load (IV + profile, cm units)")


def test_calibration_recovery():
    # Synthetic measured data from KNOWN parameters + 1% noise.
    # Junction voltages only to 0.5 V: at 0.6 V the ideal current is
    # ~1e5 A/m^2 and any Rs makes the terminal voltage unphysical.
    # Lifetimes are TIED (taun = taup = tau): one forward IV curve
    # cannot separate them — see calibrate_diode docstring.
    tau_t, cscale_t, Rs_t = 5e-7, 1.0, 2e-5
    rng = np.random.default_rng(0)

    def factory(taun, taup, cscale):
        s = DriftDiffusion1D(N=60, L=2e-6, Na=1e22, Nd=1e22,
                             taun=taun, taup=taup)
        s.set_doping_profile(s.xc, s.C * cscale)
        return s

    Vj = np.arange(0.1, 0.51, 0.05)
    _, J = diode_sweep(factory(tau_t, tau_t, cscale_t), Vj)
    V_meas = Vj + J * Rs_t
    I_meas = J * (1.0 + 0.01 * rng.standard_normal(J.size))
    fit = calibrate_diode(factory, V_meas, I_meas,
                          p0=(1e-6, 1.0, 0.0))
    assert abs(np.log10(fit["tau"] / tau_t)) < 0.3, fit
    assert abs(fit["cscale"] / cscale_t - 1) < 0.2, fit
    assert abs(fit["Rs"] / Rs_t - 1) < 0.5, fit
    assert fit["rms_log_err"] < 0.05, fit
    print("PASS calibration recovery — tau=%.2g cscale=%.2f "
          "Rs=%.2g rms=%.3f dex" % (fit["tau"], fit["cscale"], fit["Rs"],
                                    fit["rms_log_err"]))


def test_extract_diode():
    Is_t, n_t = 1e-12, 1.3
    V = np.linspace(0.05, 0.7, 26)
    I = Is_t * (np.exp(V / (n_t * Vt)) - 1.0)
    out = extract_diode(V, I)
    assert abs(out["n"] / n_t - 1) < 0.03, out
    assert abs(out["Is"] / Is_t - 1) < 0.10, out
    assert out["Rs"] < 1e-6, out
    print("PASS extract diode — n=%.3f Is=%.2g Rs=%.2g" % (out["n"],
          out["Is"], out["Rs"]))


def _mosfet_transfer(vd):
    """Verified recipe: equilibrium -> Vd ramp -> Vg sweep."""
    m = MOSFET2D.fast()
    hook = make_stabilizer("classical")
    m.set_bias(0, 0, 0)
    r = m.newton_solve(m.initial_guess(), hook=hook, max_iter=30)
    x = r["x"]
    for step in (0.1, 0.3, 0.5):
        if step > vd:
            break
        m.set_bias(0, step, 0)
        r = m.newton_solve(x, hook=hook, max_iter=30)
        x = r["x"]
    Vg, Id = [], []
    for vg in np.arange(0, 2.01, 0.25):
        m.set_bias(0, vd, vg)
        r = m.newton_solve(x, hook=hook, max_iter=30)
        x = r["x"]
        Vg.append(vg)
        Id.append(abs(m.drain_current(x)))  # codebase convention: abs()
    return np.asarray(Vg), np.asarray(Id)


def test_extract_mosfet():
    Vg, Id05 = _mosfet_transfer(0.5)
    Vg2, Id01 = _mosfet_transfer(0.1)
    out = extract_mosfet(Vg, Id05, 0.5, Vg2, Id01, 0.1)
    assert 0.1 < out["Vt"] < 1.2, out
    assert 55 < out["SS_mV_dec"] < 300, out
    assert out["Ion_Ioff"] > 1e2, out
    assert 0 < out["DIBL_mV_V"] < 300, out
    print("PASS extract mosfet — Vt=%.2fV SS=%.0f mV/dec Ion/Ioff=%.1e "
          "DIBL=%.0f mV/V" % (out["Vt"], out["SS_mV_dec"],
                              out["Ion_Ioff"], out["DIBL_mV_V"]))


if __name__ == "__main__":
    test_profile_import_1d()
    test_profile_import_2d()
    test_csv_load()
    test_calibration_recovery()
    test_extract_diode()
    test_extract_mosfet()
    print("==== ALL FAB TESTS PASS ====")
