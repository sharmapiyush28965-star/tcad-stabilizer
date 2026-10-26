"""
3D MOSFET test suite - all 8 tests must PASS.
Mirrors the 2D MOSFET protocol (test_mosfet_2d.py) on the SAME proven
x/y mesh (Nx=14, Ny_si=8, Ny_ox=3); the 3D device adds Nz=4 width
points. Currents are compared per unit width (A/m): the 3D solver
returns total A, divided by Lz.

Note: a coarser y mesh (Ny_si=6, Ny_ox=2) was verified to stall Newton
at strong inversion in BOTH the 2D and 3D solvers - the proven mesh
below is required. All bias points are reached by continuation
(standard TCAD practice), with the classical dvmax stabilizer hook on.
"""
import numpy as np, warnings, sys, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/hatch/workspace/tcad/package")
sys.path.insert(0, "/home/hatch/workspace/tcad/phase3_3d")
from mosfet_3d import MOSFET3D, eps_ox
from tcad_stabilizer.mosfet_2d import MOSFET2D
from tcad_stabilizer.dd_solver import q, eps_si, Vt, ni
from tcad_stabilizer import make_stabilizer

STAB = make_stabilizer("classical")
MESH = dict(Nx=14, Ny_si=8, Ny_ox=3)  # proven 2D mesh


def make3():
    return MOSFET3D(Nz=4, **MESH)


def equilibrium(m):
    m.set_bias(0, 0, 0)
    r = m.newton_solve(m.initial_guess(), hook=STAB, max_iter=40)
    assert r["converged"], "equilibrium failed to converge"
    return r["x"]


def step(m, x, Vd, Vg, max_iter=40):
    m.set_bias(0, Vd, Vg)
    r = m.newton_solve(x, hook=STAB, max_iter=max_iter)
    assert r["converged"], f"no convergence at Vd={Vd} Vg={Vg}"
    return r["x"]


def test1():
    """T1: 3D equilibrium converges."""
    m = make3()
    m.set_bias(0, 0, 0)
    r = m.newton_solve(m.initial_guess(), hook=STAB, max_iter=40)
    ok = r["converged"]
    print(f"T1 equilibrium: {'PASS' if ok else 'FAIL'} ({r['iterations']} iters)",
          flush=True)
    return ok


def test2():
    """T2: Vt physical - Id jumps from ~1e-10 to A/m scale between Vg 0->1."""
    m = make3()
    x = equilibrium(m)
    x = step(m, x, 0.1, 0.0)
    Ids = {}
    for Vg in [0.0, 1.0]:
        if Vg > 0:
            x = step(m, x, 0.1, Vg)
        Ids[Vg] = abs(m.drain_current_per_width(x))
    ok = (Ids[0.0] < 1e-6 and Ids[1.0] > 1e-3
          and Ids[1.0] / max(Ids[0.0], 1e-20) > 1e4)
    Na = m.Na_sub
    phif = Vt * np.log(Na / ni)
    Cox = eps_ox / m.tox
    gamma = np.sqrt(2 * q * eps_si * Na) / Cox
    Vt_an = 2 * phif + gamma * np.sqrt(2 * phif)
    print(f"T2 Vt physical: {'PASS' if ok else 'FAIL'} "
          f"(Id/W: {Ids[0.0]:.1e} @Vg0 -> {Ids[1.0]:.1e} @Vg1, "
          f"Vt_an={Vt_an:.2f}V long-channel)", flush=True)
    return ok


def test3():
    """T3: Id-Vgs increases (off -> on)."""
    m = make3()
    x = equilibrium(m)
    x = step(m, x, 0.1, 0.0)
    Ids = []
    for Vg in [0.0, 1.0, 2.0]:
        if Vg > 0:
            x = step(m, x, 0.1, Vg)
        Ids.append(abs(m.drain_current_per_width(x)))
    ok = Ids[0] < Ids[1] < Ids[2]
    print(f"T3 Id-Vgs: {'PASS' if ok else 'FAIL'} "
          f"({[f'{i:.1e}' for i in Ids]} A/m)", flush=True)
    return ok


def test4():
    """T4: Id-Vds linear then saturation (Vg=2)."""
    m = make3()
    x = equilibrium(m)
    x = step(m, x, 0.1, 0.0)
    x = step(m, x, 0.1, 1.0)
    x = step(m, x, 0.1, 2.0)
    Ids = []
    for Vd in [0.2, 0.5, 1.0, 1.5, 2.0]:
        x = step(m, x, Vd, 2.0)
        if Vd in (0.2, 1.0, 2.0):
            Ids.append(abs(m.drain_current_per_width(x)))
    ok = Ids[0] < Ids[1] < Ids[2] and (Ids[2] / Ids[1]) < (Ids[1] / Ids[0])
    print(f"T4 Id-Vds: {'PASS' if ok else 'FAIL'} "
          f"({[f'{i:.1e}' for i in Ids]} A/m)", flush=True)
    return ok


def test5():
    """T5: subthreshold slope in 50-200 mV/dec."""
    m = make3()
    x = equilibrium(m)
    x = step(m, x, 0.1, 0.0)
    Vgl = [0.1, 0.3, 0.5]
    Ids = []
    for Vg in Vgl:
        x = step(m, x, 0.1, Vg)
        Ids.append(abs(m.drain_current_per_width(x)))
    logI = np.log10(np.array(Ids))
    slope_mv = float(np.polyfit(logI, Vgl, 1)[0]) * 1000
    ok = 50 < slope_mv < 200
    print(f"T5 subthreshold: {'PASS' if ok else 'FAIL'} ({slope_mv:.0f} mV/dec)",
          flush=True)
    return ok


def test6():
    """T6: physical values at on-state (Vg=2, Vd=1)."""
    m = make3()
    x = equilibrium(m)
    x = step(m, x, 0.1, 0.0)
    x = step(m, x, 0.1, 1.0)
    x = step(m, x, 0.1, 2.0)
    x = step(m, x, 0.5, 2.0)
    x = step(m, x, 1.0, 2.0)
    Idw = abs(m.drain_current_per_width(x))
    ok = (np.all(np.isfinite(x)) and np.max(np.abs(x[0::3])) < 10.0
          and np.isfinite(Idw) and Idw < 1e6)
    print(f"T6 physical: {'PASS' if ok else 'FAIL'} "
          f"(max|psi|={np.max(np.abs(x[0::3])):.2f}V, Id/W={Idw:.2e} A/m)",
          flush=True)
    return ok


def test7():
    """T7: 3D matches 2D MOSFET per unit width on the same x/y mesh."""
    m2 = MOSFET2D(**MESH)
    m2.set_bias(0, 0, 0)
    r = m2.newton_solve(m2.initial_guess(), hook=STAB, max_iter=40)
    assert r["converged"], "2D equilibrium failed"
    x2 = _step2(m2, r["x"], 0.1, 0.0)
    m3 = make3()
    x3 = equilibrium(m3)
    x3 = step(m3, x3, 0.1, 0.0)
    ok = True
    details = []
    for (Vd, Vg) in [(0.1, 1.0), (0.5, 2.0)]:
        # Vd=0.1,Vg=1 first (continuation), then Vd=0.5,Vg=2 via Vg step
        if (Vd, Vg) == (0.5, 2.0):
            x2 = _step2(m2, x2, 0.1, 2.0)
            x3 = step(m3, x3, 0.1, 2.0)
        m2.set_bias(0, Vd, Vg)
        r = m2.newton_solve(x2, hook=STAB, max_iter=40)
        assert r["converged"], f"2D no conv at Vd={Vd} Vg={Vg}"
        x2 = r["x"]
        x3 = step(m3, x3, Vd, Vg)
        I2 = abs(m2.drain_current(x2))
        I3 = abs(m3.drain_current_per_width(x3))
        rel = abs(I3 - I2) / I2
        details.append(f"Vd={Vd},Vg={Vg}: 2D={I2:.3e} 3D/W={I3:.3e} rel={rel:.1e}")
        ok = ok and rel < 0.02
    print(f"T7 3D==2D: {'PASS' if ok else 'FAIL'} ({'; '.join(details)})",
          flush=True)
    return ok


def _step2(m, x, Vd, Vg):
    m.set_bias(0, Vd, Vg)
    r = m.newton_solve(x, hook=STAB, max_iter=40)
    assert r["converged"], f"2D no conv at Vd={Vd} Vg={Vg}"
    return r["x"]


def test8():
    """T8: stabilizer rescues brutal gate jump (eq -> Vd=0.1, Vg=2 in one
    step) where plain Newton diverges."""
    m = make3()
    x = equilibrium(m)
    m.set_bias(0, 0.1, 2.0)
    rp = m.newton_solve(x, hook=None, max_iter=30)
    rs = m.newton_solve(x, hook=STAB, max_iter=60)
    ok = (not rp["converged"]) and rs["converged"]
    extra = (f" Id/W={abs(m.drain_current_per_width(rs['x'])):.2e} A/m"
             if rs["converged"] else "")
    print(f"T8 stabilizer rescue: {'PASS' if ok else 'FAIL'} "
          f"(plain conv={rp['converged']}@{rp['iterations']}, "
          f"stab conv={rs['converged']}@{rs['iterations']}{extra})", flush=True)
    return ok


if __name__ == "__main__":
    t0 = time.time()
    R = [test1(), test2(), test3(), test4(), test5(), test6(), test7(), test8()]
    print(f"\nRESULT: {sum(R)}/8 PASS ({time.time()-t0:.0f}s total)", flush=True)
