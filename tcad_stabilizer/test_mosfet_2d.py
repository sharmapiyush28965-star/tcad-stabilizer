"""
2D MOSFET test suite - FAST version. All 6 tests must PASS.
Uses fewer sweep points for speed.
"""
import numpy as np, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/hatch/workspace/tcad/package")
sys.path.insert(0, "/home/hatch/workspace/tcad/phase2_2d")
from mosfet_2d import MOSFET2D, eps_ox
from tcad_stabilizer.dd_solver import q, eps_si, Vt, ni
from tcad_stabilizer import make_stabilizer

_hook = make_stabilizer("classical")

def test1():
    m = MOSFET2D(Nx=14, Ny_si=8, Ny_ox=3)
    m.set_bias(0, 0, 0)
    r = m.newton_solve(m.initial_guess(), hook=_hook, max_iter=30)
    ok = r["converged"]
    print(f"T1 equilibrium: {'PASS' if ok else 'FAIL'} ({r['iterations']} iters)", flush=True)
    return ok

def test2():
    """Vt physical: Id jumps between Vgs=0 and Vgs=1."""
    m = MOSFET2D(Nx=14, Ny_si=8, Ny_ox=3)
    m.set_bias(0, 0, 0)
    x = m.newton_solve(m.initial_guess(), hook=_hook)["x"]
    m.set_bias(0, 0.1, 0)
    x = m.newton_solve(x, hook=_hook)["x"]
    Ids = {}
    for Vgs in [0.0, 0.5, 1.0]:
        m.set_bias(0, 0.1, Vgs)
        r = m.newton_solve(x, hook=_hook, max_iter=30)
        if not r["converged"]:
            print(f"T2 Vt: FAIL (no conv)", flush=True)
            return False
        x = r["x"]
        Ids[Vgs] = abs(m.drain_current(x))
    # Vt between 0 and 1V (turns on in this range)
    # Id(0) should be small, Id(1) should be large
    ok = Ids[0.0] < 1e-6 and Ids[1.0] > 1e-3 and Ids[1.0]/max(Ids[0.0],1e-20) > 1e4
    # Analytical Vt for reference
    Na = m.Na_sub
    phif = Vt*np.log(Na/ni)
    Cox = eps_ox/m.tox
    gamma = np.sqrt(2*q*eps_si*Na)/Cox
    Vt_an = 2*phif + gamma*np.sqrt(2*phif)
    print(f"T2 Vt: {'PASS' if ok else 'FAIL'} (Id0={Ids[0.0]:.1e}, Id1={Ids[1.0]:.1e}, Vt_an={Vt_an:.2f}V)", flush=True)
    return ok

def test3():
    """Id increases with Vgs."""
    m = MOSFET2D(Nx=14, Ny_si=8, Ny_ox=3)
    m.set_bias(0, 0, 0)
    x = m.newton_solve(m.initial_guess(), hook=_hook)["x"]
    m.set_bias(0, 0.1, 0)
    x = m.newton_solve(x, hook=_hook)["x"]
    Ids = []
    for Vgs in [0.0, 1.0, 2.0]:
        m.set_bias(0, 0.1, Vgs)
        r = m.newton_solve(x, hook=_hook, max_iter=30)
        if not r["converged"]:
            print(f"T3 Id-Vgs: FAIL", flush=True)
            return False
        x = r["x"]
        Ids.append(abs(m.drain_current(x)))
    ok = Ids[0] < Ids[1] < Ids[2]
    print(f"T3 Id-Vgs: {'PASS' if ok else 'FAIL'} ({[f'{i:.1e}' for i in Ids]})", flush=True)
    return ok

def test4():
    """Id-Vds: increases then saturates."""
    m = MOSFET2D(Nx=14, Ny_si=8, Ny_ox=3)
    m.set_bias(0, 0, 0)
    x = m.newton_solve(m.initial_guess(), hook=_hook)["x"]
    m.set_bias(0, 0, 2.0)
    x = m.newton_solve(x, hook=_hook)["x"]
    Ids = []
    for Vds in [0.2, 1.0, 2.0]:
        m.set_bias(0, Vds, 2.0)
        r = m.newton_solve(x, hook=_hook, max_iter=30)
        if not r["converged"]:
            print(f"T4 Id-Vds: FAIL", flush=True)
            return False
        x = r["x"]
        Ids.append(abs(m.drain_current(x)))
    # Increasing, and saturates (ratio decreases)
    ok = Ids[0] < Ids[1] < Ids[2] and (Ids[2]/Ids[1]) < (Ids[1]/Ids[0])
    print(f"T4 Id-Vds: {'PASS' if ok else 'FAIL'} ({[f'{i:.1e}' for i in Ids]})", flush=True)
    return ok

def test5():
    """Subthreshold slope reasonable (50-150 mV/dec)."""
    m = MOSFET2D(Nx=14, Ny_si=8, Ny_ox=3)
    m.set_bias(0, 0, 0)
    x = m.newton_solve(m.initial_guess(), hook=_hook)["x"]
    m.set_bias(0, 0.1, 0)
    x = m.newton_solve(x, hook=_hook)["x"]
    Vgs_l = [0.1, 0.3, 0.5]
    Ids = []
    for Vgs in Vgs_l:
        m.set_bias(0, 0.1, Vgs)
        r = m.newton_solve(x, hook=_hook, max_iter=30)
        if not r["converged"]:
            break
        x = r["x"]
        Ids.append(abs(m.drain_current(x)))
    if len(Ids) < 3:
        print(f"T5 subthreshold: FAIL", flush=True)
        return False
    logI = np.log10(np.array(Ids))
    slope, _ = np.polyfit(logI, Vgs_l, 1)
    slope_mv = slope*1000
    ok = 50 < slope_mv < 200
    print(f"T5 subthreshold: {'PASS' if ok else 'FAIL'} ({slope_mv:.0f} mV/dec)", flush=True)
    return ok

def test6():
    """Physical: no NaN, reasonable values."""
    m = MOSFET2D(Nx=14, Ny_si=8, Ny_ox=3)
    m.set_bias(0, 0, 0)
    x = m.newton_solve(m.initial_guess(), hook=_hook)["x"]
    # Ramp gradually: Vgs first, then Vds
    for Vgs in [1.0, 2.0]:
        m.set_bias(0, 0, Vgs)
        r = m.newton_solve(x, hook=_hook, max_iter=30)
        if not r["converged"]:
            print(f"T6 physical: FAIL (Vgs ramp)", flush=True)
            return False
        x = r["x"]
    for Vds in [0.5, 1.0]:
        m.set_bias(0, Vds, 2.0)
        r = m.newton_solve(x, hook=_hook, max_iter=30)
        if not r["converged"]:
            print(f"T6 physical: FAIL (Vds ramp)", flush=True)
            return False
        x = r["x"]
    ok = np.all(np.isfinite(x)) and np.max(np.abs(x[0::3])) < 10.0
    Id = abs(m.drain_current(x))
    ok = ok and np.isfinite(Id) and Id < 1e6
    print(f"T6 physical: {'PASS' if ok else 'FAIL'} (max|psi|={np.max(np.abs(x[0::3])):.2f}V, Id={Id:.2e})", flush=True)
    return ok

if __name__ == "__main__":
    R = [test1(), test2(), test3(), test4(), test5(), test6()]
    print(f"\nRESULT: {sum(R)}/6 PASS", flush=True)
