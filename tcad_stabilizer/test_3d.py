"""
3D diode test suite - all 5 tests must PASS.
"""
import numpy as np, warnings, sys, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/hatch/workspace/tcad/package")
sys.path.insert(0, "/home/hatch/workspace/tcad/phase3_3d")
from tcad_stabilizer.dd_solver import Vt
from tcad_stabilizer.dd_solver_2d import DDSolver2D
from tcad_stabilizer import make_stabilizer
from diode_3d import DDSolver3D

NX, NY, NZ = 8, 5, 4  # small mesh for speed

def test1_equilibrium():
    s = DDSolver3D(Nx=NX, Ny=NY, Nz=NZ)
    t0 = time.time()
    r = s.newton_solve(s.initial_guess(), max_iter=30)
    ok = r["converged"]
    print(f"Test 1 (3D equilibrium converges): {'PASS' if ok else 'FAIL'} "
          f"(iters={r['iterations']}, {time.time()-t0:.0f}s)")
    return ok, s, r["x"]

def test2_matches_2d(s3, x3):
    """3D mid-z plane matches 2D solution (z-uniform problem)."""
    s2 = DDSolver2D(Nx=NX, Ny=NY)
    s2.set_bias(0, 0)
    x2 = s2.newton_solve(s2.initial_guess(), max_iter=40)["x"]
    psi2 = x2[0::3].reshape(NY, NX)
    psi3 = x3[0::3].reshape(NZ, NY, NX)
    err = np.max(np.abs(psi3[NZ//2] - psi2)) / max(np.max(np.abs(psi2)), 1e-12)
    ok = err < 1e-6
    print(f"Test 2 (3D matches 2D): {'PASS' if ok else 'FAIL'} (rel err={err:.2e})")
    return ok

def test3_iv_shape():
    s = DDSolver3D(Nx=NX, Ny=NY, Nz=NZ)
    s.set_bias(0, 0)
    x = s.newton_solve(s.initial_guess(),
                       hook=make_stabilizer("classical"))["x"]
    Vs = np.arange(0.3, 0.65, 0.05)
    Is = []
    for V in Vs:
        s.set_bias(V, 0)
        r = s.newton_solve(x, hook=make_stabilizer("classical"), max_iter=40)
        assert r["converged"], f"3D IV failed at {V}V"
        x = r["x"]
        Is.append(s.terminal_current(x))
    Is = np.array(Is)
    slope = (np.log(Is[-1])-np.log(Is[0]))/(Vs[-1]-Vs[0])
    err = abs(slope - 1/Vt)/(1/Vt)
    ok = err < 0.05
    print(f"Test 3 (3D IV exponential): {'PASS' if ok else 'FAIL'} "
          f"(slope={slope:.1f}, ideal={1/Vt:.1f}, err={err*100:.1f}%)")
    return ok

def test4_stabilizer_rescues():
    s = DDSolver3D(Nx=NX, Ny=NY, Nz=NZ)
    s.set_bias(0, 0)
    x0 = s.newton_solve(s.initial_guess(),
                        hook=make_stabilizer("classical"))["x"]
    s.set_bias(0.5, 0)
    rp = s.newton_solve(x0, hook=None, max_iter=25)
    rs = s.newton_solve(x0, hook=make_stabilizer("classical"), max_iter=40)
    ok = (not rp["converged"]) and rs["converged"]
    print(f"Test 4 (stabilizer rescues 3D): {'PASS' if ok else 'FAIL'} "
          f"(plain conv={rp['converged']}@{rp['iterations']}, "
          f"stab conv={rs['converged']}@{rs['iterations']})")
    return ok

def test5_physical(s, x):
    psi = x[0::3]
    I = s.terminal_current(x)
    ok = (np.all(np.isfinite(x)) and np.max(np.abs(psi)) < 5.0
          and abs(I) < 1e-12)  # equilibrium: ~zero current (tolerance for fp noise)
    print(f"Test 5 (3D physical): {'PASS' if ok else 'FAIL'} "
          f"(max|psi|={np.max(np.abs(psi)):.2f}V, I={I:.2e}A)")
    return ok

if __name__ == "__main__":
    t0 = time.time()
    results = []
    ok1, s3, x3 = test1_equilibrium(); results.append(ok1)
    results.append(test2_matches_2d(s3, x3))
    results.append(test3_iv_shape())
    results.append(test4_stabilizer_rescues())
    results.append(test5_physical(s3, x3))
    n = sum(results)
    print(f"\n{'='*50}\n3D TESTS: {n}/5 PASS ({time.time()-t0:.0f}s total)")
    print("ALL PASS - 3D is complete!" if n == 5 else "SOME FAILED")
    sys.exit(0 if n == 5 else 1)
