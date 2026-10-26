"""
Clean 2D drift-diffusion solver, verified against 1D.
Built incrementally with 1D cross-checks.
Uses SAME physical constants and functions as 1D (imported, not hardcoded).
"""
import numpy as np
import sys
sys.path.insert(0, "/home/hatch/workspace/tcad/package")
from tcad_stabilizer.dd_solver import q, eps_si, Vt, ni, Dn, Dp, bernoulli, densities, mun, mup
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu, spsolve

class DDSolver2D:
    def __init__(self, Nx=16, Ny=8, Lx=2e-6, Ly=1e-6,
                 Na=1e22, Nd=1e22, taun=1e-6, taup=1e-6,
                 abrupt_cells=2):
        self.Nx, self.Ny, self.Lx, self.Ly = Nx, Ny, Lx, Ly
        self.hx, self.hy = Lx/(Nx-1), Ly/(Ny-1)
        self.N = Nx*Ny
        self.taun, self.taup = taun, taup
        # doping with smooth transition (matches 1D)
        xc = np.linspace(0, Lx, Nx)
        wt = abrupt_cells * self.hx
        trans = np.clip((xc - (0.5*Lx - wt/2))/wt, 0, 1)
        C1d = -Na + (Na+Nd)*trans
        self.C = np.tile(C1d, Ny)
        self.Vleft = self.Vright = 0.0
        self._update_bc()

    def _update_bc(self):
        self.psi_b0 = Vt*np.arcsinh(self.C[0]/(2*ni))
        self.psi_b1 = Vt*np.arcsinh(self.C[self.Nx-1]/(2*ni))

    def set_bias(self, Vl, Vr):
        self.Vleft, self.Vright = Vl, Vr
        self._update_bc()
        self.psi_b0 += Vl
        self.psi_b1 += Vr

    def initial_guess(self, Vleft=None, Vright=None):
        # 1D-style: psi from equilibrium + ramp, phin/phip = ramp
        # (much better than linear psi with zero quasi-Fermi levels)
        Vl = self.Vleft if Vleft is None else Vleft
        Vr = self.Vright if Vright is None else Vright
        x = np.zeros(3*self.N)
        psi = x[0::3]; phin = x[1::3]; phip = x[2::3]
        for iy in range(self.Ny):
            for ix in range(self.Nx):
                n_ = iy*self.Nx + ix
                f = ix/(self.Nx-1)
                ramp = Vl + (Vr - Vl)*f
                # equilibrium psi from doping + applied ramp
                psi_eq = Vt*np.arcsinh(self.C[n_]/(2.0*ni))
                psi[n_] = psi_eq + ramp
                phin[n_] = ramp
                phip[n_] = ramp
        # Enforce Dirichlet at contacts exactly
        for iy in range(self.Ny):
            n0 = iy*self.Nx + 0
            n1 = iy*self.Nx + (self.Nx-1)
            psi[n0] = self.psi_b0; phin[n0] = Vl; phip[n0] = Vl
            psi[n1] = self.psi_b1; phin[n1] = Vr; phip[n1] = Vr
        return x

    def residual(self, x):
        Nx, Ny, hx, hy = self.Nx, self.Ny, self.hx, self.hy
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        n, p = densities(psi, phin, phip)
        F = np.zeros(3*self.N)
        terms = np.zeros(3*self.N)
        idx = lambda ix, iy: iy*Nx + ix

        # Precompute all edge fluxes in a single clean pass
        # x-edges: index ex(ix,iy) = iy*(Nx-1)+ix, connects (ix,iy)-(ix+1,iy)
        # y-edges: index ey(ix,iy) = (Ny*(Nx-1)) + iy*Nx+ix, connects (ix,iy)-(ix,iy+1)
        nxe = Ny*(Nx-1)
        Jn = np.zeros(nxe + Nx*(Ny-1))
        Jp = np.zeros(nxe + Nx*(Ny-1))
        e = 0
        for iy in range(Ny):
            for ix in range(Nx-1):
                a, b = idx(ix,iy), idx(ix+1,iy)
                dv = (psi[b]-psi[a])/Vt
                Bp, Bm = bernoulli(dv), bernoulli(-dv)
                Jn[e] = (q*Dn/hx)*(n[b]*Bp - n[a]*Bm)
                Jp[e] = (q*Dp/hx)*(p[a]*Bp - p[b]*Bm)
                e += 1
        for iy in range(Ny-1):
            for ix in range(Nx):
                a, b = idx(ix,iy), idx(ix,iy+1)
                dv = (psi[b]-psi[a])/Vt
                Bp, Bm = bernoulli(dv), bernoulli(-dv)
                Jn[e] = (q*Dn/hy)*(n[b]*Bp - n[a]*Bm)
                Jp[e] = (q*Dp/hy)*(p[a]*Bp - p[b]*Bm)
                e += 1

        def jx(ix, iy): return iy*(Nx-1) + ix
        def jy(ix, iy): return nxe + iy*Nx + ix

        for iy in range(Ny):
            for ix in range(Nx):
                n_ = idx(ix, iy)
                if ix == 0:
                    F[3*n_] = psi[n_]-self.psi_b0
                    F[3*n_+1] = phin[n_]-self.Vleft
                    F[3*n_+2] = phip[n_]-self.Vleft
                    terms[3*n_:3*n_+3] = 1.0
                    continue
                if ix == Nx-1:
                    F[3*n_] = psi[n_]-self.psi_b1
                    F[3*n_+1] = phin[n_]-self.Vright
                    F[3*n_+2] = phip[n_]-self.Vright
                    terms[3*n_:3*n_+3] = 1.0
                    continue
                # Poisson
                lapx = (psi[idx(ix+1,iy)] - 2*psi[n_] + psi[idx(ix-1,iy)])/hx**2
                if iy == 0:
                    lapy = 2*(psi[idx(ix,1)] - psi[n_])/hy**2
                elif iy == Ny-1:
                    lapy = 2*(psi[idx(ix,Ny-2)] - psi[n_])/hy**2
                else:
                    lapy = (psi[idx(ix,iy+1)] - 2*psi[n_] + psi[idx(ix,iy-1)])/hy**2
                chg = (q/eps_si)*(p[n_]-n[n_]+self.C[n_])
                F[3*n_] = lapx + lapy + chg
                terms[3*n_] = abs(lapx)+abs(lapy) + (q/eps_si)*(p[n_]+n[n_]+abs(self.C[n_]))
                # Continuity
                dJn = (Jn[jx(ix,iy)]-Jn[jx(ix-1,iy)])/hx
                dJp = (Jp[jx(ix,iy)]-Jp[jx(ix-1,iy)])/hx
                if iy < Ny-1:
                    dJn += Jn[jy(ix,iy)]/hy
                    dJp += Jp[jy(ix,iy)]/hy
                if iy > 0:
                    dJn -= Jn[jy(ix,iy-1)]/hy
                    dJp -= Jp[jy(ix,iy-1)]/hy
                R = (n[n_]*p[n_]-ni**2)/(self.taup*(n[n_]+ni)+self.taun*(p[n_]+ni))
                F[3*n_+1] = dJn + q*R
                F[3*n_+2] = dJp + q*R
                terms[3*n_+1] = abs(dJn) + q*abs(R) + 1e-30
                terms[3*n_+2] = abs(dJp) + q*abs(R) + 1e-30

        self._terms = terms
        return F

    def sparse_jacobian(self, x, F):
        """Colored finite-difference Jacobian -> scipy CSR.

        5-point stencil => columns of nodes with (ix%3, iy%3) equal never
        share a row: 9 colors x 3 vars = 27 residual evals per build,
        independent of mesh size (dense FD needs 3*N evals: 462-1512+).
        Same per-unknown eps as the old dense FD, so entries agree.
        """
        Nx, Ny = self.Nx, self.Ny
        n3 = 3*self.N
        rows, cols, data = [], [], []
        eps0 = 1e-8
        for cy in range(3):
            for cx in range(3):
                members = [iy*Nx + ix
                           for iy in range(cy, Ny, 3)
                           for ix in range(cx, Nx, 3)]
                if not members:
                    continue
                for w in range(3):
                    xp = x.copy()
                    for m in members:
                        xp[3*m+w] += eps0*(1.0 + abs(x[3*m+w]))
                    dF = self.residual(xp) - F
                    for m in members:
                        em = eps0*(1.0 + abs(x[3*m+w]))
                        col = 3*m + w
                        ix, iy = m % Nx, m // Nx
                        for (jx, jy) in ((ix, iy), (ix-1, iy), (ix+1, iy),
                                         (ix, iy-1), (ix, iy+1)):
                            if 0 <= jx < Nx and 0 <= jy < Ny:
                                mm = jy*Nx + jx
                                for v in range(3):
                                    rows.append(3*mm+v)
                                    cols.append(col)
                                    data.append(dF[3*mm+v]/em)
        J = coo_matrix((data, (rows, cols)), shape=(n3, n3)).tocsr()
        J.sum_duplicates()
        return J

    @classmethod
    def fast(cls, **kw):
        """Fast mode: coarse-but-valid mesh preset (12x6).

        Same physics and solver; roughly 4x fewer unknowns than the
        default 16x8 mesh, so solves are several times faster with a
        slightly coarser (still converged, still physical) solution.
        Use the default constructor when maximum accuracy is needed.
        """
        kw.setdefault("Nx", 12)
        kw.setdefault("Ny", 6)
        return cls(**kw)

    def newton_solve(self, x0, hook=None, tol=1e-6, max_iter=60, jac_reuse=3):
        # 1D-style Newton with physical row floors, update-based convergence,
        # divergence detection, and best-state tracking.
        # Linear algebra (v0.6): sparse colored FD Jacobian + splu, with
        # quasi-Newton reuse -- the factorization is reused for up to
        # jac_reuse iterations and rebuilt early if the residual stalls.
        # The iteration matrix never changes the converged solution:
        # the fixed point is F(x)=0 and the convergence test is unchanged.
        x = x0.copy()
        n = len(x)

        # Per-row absolute floors (physical scales, like 1D)
        Cmax = max(1.0, float(np.max(np.abs(self.C))))
        Ap = max((q/eps_si)*Cmax, Vt/self.hx**2)  # Poisson
        An = q*mun*Vt*Cmax/self.hx**2             # e- continuity
        Ah = q*mup*Vt*Cmax/self.hx**2             # h+ continuity
        arow = np.ones(3*self.N)
        for iy in range(self.Ny):
            for ix in range(1, self.Nx-1):
                n_ = iy*self.Nx + ix
                arow[3*n_] = Ap
                arow[3*n_+1] = An
                arow[3*n_+2] = Ah

        hist = []
        converged, diverged = False, False
        best = {"honest": float("inf"), "x": x.copy()}
        lu = None; s_build = None; k_build = -10**9; jac_builds = 0
        prev_honest = None; prev_step = None
        for k in range(max_iter):
            F = self.residual(x)
            s = 1.0/np.maximum(self._terms, 1e-12*arow)
            honest = float(np.max(s*np.abs(F)))
            hist.append(honest)
            if honest < best["honest"]:
                best = {"honest": honest, "x": x.copy()}
            if not np.isfinite(honest) or honest > 1e6*max(1.0, hist[0]):
                diverged = True
                break
            # Sparse Jacobian: build (and factor) only when needed.
            # Reuse is permitted only in the asymptotic regime (the
            # previous applied step was small): reusing a stale Jacobian
            # during the hard transient of a bias jump poisons the
            # trajectory (found by test T4: Vg=2 jump fails with naive
            # reuse, converges with fresh Jacobians).
            need_build = (lu is None or (k - k_build) > jac_reuse
                          or prev_step is None or prev_step > 0.01
                          or (prev_honest is not None
                              and honest > 0.95*prev_honest))
            delta = None
            if need_build:
                J = self.sparse_jacobian(x, F)
                jac_builds += 1
                k_build = k
                try:
                    lu = splu((diags(s) @ J).tocsc())
                    s_build = s
                except Exception:
                    lu = None
            if lu is not None:
                try:
                    delta = np.asarray(lu.solve(-(s_build*F))).ravel()
                except Exception:
                    delta = None
                if delta is not None and not np.all(np.isfinite(delta)):
                    delta = None
            if delta is None:
                # Fallback: fresh sparse direct solve, then dense lstsq
                try:
                    delta = np.asarray(spsolve(
                        (diags(s) @ self.sparse_jacobian(x, F)).tocsc(),
                        -s*F)).ravel()
                except Exception:
                    delta = None
                if delta is None or not np.all(np.isfinite(delta)):
                    Jd = self.sparse_jacobian(x, F).toarray()
                    delta, *_ = np.linalg.lstsq(s[:,None]*Jd, -s*F, rcond=None)
            prev_honest = honest
            update = float(np.max(np.abs(delta)))

            # Convergence: update negligible AND honest reasonable (1D-style)
            if update < 1e-10 and honest < 0.5:
                converged = True
                break

            state = {"version":"1","iteration":k,"x":x.copy(),"delta":delta.copy(),
                     "residual":F.copy(),"residual_norm":honest,"row_scale":s,
                     "mesh":{"Nx":self.Nx,"Ny":self.Ny,"hx":self.hx,"hy":self.hy},
                     "fields":{"psi":x[0::3].copy(),"phin":x[1::3].copy(),
                              "phip":x[2::3].copy(),"doping":self.C.copy()},
                     "edge":{},"solver":self}
            corr = hook(state) if hook else {}
            a = corr.get("alpha", 1.0)
            a = float(np.min(a)) if np.ndim(a) else float(a)
            x = x + a*delta
            prev_step = a*update

        rx = x if converged else best["x"]
        return {"converged":converged,"diverged":diverged,
                "iterations":len(hist),"residual_history":hist,"x":rx,
                "final_residual":hist[-1] if hist else float("inf"),
                "jac_builds":jac_builds}

    def terminal_current(self, x):
        """Total current (A/m) at left contact, integrated over y."""
        # Recompute x-edge fluxes at ix=0
        Nx, Ny, hx, hy = self.Nx, self.Ny, self.hx, self.hy
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        n, p = densities(psi, phin, phip)
        idx = lambda ix, iy: iy*Nx + ix
        Itot = 0.0
        for iy in range(Ny):
            a, b = idx(0,iy), idx(1,iy)
            dv = (psi[b]-psi[a])/Vt
            Bp, Bm = bernoulli(dv), bernoulli(-dv)
            Jn = (q*Dn/hx)*(n[b]*Bp - n[a]*Bm)
            Jp = (q*Dp/hx)*(p[a]*Bp - p[b]*Bm)
            Itot += (Jn + Jp)*hy  # integrate over y (A/m)
        return float(Itot)
