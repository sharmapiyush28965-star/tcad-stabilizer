"""
3D drift-diffusion diode solver with SPARSE Jacobian.
Extends the proven 2D solver (dd_solver_2d.py) to 3D.

Key differences from 2D:
- 7-point Laplacian for Poisson (6 neighbors)
- SG fluxes on x/y/z edges
- Colored finite-difference Jacobian (27 colors x 3 vars = 81 residual
  evals per Newton step, independent of mesh size) assembled into
  scipy.sparse CSR; solved with spsolve.
- Contacts on x-faces (Dirichlet); Neumann (zero flux) on y/z faces.

Uses SAME physical constants and functions as 1D/2D (imported).
"""
import numpy as np
import sys
sys.path.insert(0, "/home/hatch/workspace/tcad/package")
from tcad_stabilizer.dd_solver import q, eps_si, Vt, ni, Dn, Dp, bernoulli, densities, mun, mup
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


class DDSolver3D:
    def __init__(self, Nx=8, Ny=6, Nz=5, Lx=2e-6, Ly=1e-6, Lz=1e-6,
                 Na=1e22, Nd=1e22, taun=1e-6, taup=1e-6,
                 abrupt_cells=2):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.hx, self.hy, self.hz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
        self.N = Nx*Ny*Nz
        self.taun, self.taup = taun, taup
        # doping: x-dependent only (matches 1D/2D), tiled in y,z
        xc = np.linspace(0, Lx, Nx)
        wt = abrupt_cells * self.hx
        trans = np.clip((xc - (0.5*Lx - wt/2))/wt, 0, 1)
        C1d = -Na + (Na+Nd)*trans
        self.C = np.tile(C1d, Ny*Nz)  # node-major: n=(iz*Ny+iy)*Nx+ix
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

    def idx(self, ix, iy, iz):
        return (iz*self.Ny + iy)*self.Nx + ix

    def initial_guess(self, Vleft=None, Vright=None):
        Vl = self.Vleft if Vleft is None else Vleft
        Vr = self.Vright if Vright is None else Vright
        x = np.zeros(3*self.N)
        psi = x[0::3]; phin = x[1::3]; phip = x[2::3]
        for iz in range(self.Nz):
            for iy in range(self.Ny):
                for ix in range(self.Nx):
                    n_ = self.idx(ix, iy, iz)
                    f = ix/(self.Nx-1)
                    ramp = Vl + (Vr - Vl)*f
                    psi[n_] = Vt*np.arcsinh(self.C[n_]/(2.0*ni)) + ramp
                    phin[n_] = ramp
                    phip[n_] = ramp
        # Enforce Dirichlet at x-contacts exactly
        for iz in range(self.Nz):
            for iy in range(self.Ny):
                n0 = self.idx(0, iy, iz)
                n1 = self.idx(self.Nx-1, iy, iz)
                psi[n0] = self.psi_b0; phin[n0] = Vl; phip[n0] = Vl
                psi[n1] = self.psi_b1; phin[n1] = Vr; phip[n1] = Vr
        return x

    def residual(self, x):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        hx, hy, hz = self.hx, self.hy, self.hz
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        n, p = densities(psi, phin, phip)
        F = np.zeros(3*self.N)
        terms = np.zeros(3*self.N)
        idx = self.idx

        # ---- SG edge fluxes ----
        # x-edges: ex(ix,iy,iz) = ((iz*Ny+iy)*(Nx-1)+ix)
        # y-edges: ey(ix,iy,iz) = nxe + ((iz*Nx+ix)*(Ny-1)+iy)
        # z-edges: ez(ix,iy,iz) = nxe+nye + ((iy*Nx+ix)*(Nz-1)+iz)
        nxe = Ny*Nz*(Nx-1)
        nye = Nx*Nz*(Ny-1)
        nze = Nx*Ny*(Nz-1)
        Jn = np.zeros(nxe+nye+nze)
        Jp = np.zeros(nxe+nye+nze)
        e = 0
        for iz in range(Nz):
            for iy in range(Ny):
                for ix in range(Nx-1):
                    a, b = idx(ix,iy,iz), idx(ix+1,iy,iz)
                    dv = (psi[b]-psi[a])/Vt
                    Bp, Bm = bernoulli(dv), bernoulli(-dv)
                    Jn[e] = (q*Dn/hx)*(n[b]*Bp - n[a]*Bm)
                    Jp[e] = (q*Dp/hx)*(p[a]*Bp - p[b]*Bm)
                    e += 1
        for iz in range(Nz):
            for ix in range(Nx):
                for iy in range(Ny-1):
                    a, b = idx(ix,iy,iz), idx(ix,iy+1,iz)
                    dv = (psi[b]-psi[a])/Vt
                    Bp, Bm = bernoulli(dv), bernoulli(-dv)
                    Jn[e] = (q*Dn/hy)*(n[b]*Bp - n[a]*Bm)
                    Jp[e] = (q*Dp/hy)*(p[a]*Bp - p[b]*Bm)
                    e += 1
        for iy in range(Ny):
            for ix in range(Nx):
                for iz in range(Nz-1):
                    a, b = idx(ix,iy,iz), idx(ix,iy,iz+1)
                    dv = (psi[b]-psi[a])/Vt
                    Bp, Bm = bernoulli(dv), bernoulli(-dv)
                    Jn[e] = (q*Dn/hz)*(n[b]*Bp - n[a]*Bm)
                    Jp[e] = (q*Dp/hz)*(p[a]*Bp - p[b]*Bm)
                    e += 1

        def ex(ix, iy, iz): return (iz*Ny + iy)*(Nx-1) + ix
        def ey(ix, iy, iz): return nxe + (iz*Nx + ix)*(Ny-1) + iy
        def ez(ix, iy, iz): return nxe + nye + (iy*Nx + ix)*(Nz-1) + iz

        for iz in range(Nz):
            for iy in range(Ny):
                for ix in range(Nx):
                    n_ = idx(ix, iy, iz)
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
                    # Poisson: 7-point Laplacian, Neumann mirror on y/z faces
                    lapx = (psi[idx(ix+1,iy,iz)] - 2*psi[n_]
                            + psi[idx(ix-1,iy,iz)])/hx**2
                    if iy == 0:
                        lapy = 2*(psi[idx(ix,1,iz)] - psi[n_])/hy**2
                    elif iy == Ny-1:
                        lapy = 2*(psi[idx(ix,Ny-2,iz)] - psi[n_])/hy**2
                    else:
                        lapy = (psi[idx(ix,iy+1,iz)] - 2*psi[n_]
                                + psi[idx(ix,iy-1,iz)])/hy**2
                    if iz == 0:
                        lapz = 2*(psi[idx(ix,iy,1)] - psi[n_])/hz**2
                    elif iz == Nz-1:
                        lapz = 2*(psi[idx(ix,iy,Nz-2)] - psi[n_])/hz**2
                    else:
                        lapz = (psi[idx(ix,iy,iz+1)] - 2*psi[n_]
                                + psi[idx(ix,iy,iz-1)])/hz**2
                    chg = (q/eps_si)*(p[n_]-n[n_]+self.C[n_])
                    F[3*n_] = lapx + lapy + lapz + chg
                    terms[3*n_] = (abs(lapx)+abs(lapy)+abs(lapz)
                                   + (q/eps_si)*(p[n_]+n[n_]+abs(self.C[n_])))
                    # Continuity: div J + qR = 0
                    dJn = (Jn[ex(ix,iy,iz)]-Jn[ex(ix-1,iy,iz)])/hx
                    dJp = (Jp[ex(ix,iy,iz)]-Jp[ex(ix-1,iy,iz)])/hx
                    if iy < Ny-1:
                        dJn += Jn[ey(ix,iy,iz)]/hy
                        dJp += Jp[ey(ix,iy,iz)]/hy
                    if iy > 0:
                        dJn -= Jn[ey(ix,iy-1,iz)]/hy
                        dJp -= Jp[ey(ix,iy-1,iz)]/hy
                    if iz < Nz-1:
                        dJn += Jn[ez(ix,iy,iz)]/hz
                        dJp += Jp[ez(ix,iy,iz)]/hz
                    if iz > 0:
                        dJn -= Jn[ez(ix,iy,iz-1)]/hz
                        dJp -= Jp[ez(ix,iy,iz-1)]/hz
                    R = ((n[n_]*p[n_]-ni**2)
                         / (self.taup*(n[n_]+ni)+self.taun*(p[n_]+ni)))
                    F[3*n_+1] = dJn + q*R
                    F[3*n_+2] = dJp + q*R
                    terms[3*n_+1] = abs(dJn) + q*abs(R) + 1e-30
                    terms[3*n_+2] = abs(dJp) + q*abs(R) + 1e-30

        self._terms = terms
        return F

    def sparse_jacobian(self, x, F):
        """Colored finite-difference Jacobian -> scipy CSR.

        7-point stencil => columns of nodes with (ix%3,iy%3,iz%3) equal
        never share a row. 27 colors x 3 vars = 81 residual evals.
        """
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        n3 = 3*self.N
        rows, cols, data = [], [], []
        eps0 = 1e-8
        for cz in range(3):
            for cy in range(3):
                for cx in range(3):
                    # nodes of this color
                    members = [self.idx(ix, iy, iz)
                               for iz in range(cz, Nz, 3)
                               for iy in range(cy, Ny, 3)
                               for ix in range(cx, Nx, 3)]
                    if not members:
                        continue
                    for w in range(3):
                        xp = x.copy()
                        ej = eps0*(1.0 + np.abs(x[0::3]))
                        # per-node epsilon for the perturbed unknowns
                        for m in members:
                            xp[3*m+w] += eps0*(1.0 + abs(x[3*m+w]))
                        Fp = self.residual(xp)
                        dF = Fp - F
                        for m in members:
                            em = eps0*(1.0 + abs(x[3*m+w]))
                            col = 3*m + w
                            ix = m % Nx
                            iy = (m//Nx) % Ny
                            iz = m//(Nx*Ny)
                            # stencil: self + 6 neighbors
                            for (jx, jy, jz) in (
                                    (ix, iy, iz),
                                    (ix-1, iy, iz), (ix+1, iy, iz),
                                    (ix, iy-1, iz), (ix, iy+1, iz),
                                    (ix, iy, iz-1), (ix, iy, iz+1)):
                                if 0 <= jx < Nx and 0 <= jy < Ny and 0 <= jz < Nz:
                                    mm = self.idx(jx, jy, jz)
                                    for v in range(3):
                                        rows.append(3*mm+v)
                                        cols.append(col)
                                        data.append(dF[3*mm+v]/em)
        J = coo_matrix((data, (rows, cols)), shape=(n3, n3)).tocsr()
        # drop exact-zero duplicates summed by coo (keep as is)
        J.sum_duplicates()
        return J

    def newton_solve(self, x0, hook=None, tol=1e-6, max_iter=60):
        # Same Newton as 2D (physical row floors, update-based convergence,
        # divergence detection, best-state tracking), but sparse linear solve.
        x = x0.copy()
        Cmax = max(1.0, float(np.max(np.abs(self.C))))
        Ap = max((q/eps_si)*Cmax, Vt/self.hx**2)
        An = q*mun*Vt*Cmax/self.hx**2
        Ah = q*mup*Vt*Cmax/self.hx**2
        arow = np.ones(3*self.N)
        for iz in range(self.Nz):
            for iy in range(self.Ny):
                for ix in range(1, self.Nx-1):
                    n_ = self.idx(ix, iy, iz)
                    arow[3*n_] = Ap
                    arow[3*n_+1] = An
                    arow[3*n_+2] = Ah

        hist = []
        converged, diverged = False, False
        best = {"honest": float("inf"), "x": x.copy()}
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
            J = self.sparse_jacobian(x, F)
            SJ = (diags(s) @ J).tocsc()  # row scaling
            try:
                delta = spsolve(SJ, -(s*F))
            except Exception:
                # fall back: dense least squares on small problems
                delta, *_ = np.linalg.lstsq(SJ.toarray(), -(s*F), rcond=None)
            delta = np.asarray(delta).ravel()
            update = float(np.max(np.abs(delta)))

            if update < 1e-10 and honest < 0.5:
                converged = True
                break

            state = {"version": "1", "iteration": k, "x": x.copy(),
                     "delta": delta.copy(), "residual": F.copy(),
                     "residual_norm": honest, "row_scale": s,
                     "mesh": {"Nx": self.Nx, "Ny": self.Ny, "Nz": self.Nz,
                              "hx": self.hx, "hy": self.hy, "hz": self.hz},
                     "fields": {"psi": x[0::3].copy(), "phin": x[1::3].copy(),
                                "phip": x[2::3].copy(), "doping": self.C.copy()},
                     "edge": {}, "solver": self}
            corr = hook(state) if hook else {}
            a = corr.get("alpha", 1.0)
            a = float(np.min(a)) if np.ndim(a) else float(a)
            x = x + a*delta

        rx = x if converged else best["x"]
        return {"converged": converged, "diverged": diverged,
                "iterations": len(hist), "residual_history": hist, "x": rx,
                "final_residual": hist[-1] if hist else float("inf")}

    def terminal_current(self, x):
        """Total current (A) at left contact, integrated over y-z face."""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        hx, hy, hz = self.hx, self.hy, self.hz
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        n, p = densities(psi, phin, phip)
        idx = self.idx
        Itot = 0.0
        for iz in range(Nz):
            for iy in range(Ny):
                a, b = idx(0,iy,iz), idx(1,iy,iz)
                dv = (psi[b]-psi[a])/Vt
                Bp, Bm = bernoulli(dv), bernoulli(-dv)
                Jn = (q*Dn/hx)*(n[b]*Bp - n[a]*Bm)
                Jp = (q*Dp/hx)*(p[a]*Bp - p[b]*Bm)
                Itot += (Jn + Jp)*hy*hz
        return float(Itot)

    def cross_section(self, x, plane="xy", fixed=None):
        """Return 2D arrays of psi for a mid-plane cross-section.

        plane: 'xy' (fixed iz), 'xz' (fixed iy), 'yz' (fixed ix).
        Returns (X, Y, Psi) for plotting.
        """
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        psi = x[0::3].reshape(Nz, Ny, Nx)
        if plane == "xy":
            iz = Nz//2 if fixed is None else fixed
            return psi[iz]
        elif plane == "xz":
            iy = Ny//2 if fixed is None else fixed
            return psi[:, iy, :]
        elif plane == "yz":
            ix = Nx//2 if fixed is None else fixed
            return psi[:, :, ix]
        raise ValueError("plane must be xy, xz or yz")
