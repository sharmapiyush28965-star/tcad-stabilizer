"""
3D n-channel MOSFET drift-diffusion solver.

Combines two proven solvers:
- MOSFET physics from mosfet_2d.py: gate oxide (SiO2, Poisson-only),
  Si/SiO2 interface (displacement continuity, zero carrier flux),
  source/drain ohmic contacts, gate metal contact.
- 3D infrastructure from diode_3d.py: 3D mesh, sparse colored
  finite-difference Jacobian (scipy CSR + spsolve), same Newton
  (physical row floors, update-based convergence, best-state tracking)
  and the same stabilizer hook interface.

Coordinates:
- x: source -> drain (channel direction)
- y: vertical (silicon below, gate oxide on top)
- z: width direction (Neumann faces, solution uniform in z)

Node index: idx(ix, iy, iz) = (iz*Ny + iy)*Nx + ix  (as in diode_3d).
Unknowns per node: (psi, phin, phip), interleaved x[3n+0..2].
"""
import numpy as np
import sys
sys.path.insert(0, "/home/hatch/workspace/tcad/package")
from tcad_stabilizer.dd_solver import (q, eps_si, Vt, ni, Dn, Dp,
                                       bernoulli, densities, mun, mup)
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

eps0 = 8.854187817e-12
eps_ox = 3.9 * eps0  # SiO2 permittivity


class MOSFET3D:
    def __init__(self, Nx=14, Ny_si=8, Ny_ox=3, Nz=4,
                 Lx=2e-6, Hsi=1e-6, tox=8e-9, Lz=1e-6,
                 Na_sub=3e22, Nd_sd=1e26,
                 taun=1e-6, taup=1e-6,
                 Lch_frac=0.7):
        """
        Nx: points in x (source->drain)
        Ny_si: points in silicon (y)
        Ny_ox: points in oxide (y), >= 1 (top row is the gate)
        Nz: points in z (width)
        Lx: total length (m); Hsi: silicon depth (m); tox: oxide (m)
        Lz: device width (m)
        Na_sub: substrate doping (m^-3, p-type)
        Nd_sd: source/drain doping (m^-3, n+)
        Lch_frac: channel length as fraction of Lx (centered)
        """
        self.Nx, self.Ny_si, self.Ny_ox, self.Nz = Nx, Ny_si, Ny_ox, Nz
        self.Ny = Ny_si + Ny_ox
        self.Lx, self.Hsi, self.tox, self.Lz = Lx, Hsi, tox, Lz
        self.N = Nx * self.Ny * Nz
        self.taun, self.taup = taun, taup
        self.Na_sub, self.Nd_sd = Na_sub, Nd_sd

        # x mesh (uniform)
        self.hx = Lx / (Nx - 1)
        self.xc = np.linspace(0, Lx, Nx)
        # z mesh (uniform)
        self.hz = Lz / (Nz - 1) if Nz > 1 else Lz

        # y mesh: silicon (0..Hsi), oxide (Hsi..Hsi+tox)  [as mosfet_2d]
        self.yc = np.zeros(self.Ny)
        self.yc[:Ny_si] = np.linspace(0, Hsi, Ny_si)
        if Ny_ox > 0:
            y_ox = Hsi + np.linspace(0, tox, Ny_ox + 1)[1:]
            self.yc[Ny_si:] = y_ox
        self.hy_arr = np.diff(self.yc)  # length Ny-1, per y-interval

        # Region mask over y: True = silicon
        self.is_si = np.zeros(self.Ny, dtype=bool)
        self.is_si[:Ny_si] = True
        self.iy_interface = Ny_si - 1  # top silicon row

        # Doping profile in x (identical construction to mosfet_2d)
        Lch = Lx * Lch_frac
        x_src_end = (Lx - Lch) / 2
        x_drn_start = (Lx + Lch) / 2
        wt = 2 * self.hx
        C1d = np.zeros(Nx)
        for ix, xv in enumerate(self.xc):
            if xv < x_src_end - wt / 2:
                C1d[ix] = Nd_sd
            elif xv < x_src_end + wt / 2:
                f = (xv - (x_src_end - wt / 2)) / wt
                C1d[ix] = Nd_sd * (1 - f) + (-Na_sub) * f
            elif xv < x_drn_start - wt / 2:
                C1d[ix] = -Na_sub
            elif xv < x_drn_start + wt / 2:
                f = (xv - (x_drn_start - wt / 2)) / wt
                C1d[ix] = (-Na_sub) * (1 - f) + Nd_sd * f
            else:
                C1d[ix] = Nd_sd
        self.C1d = C1d
        # Tile: uniform in z; in y only inside silicon (0 in oxide)
        self.C = np.zeros(self.N)
        for iz in range(Nz):
            for iy in range(self.Ny):
                base = (iz * self.Ny + iy) * Nx
                if self.is_si[iy]:
                    self.C[base:base + Nx] = C1d
                # oxide rows stay 0

        # Biases
        self.Vs, self.Vd, self.Vg = 0.0, 0.0, 0.0
        self.Vfb = 0.0
        self._update_bc()

    # ------------------------------------------------------------------ BCs
    def _update_bc(self):
        # Ohmic contacts: psi from doping (as mosfet_2d)
        self.psi_s = Vt * np.arcsinh(self.C[self.idx(0, 0, 0)] / (2 * ni)) + self.Vs
        self.psi_d = Vt * np.arcsinh(self.C[self.idx(self.Nx - 1, 0, 0)] / (2 * ni)) + self.Vd
        # Gate (top of oxide): metal
        self.psi_g = self.Vg + self.Vfb

    def set_bias(self, Vs, Vd, Vg):
        self.Vs, self.Vd, self.Vg = Vs, Vd, Vg
        self._update_bc()

    def idx(self, ix, iy, iz):
        return (iz * self.Ny + iy) * self.Nx + ix

    def is_contact(self, ix, iy):
        """Dirichlet node? (independent of iz)"""
        if iy == self.Ny - 1:
            return True  # gate
        if self.is_si[iy] and (ix == 0 or ix == self.Nx - 1):
            return True  # source / drain
        return False

    # ------------------------------------------------------------ initial x
    def initial_guess(self):
        x = np.zeros(3 * self.N)
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        # Silicon pass
        for iz in range(self.Nz):
            for iy in range(self.Ny_si):
                for ix in range(self.Nx):
                    n_ = self.idx(ix, iy, iz)
                    psi_eq = Vt * np.arcsinh(self.C[n_] / (2 * ni))
                    f = ix / (self.Nx - 1)
                    ramp = self.Vs + (self.Vd - self.Vs) * f
                    psi[n_] = psi_eq + ramp
                    phin[n_] = ramp
                    phip[n_] = ramp
        # Oxide pass: linear between interface psi and gate psi
        iy_top = self.Ny - 1
        for iz in range(self.Nz):
            for iy in range(self.Ny_si, self.Ny):
                frac = (iy - self.iy_interface) / max(1, iy_top - self.iy_interface)
                for ix in range(self.Nx):
                    n_ = self.idx(ix, iy, iz)
                    psi_int = psi[self.idx(ix, self.iy_interface, iz)]
                    psi[n_] = (1 - frac) * psi_int + frac * self.psi_g
                    phin[n_] = 0.0
                    phip[n_] = 0.0
        self._apply_dirichlet(x)
        return x

    def _apply_dirichlet(self, x):
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        Nx = self.Nx
        for iz in range(self.Nz):
            for iy in range(self.Ny_si):
                n0 = self.idx(0, iy, iz)
                psi[n0] = self.psi_s; phin[n0] = self.Vs; phip[n0] = self.Vs
                n1 = self.idx(Nx - 1, iy, iz)
                psi[n1] = self.psi_d; phin[n1] = self.Vd; phip[n1] = self.Vd
            iy = self.Ny - 1
            for ix in range(Nx):
                n_ = self.idx(ix, iy, iz)
                psi[n_] = self.psi_g; phin[n_] = 0.0; phip[n_] = 0.0

    # -------------------------------------------------------------- residual
    def residual(self, x):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        hx, hz = self.hx, self.hz
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        n, p = densities(psi, phin, phip)
        F = np.zeros(3 * self.N)
        terms = np.zeros(3 * self.N)
        idx = self.idx
        is_si = self.is_si
        iy_if = self.iy_interface

        # ---- SG edge fluxes (carriers only inside silicon) ----
        nxe = Ny * Nz * (Nx - 1)
        nye = Nx * Nz * (Ny - 1)
        nze = Nx * Ny * (Nz - 1)
        Jn = np.zeros(nxe + nye + nze)
        Jp = np.zeros(nxe + nye + nze)
        e = 0
        # x-edges
        for iz in range(Nz):
            for iy in range(Ny):
                si = is_si[iy]
                for ix in range(Nx - 1):
                    if si:
                        a, b = idx(ix, iy, iz), idx(ix + 1, iy, iz)
                        dv = (psi[b] - psi[a]) / Vt
                        Bp, Bm = bernoulli(dv), bernoulli(-dv)
                        Jn[e] = (q * Dn / hx) * (n[b] * Bp - n[a] * Bm)
                        Jp[e] = (q * Dp / hx) * (p[a] * Bp - p[b] * Bm)
                    e += 1
        # y-edges (hy varies; flux only if both nodes silicon)
        # NOTE loop order iz,ix,iy must match ey() indexing (diode_3d).
        for iz in range(Nz):
            for ix in range(Nx):
                for iy in range(Ny - 1):
                    hy = self.hy_arr[iy]
                    if is_si[iy] and is_si[iy + 1]:
                        a, b = idx(ix, iy, iz), idx(ix, iy + 1, iz)
                        dv = (psi[b] - psi[a]) / Vt
                        Bp, Bm = bernoulli(dv), bernoulli(-dv)
                        Jn[e] = (q * Dn / hy) * (n[b] * Bp - n[a] * Bm)
                        Jp[e] = (q * Dp / hy) * (p[a] * Bp - p[b] * Bm)
                    e += 1
        # z-edges
        for iy in range(Ny):
            si = is_si[iy]
            for ix in range(Nx):
                for iz in range(Nz - 1):
                    if si:
                        a, b = idx(ix, iy, iz), idx(ix, iy, iz + 1)
                        dv = (psi[b] - psi[a]) / Vt
                        Bp, Bm = bernoulli(dv), bernoulli(-dv)
                        Jn[e] = (q * Dn / hz) * (n[b] * Bp - n[a] * Bm)
                        Jp[e] = (q * Dp / hz) * (p[a] * Bp - p[b] * Bm)
                    e += 1

        def ex(ix, iy, iz): return (iz * Ny + iy) * (Nx - 1) + ix
        def ey(ix, iy, iz): return nxe + (iz * Nx + ix) * (Ny - 1) + iy
        def ez(ix, iy, iz): return nxe + nye + (iy * Nx + ix) * (Nz - 1) + iz

        for iz in range(Nz):
            for iy in range(Ny):
                for ix in range(Nx):
                    n_ = idx(ix, iy, iz)

                    # ---------- Dirichlet contacts ----------
                    if self.is_contact(ix, iy):
                        if iy == Ny - 1:  # gate
                            F[3 * n_] = psi[n_] - self.psi_g
                            F[3 * n_ + 1] = phin[n_]
                            F[3 * n_ + 2] = phip[n_]
                        elif ix == 0:  # source
                            F[3 * n_] = psi[n_] - self.psi_s
                            F[3 * n_ + 1] = phin[n_] - self.Vs
                            F[3 * n_ + 2] = phip[n_] - self.Vs
                        else:  # drain
                            F[3 * n_] = psi[n_] - self.psi_d
                            F[3 * n_ + 1] = phin[n_] - self.Vd
                            F[3 * n_ + 2] = phip[n_] - self.Vd
                        terms[3 * n_:3 * n_ + 3] = 1.0
                        continue

                    # z-Laplacian (Neumann mirror on z faces, as diode_3d)
                    if Nz == 1:
                        lapz = 0.0
                    elif iz == 0:
                        lapz = 2 * (psi[idx(ix, iy, 1)] - psi[n_]) / hz ** 2
                    elif iz == Nz - 1:
                        lapz = 2 * (psi[idx(ix, iy, Nz - 2)] - psi[n_]) / hz ** 2
                    else:
                        lapz = (psi[idx(ix, iy, iz + 1)] - 2 * psi[n_]
                                + psi[idx(ix, iy, iz - 1)]) / hz ** 2

                    if not is_si[iy]:
                        # ---------- OXIDE: Poisson only ----------
                        # x-term: Neumann at x edges (as mosfet_2d)
                        if 0 < ix < Nx - 1:
                            lapx = (psi[idx(ix + 1, iy, iz)] - 2 * psi[n_]
                                    + psi[idx(ix - 1, iy, iz)]) / hx ** 2
                        else:
                            lapx = 0.0
                        # y-term: flux form, eps of region below for F_below
                        hy_below = self.hy_arr[iy - 1]
                        hy_above = self.hy_arr[iy]
                        eps_below = eps_si if is_si[iy - 1] else eps_ox
                        psi_below = psi[idx(ix, iy - 1, iz)]
                        psi_above = psi[idx(ix, iy + 1, iz)]
                        F_below = eps_below * (psi[n_] - psi_below) / hy_below
                        F_above = eps_ox * (psi_above - psi[n_]) / hy_above
                        hy_avg = 0.5 * (hy_below + hy_above)
                        ddy = (F_above - F_below) / hy_avg
                        F[3 * n_] = eps_ox * lapx + ddy + eps_ox * lapz
                        terms[3 * n_] = (abs(eps_ox * lapx) + abs(eps_ox * lapz)
                                         + (abs(F_above) + abs(F_below)) / hy_avg
                                         + 1e-20)
                        # pin phin/phip (no carriers in oxide)
                        F[3 * n_ + 1] = phin[n_]
                        F[3 * n_ + 2] = phip[n_]
                        terms[3 * n_ + 1] = 1.0
                        terms[3 * n_ + 2] = 1.0
                        continue

                    # ---------- SILICON ----------
                    lapx = (psi[idx(ix + 1, iy, iz)] - 2 * psi[n_]
                            + psi[idx(ix - 1, iy, iz)]) / hx ** 2
                    # y-term
                    if iy == 0:
                        # bottom Neumann
                        hy0 = self.hy_arr[0]
                        ddy = eps_si * 2 * (psi[idx(ix, 1, iz)] - psi[n_]) / hy0 ** 2
                        ddy_t = eps_si * 2 * abs(psi[idx(ix, 1, iz)] - psi[n_]) / hy0 ** 2
                    else:
                        hy_below = self.hy_arr[iy - 1]
                        hy_above = self.hy_arr[iy]
                        eps_above = eps_ox if iy == iy_if else eps_si
                        psi_below = psi[idx(ix, iy - 1, iz)]
                        psi_above = psi[idx(ix, iy + 1, iz)]
                        F_below = eps_si * (psi[n_] - psi_below) / hy_below
                        F_above = eps_above * (psi_above - psi[n_]) / hy_above
                        hy_avg = 0.5 * (hy_below + hy_above)
                        ddy = (F_above - F_below) / hy_avg
                        ddy_t = (abs(F_above) + abs(F_below)) / hy_avg
                    F[3 * n_] = (eps_si * lapx + ddy + eps_si * lapz
                                 + q * (p[n_] - n[n_] + self.C[n_]))
                    terms[3 * n_] = (abs(eps_si * lapx) + ddy_t
                                     + abs(eps_si * lapz)
                                     + q * (p[n_] + n[n_] + abs(self.C[n_]))
                                     + 1e-20)

                    # Continuity (no carrier flux into oxide / out of z faces)
                    dJn = (Jn[ex(ix, iy, iz)] - Jn[ex(ix - 1, iy, iz)]) / hx
                    dJp = (Jp[ex(ix, iy, iz)] - Jp[ex(ix - 1, iy, iz)]) / hx
                    if iy < iy_if:
                        hy = self.hy_arr[iy]
                        dJn += Jn[ey(ix, iy, iz)] / hy
                        dJp += Jp[ey(ix, iy, iz)] / hy
                    if iy > 0:
                        hy = self.hy_arr[iy - 1]
                        dJn -= Jn[ey(ix, iy - 1, iz)] / hy
                        dJp -= Jp[ey(ix, iy - 1, iz)] / hy
                    if iz < Nz - 1:
                        dJn += Jn[ez(ix, iy, iz)] / hz
                        dJp += Jp[ez(ix, iy, iz)] / hz
                    if iz > 0:
                        dJn -= Jn[ez(ix, iy, iz - 1)] / hz
                        dJp -= Jp[ez(ix, iy, iz - 1)] / hz
                    R = ((n[n_] * p[n_] - ni ** 2)
                         / (self.taup * (n[n_] + ni) + self.taun * (p[n_] + ni)))
                    F[3 * n_ + 1] = dJn + q * R
                    F[3 * n_ + 2] = dJp + q * R
                    terms[3 * n_ + 1] = abs(dJn) + q * abs(R) + 1e-30
                    terms[3 * n_ + 2] = abs(dJp) + q * abs(R) + 1e-30

        self._terms = terms
        return F

    # ---------------------------------------------------- sparse Jacobian
    def sparse_jacobian(self, x, F):
        """Colored finite-difference Jacobian -> scipy CSR.

        Nearest-neighbour stencil => nodes with (ix%3, iy%3, iz%3) equal
        never share a row. 27 colors x 3 vars = 81 residual evals,
        independent of mesh size. (Same scheme as diode_3d, verified exact
        against direct FD there.)
        """
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        n3 = 3 * self.N
        rows, cols, data = [], [], []
        eps_fd = 1e-8
        for cz in range(3):
            for cy in range(3):
                for cx in range(3):
                    members = [self.idx(ix, iy, iz)
                               for iz in range(cz, Nz, 3)
                               for iy in range(cy, Ny, 3)
                               for ix in range(cx, Nx, 3)]
                    if not members:
                        continue
                    for w in range(3):
                        xp = x.copy()
                        for m in members:
                            xp[3 * m + w] += eps_fd * (1.0 + abs(x[3 * m + w]))
                        Fp = self.residual(xp)
                        dF = Fp - F
                        for m in members:
                            em = eps_fd * (1.0 + abs(x[3 * m + w]))
                            col = 3 * m + w
                            ix = m % Nx
                            iy = (m // Nx) % Ny
                            iz = m // (Nx * Ny)
                            for (jx, jy, jz) in (
                                    (ix, iy, iz),
                                    (ix - 1, iy, iz), (ix + 1, iy, iz),
                                    (ix, iy - 1, iz), (ix, iy + 1, iz),
                                    (ix, iy, iz - 1), (ix, iy, iz + 1)):
                                if 0 <= jx < Nx and 0 <= jy < Ny and 0 <= jz < Nz:
                                    mm = self.idx(jx, jy, jz)
                                    for v in range(3):
                                        rows.append(3 * mm + v)
                                        cols.append(col)
                                        data.append(dF[3 * mm + v] / em)
        J = coo_matrix((data, (rows, cols)), shape=(n3, n3)).tocsr()
        J.sum_duplicates()
        return J

    # ------------------------------------------------------------ Newton
    def newton_solve(self, x0, hook=None, tol=1e-6, max_iter=60):
        """Newton with stabilizer hook (same scheme as diode_3d /
        mosfet_2d: physical row floors, update-based convergence,
        divergence detection, best-state tracking, sparse solve)."""
        x = x0.copy()
        Cmax = max(1.0, float(np.max(np.abs(self.C))))
        Ap = max((q / eps_si) * Cmax, Vt / self.hx ** 2)
        An = q * mun * Vt * Cmax / self.hx ** 2
        Ah = q * mup * Vt * Cmax / self.hx ** 2
        arow = np.ones(3 * self.N)
        for iz in range(self.Nz):
            for iy in range(self.Ny):
                for ix in range(self.Nx):
                    if self.is_contact(ix, iy):
                        continue
                    nn = self.idx(ix, iy, iz)
                    if self.is_si[iy]:
                        arow[3 * nn] = Ap
                        arow[3 * nn + 1] = An
                        arow[3 * nn + 2] = Ah
                    else:
                        arow[3 * nn] = Vt / self.hy_arr[min(iy, len(self.hy_arr) - 1)] ** 2

        hist = []
        converged, diverged = False, False
        best = {"honest": float("inf"), "x": x.copy()}
        for k in range(max_iter):
            F = self.residual(x)
            s = 1.0 / np.maximum(self._terms, 1e-12 * arow)
            honest = float(np.max(s * np.abs(F)))
            hist.append(honest)
            if honest < best["honest"]:
                best = {"honest": honest, "x": x.copy()}
            if not np.isfinite(honest) or honest > 1e6 * max(1.0, hist[0]):
                diverged = True
                break
            J = self.sparse_jacobian(x, F)
            SJ = (diags(s) @ J).tocsc()
            try:
                delta = spsolve(SJ, -(s * F))
            except Exception:
                delta, *_ = np.linalg.lstsq(SJ.toarray(), -(s * F), rcond=None)
            delta = np.asarray(delta).ravel()
            if not np.all(np.isfinite(delta)):
                diverged = True
                break
            update = float(np.max(np.abs(delta)))

            if update < 1e-10 and honest < 0.5:
                converged = True
                break

            state = {"version": "1", "iteration": k, "x": x.copy(),
                     "delta": delta.copy(), "residual": F.copy(),
                     "residual_norm": honest, "row_scale": s,
                     "mesh": {"Nx": self.Nx, "Ny": self.Ny, "Nz": self.Nz,
                              "hx": self.hx, "hy": float(self.hy_arr[0]),
                              "hz": self.hz},
                     "fields": {"psi": x[0::3].copy(), "phin": x[1::3].copy(),
                                "phip": x[2::3].copy(), "doping": self.C.copy()},
                     "edge": {}, "solver": self}
            corr = hook(state) if hook else {}
            a = corr.get("alpha", 1.0)
            a = float(np.min(a)) if np.ndim(a) else float(a)
            x = x + a * delta

        rx = x if converged else best["x"]
        return {"converged": converged, "diverged": diverged,
                "iterations": len(hist), "residual_history": hist, "x": rx,
                "final_residual": hist[-1] if hist else float("inf")}

    # ------------------------------------------------------------- outputs
    def drain_current(self, x):
        """Total drain current (A), integrated over the y-z face at drain."""
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        n, p = densities(psi, phin, phip)
        Nx, Nz = self.Nx, self.Nz
        idx = self.idx
        hx, hz = self.hx, self.hz
        Itot = 0.0
        ix = Nx - 2  # edge between Nx-2 and Nx-1 (drain)
        for iz in range(Nz):
            if Nz == 1:
                wz = self.Lz
            else:
                wz = hz / 2 if (iz == 0 or iz == Nz - 1) else hz
            for iy in range(self.Ny_si):
                if iy == 0:
                    wy = self.hy_arr[0] / 2
                elif iy == self.Ny_si - 1:
                    wy = self.hy_arr[iy - 1] / 2
                else:
                    wy = (self.hy_arr[iy - 1] + self.hy_arr[iy]) / 2
                a, b = idx(ix, iy, iz), idx(ix + 1, iy, iz)
                dv = (psi[b] - psi[a]) / Vt
                Bp, Bm = bernoulli(dv), bernoulli(-dv)
                Jn = (q * Dn / hx) * (n[b] * Bp - n[a] * Bm)
                Jp = (q * Dp / hx) * (p[a] * Bp - p[b] * Bm)
                Itot += (Jn + Jp) * wy * wz
        return float(Itot)

    def drain_current_per_width(self, x):
        """Drain current per unit width (A/m) — comparable to MOSFET2D."""
        return self.drain_current(x) / self.Lz

    def get_fields(self, x):
        """Dict of 3D arrays shaped (Nz, Ny, Nx): psi, phin, phip, n, p."""
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        n, p = densities(psi, phin, phip)
        shp = (self.Nz, self.Ny, self.Nx)
        return {"psi": psi.reshape(shp), "phin": phin.reshape(shp),
                "phip": phip.reshape(shp), "n": n.reshape(shp),
                "p": p.reshape(shp)}

    def cross_section(self, x, iz=None):
        """psi on the x-y plane at mid-width (iz = Nz//2).

        Returns array shaped (Ny, Nx): rows y (silicon then oxide),
        columns x (source -> drain)."""
        if iz is None:
            iz = self.Nz // 2
        return self.get_fields(x)["psi"][iz]
