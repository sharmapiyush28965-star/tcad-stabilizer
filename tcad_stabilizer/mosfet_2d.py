"""
2D n-channel MOSFET drift-diffusion solver.
Extends the 2D diode solver with:
- Gate oxide region (SiO2, Poisson only)
- Si/SiO2 interface (displacement continuity, no carrier flux)
- 3 contacts: source, drain (ohmic), gate (metal)

Structure (y up):
    +---------------------------+
    |        SiO2 (oxide)       |  <- gate on top
    +---------------------------+
    | n+    |   p-sub   | n+    |  <- silicon
    |source |  channel  | drain |
    +---------------------------+
"""
import numpy as np
import sys
sys.path.insert(0, "/home/hatch/workspace/tcad/package")
from tcad_stabilizer.dd_solver import (q, eps_si, Vt, ni, Dn, Dp,
                                       bernoulli, densities, mun, mup)
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu, spsolve

eps0 = 8.854187817e-12
eps_ox = 3.9 * eps0  # SiO2 permittivity

class MOSFET2D:
    def __init__(self, Nx=28, Ny_si=14, Ny_ox=4,
                 Lx=2e-6, Hsi=1e-6, tox=8e-9,
                 Na_sub=3e22, Nd_sd=1e26,
                 taun=1e-6, taup=1e-6,
                 Lch_frac=0.7,
                 x_grade=1.0, y_grade=1.0):
        """
        Nx: points in x
        Ny_si: points in silicon (y)
        Ny_ox: points in oxide (y)
        Lx: total width (m)
        Hsi: silicon depth (m)
        tox: oxide thickness (m)
        Na_sub: substrate doping (m^-3, p-type)
        Nd_sd: source/drain doping (m^-3, n+)
        Lch_frac: channel length as fraction of Lx (centered)
        x_grade: >1 clusters x points geometrically toward the two
            junctions (adaptive mesh); 1.0 = uniform (default).
            Recommended <= 1.4.
        y_grade: >1 clusters silicon y points geometrically toward
            the Si/SiO2 interface; 1.0 = uniform (default).
            Recommended <= 1.5. Stronger/combined grading can leave
            Newton hovering near the solution and reporting
            non-convergence (graceful; best state still returned) --
            see SPEED.md known limits.
        """
        self.Nx = Nx
        self.Ny_si = Ny_si
        self.Ny_ox = Ny_ox
        self.Ny = Ny_si + Ny_ox
        self.Lx, self.Hsi, self.tox = Lx, Hsi, tox
        self.N = Nx * self.Ny
        self.taun, self.taup = taun, taup
        self.Na_sub, self.Nd_sd = Na_sub, Nd_sd

        # x mesh (uniform by default; x_grade > 1 clusters points
        # geometrically toward the two junctions -- see _graded_x)
        self.x_grade = float(x_grade)
        self.y_grade = float(y_grade)
        self.hx = Lx / (Nx - 1)  # nominal spacing (row floors etc.)
        if self.x_grade != 1.0:
            self.xc = self._graded_x(Lx, Nx, Lch_frac)
        else:
            self.xc = np.linspace(0, Lx, Nx)
        self.hx_arr = np.diff(self.xc)              # per x-interval
        self._hxc = np.empty(Nx)                    # node control widths
        self._hxc[1:-1] = 0.5*(self.hx_arr[:-1] + self.hx_arr[1:])
        self._hxc[0] = self.hx_arr[0]
        self._hxc[-1] = self.hx_arr[-1]
        self._xgraded = self.x_grade != 1.0

        # y mesh: silicon (0..Hsi), oxide (Hsi..Hsi+tox)
        # Silicon: Ny_si points including y=Hsi at top
        # Oxide: Ny_ox points from above Hsi to Hsi+tox
        self.yc = np.zeros(self.Ny)
        if self.y_grade != 1.0 and Ny_si > 2:
            # Geometric grading in silicon: finest spacing at the
            # Si/SiO2 interface (top row), coarsest at the bottom.
            # The y-discretization already uses per-interval hy.
            m = Ny_si - 1
            s = (self.y_grade**np.arange(m+1) - 1.0)/(self.y_grade**m - 1.0)
            self.yc[:Ny_si] = Hsi*(1.0 - s[::-1])
        else:
            self.yc[:Ny_si] = np.linspace(0, Hsi, Ny_si)
        if Ny_ox > 0:
            # oxide points: exclude Hsi (already at Ny_si-1), include Hsi+tox
            y_ox = Hsi + np.linspace(0, tox, Ny_ox+1)[1:]
            self.yc[Ny_si:] = y_ox
        # hy array per y-interval (between iy and iy+1)
        self.hy_arr = np.diff(self.yc)  # length Ny-1

        # Region mask: True for silicon, False for oxide
        # iy = 0..Ny_si-1: silicon; iy = Ny_si..Ny-1: oxide
        self.is_si = np.zeros(self.Ny, dtype=bool)
        self.is_si[:Ny_si] = True
        self.iy_interface = Ny_si - 1  # top silicon row
        self.iy_ox_bottom = Ny_si      # bottom oxide row

        # Doping in silicon (net: + for donors, - for acceptors)
        # Source/drain n+, channel p-sub
        Lch = Lx * Lch_frac
        x_src_end = (Lx - Lch) / 2
        x_drn_start = (Lx + Lch) / 2
        # Smooth transitions
        wt = 2 * self.hx
        # n+ source: high Nd for x < x_src_end
        # p-sub: Na in channel
        # n+ drain: high Nd for x > x_drn_start
        C1d = np.zeros(Nx)
        for ix, x in enumerate(self.xc):
            # source side
            if x < x_src_end - wt/2:
                C1d[ix] = Nd_sd
            elif x < x_src_end + wt/2:
                f = (x - (x_src_end - wt/2)) / wt
                C1d[ix] = Nd_sd*(1-f) + (-Na_sub)*f
            elif x < x_drn_start - wt/2:
                C1d[ix] = -Na_sub
            elif x < x_drn_start + wt/2:
                f = (x - (x_drn_start - wt/2)) / wt
                C1d[ix] = (-Na_sub)*(1-f) + Nd_sd*f
            else:
                C1d[ix] = Nd_sd
        # Tile in y (doping uniform in y within silicon)
        self.C = np.zeros(self.N)
        for iy in range(self.Ny):
            for ix in range(Nx):
                n_ = iy*Nx + ix
                if self.is_si[iy]:
                    self.C[n_] = C1d[ix]
                else:
                    self.C[n_] = 0.0  # no doping in oxide

        # Biases
        self.Vs, self.Vd, self.Vg = 0.0, 0.0, 0.0
        self.Vfb = 0.0  # flatband voltage (can tune)
        self._update_bc()

    @staticmethod
    def _geo(a, b, n, r, refine):
        """n intervals on [a, b] with geometric spacing; `refine` picks
        the end(s) with the smallest steps: 'left', 'right' or 'both'."""
        if n <= 0:
            return np.array([a])
        if refine == "both":
            mid = 0.5*(a + b)
            nl = n//2
            left = MOSFET2D._geo(a, mid, nl, r, "right")
            right = MOSFET2D._geo(mid, b, n - nl, r, "left")
            return np.concatenate([left, right[1:]])
        s = (r**np.arange(n + 1) - 1.0)/(r**n - 1.0)  # clustered near 0
        if refine == "left":
            return a + (b - a)*s
        return a + (b - a)*(1.0 - s[::-1])              # clustered near b

    def _graded_x(self, Lx, Nx, Lch_frac):
        """x mesh clustered toward both metallurgical junctions."""
        r = self.x_grade
        Lch = Lx*Lch_frac
        xj1 = (Lx - Lch)/2
        xj2 = (Lx + Lch)/2
        ntot = Nx - 1
        lens = np.array([xj1, Lch, Lx - xj2])
        ns = np.maximum(2, np.round(ntot*lens/Lx).astype(int))
        while ns.sum() > ntot:
            ns[np.argmax(ns)] -= 1
        while ns.sum() < ntot:
            ns[np.argmax(lens)] += 1
        p1 = self._geo(0.0, xj1, int(ns[0]), r, "right")
        p2 = self._geo(xj1, xj2, int(ns[1]), r, "both")
        p3 = self._geo(xj2, Lx, int(ns[2]), r, "left")
        xc = np.concatenate([p1, p2[1:], p3[1:]])
        assert len(xc) == Nx and np.all(np.diff(xc) > 0)
        return xc

    def _update_bc(self):
        # Ohmic contacts: psi from doping
        # Source (ix=0, silicon): n+
        self.psi_s = Vt*np.arcsinh(self.C[0]/(2*ni)) + self.Vs
        # Drain (ix=Nx-1, silicon): n+
        n_drn = (self.Ny_si-1)*self.Nx + (self.Nx-1)
        self.psi_d = Vt*np.arcsinh(self.C[n_drn]/(2*ni)) + self.Vd
        # Gate (top of oxide): metal
        self.psi_g = self.Vg + self.Vfb

    def set_bias(self, Vs, Vd, Vg):
        self.Vs, self.Vd, self.Vg = Vs, Vd, Vg
        self._update_bc()

    def initial_guess(self):
        x = np.zeros(3*self.N)
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        for iy in range(self.Ny):
            for ix in range(self.Nx):
                n_ = iy*self.Nx + ix
                if self.is_si[iy]:
                    # 1D-style: equilibrium + ramp
                    psi_eq = Vt*np.arcsinh(self.C[n_]/(2*ni))
                    # lateral ramp from source to drain
                    f = self.xc[ix]/self.Lx if self._xgraded else ix/(self.Nx-1)
                    ramp = self.Vs + (self.Vd-self.Vs)*f
                    psi[n_] = psi_eq + ramp
                    phin[n_] = ramp
                    phip[n_] = ramp
                else:
                    # Oxide: linear from interface to gate
                    # (will be overwritten by Newton)
                    f_ox = (iy - self.Ny_si) / max(1, self.Ny_ox-1)
                    psi[n_] = self.psi_g * f_ox  # rough
                    phin[n_] = 0.0
                    phip[n_] = 0.0
        # Enforce Dirichlet
        self._apply_dirichlet(x)
        return x

    def _apply_dirichlet(self, x):
        """Enforce Dirichlet BCs on contacts."""
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        Nx = self.Nx
        # Source: ix=0, silicon rows
        for iy in range(self.Ny_si):
            n_ = iy*Nx + 0
            psi[n_] = self.psi_s
            phin[n_] = self.Vs
            phip[n_] = self.Vs
        # Drain: ix=Nx-1, silicon rows
        for iy in range(self.Ny_si):
            n_ = iy*Nx + (Nx-1)
            psi[n_] = self.psi_d
            phin[n_] = self.Vd
            phip[n_] = self.Vd
        # Gate: iy=Ny-1 (top oxide), all ix
        iy = self.Ny - 1
        for ix in range(Nx):
            n_ = iy*Nx + ix
            psi[n_] = self.psi_g
            # phin/phip in oxide pinned to 0 (no meaning)
            phin[n_] = 0.0
            phip[n_] = 0.0

    def is_contact(self, ix, iy):
        """Check if node is on a Dirichlet contact."""
        if iy == self.Ny-1:
            return True  # gate
        if self.is_si[iy] and (ix == 0 or ix == self.Nx-1):
            return True  # source/drain
        return False

    def residual(self, x):
        Nx, Ny = self.Nx, self.Ny
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        n, p = densities(psi, phin, phip)
        F = np.zeros(3*self.N)
        terms = np.zeros(3*self.N)
        idx = lambda ix, iy: iy*Nx + ix
        hx = self.hx
        hx_arr = self.hx_arr if self._xgraded else None

        # Precompute SG edge fluxes (silicon only for continuity,
        # but we compute everywhere and mask later)
        nxe = Ny*(Nx-1)
        nye = Nx*(Ny-1)
        Jn = np.zeros(nxe + nye)
        Jp = np.zeros(nxe + nye)
        e = 0
        # x-edges
        for iy in range(Ny):
            for ix in range(Nx-1):
                a, b = idx(ix,iy), idx(ix+1,iy)
                # Only compute flux if both nodes in silicon
                # (oxide has no carriers)
                if self.is_si[iy]:
                    dv = (psi[b]-psi[a])/Vt
                    Bp, Bm = bernoulli(dv), bernoulli(-dv)
                    hxe = hx_arr[ix] if hx_arr is not None else hx
                    Jn[e] = (q*Dn/hxe)*(n[b]*Bp - n[a]*Bm)
                    Jp[e] = (q*Dp/hxe)*(p[a]*Bp - p[b]*Bm)
                e += 1
        # y-edges (hy varies!)
        for iy in range(Ny-1):
            hy = self.hy_arr[iy]
            for ix in range(Nx):
                a, b = idx(ix,iy), idx(ix,iy+1)
                # Flux only if both in silicon (no flux into oxide)
                if self.is_si[iy] and self.is_si[iy+1]:
                    dv = (psi[b]-psi[a])/Vt
                    Bp, Bm = bernoulli(dv), bernoulli(-dv)
                    Jn[e] = (q*Dn/hy)*(n[b]*Bp - n[a]*Bm)
                    Jp[e] = (q*Dp/hy)*(p[a]*Bp - p[b]*Bm)
                # else: Jn[e]=Jp[e]=0 (no flux across Si/SiO2 or in oxide)
                e += 1

        def jx(ix, iy): return iy*(Nx-1) + ix
        def jy(ix, iy): return nxe + iy*Nx + ix

        for iy in range(Ny):
            for ix in range(Nx):
                n_ = idx(ix, iy)

                # Dirichlet contacts
                if self.is_contact(ix, iy):
                    if iy == Ny-1:  # gate
                        F[3*n_] = psi[n_] - self.psi_g
                        F[3*n_+1] = phin[n_] - 0.0
                        F[3*n_+2] = phip[n_] - 0.0
                    elif ix == 0:  # source
                        F[3*n_] = psi[n_] - self.psi_s
                        F[3*n_+1] = phin[n_] - self.Vs
                        F[3*n_+2] = phip[n_] - self.Vs
                    else:  # drain
                        F[3*n_] = psi[n_] - self.psi_d
                        F[3*n_+1] = phin[n_] - self.Vd
                        F[3*n_+2] = phip[n_] - self.Vd
                    terms[3*n_:3*n_+3] = 1.0
                    continue

                if not self.is_si[iy]:
                    # OXIDE: Poisson only (Laplace, no charge)
                    # phin/phip pinned to 0
                    if not (0 < ix < Nx-1):
                        lapx = 0.0
                    elif hx_arr is not None:
                        lapx = ((psi[idx(ix+1,iy)]-psi[n_])/hx_arr[ix]
                                - (psi[n_]-psi[idx(ix-1,iy)])/hx_arr[ix-1]) \
                               / self._hxc[ix]
                    else:
                        lapx = (psi[idx(ix+1,iy)] - 2*psi[n_] + psi[idx(ix-1,iy)])/hx**2
                    # y-Laplacian with variable hy
                    if iy == Ny-1:
                        # top (gate) - Dirichlet, handled above, shouldn't reach here
                        lapy = 0.0
                    elif iy == self.Ny_si:  # bottom oxide (interface)
                        # Interface: flux continuity
                        # d/dy(eps*dpsi/dy) at interface
                        # Below: silicon, above: oxide
                        hy_below = self.hy_arr[iy-1]  # Si side
                        hy_above = self.hy_arr[iy]    # oxide side
                        # Flux from below (Si): eps_si*(psi_int - psi_si_below)/hy_below
                        # Flux to above (ox): eps_ox*(psi_ox_above - psi_int)/hy_above
                        # Divergence: (F_above - F_below) / hy_avg
                        psi_below = psi[idx(ix, iy-1)]
                        psi_above = psi[idx(ix, iy+1)] if iy+1 < Ny else psi[n_]
                        F_below = eps_si*(psi[n_]-psi_below)/hy_below
                        F_above = eps_ox*(psi_above-psi[n_])/hy_above
                        hy_avg = 0.5*(hy_below+hy_above)
                        lapy = (F_above - F_below)/hy_avg / eps_ox  # normalize by eps_ox
                        # Actually for Poisson: d/dy(eps*dpsi/dy) = 0 in oxide
                        # So (F_above - F_below)/hy_avg = 0
                        # We want F = (F_above - F_below)/hy_avg (without /eps)
                        lapy = (F_above - F_below)/hy_avg
                        F[3*n_] = lapx*eps_ox + lapy  # scale lapx by eps_ox for consistency
                        # Hmm, need consistent scaling. Let me do:
                        # F = d/dx(eps*dpsi/dx) + d/dy(eps*dpsi/dy) = 0
                        # d/dx term: eps_ox * lapx
                        # d/dy term: (F_above - F_below)/hy_avg
                        F[3*n_] = eps_ox*lapx + (F_above - F_below)/hy_avg
                        terms[3*n_] = abs(eps_ox*lapx) + (abs(F_above)+abs(F_below))/hy_avg + 1e-20
                    else:
                        # Interior oxide
                        hy_below = self.hy_arr[iy-1]
                        hy_above = self.hy_arr[iy]
                        psi_below = psi[idx(ix, iy-1)]
                        psi_above = psi[idx(ix, iy+1)]
                        F_below = eps_ox*(psi[n_]-psi_below)/hy_below
                        F_above = eps_ox*(psi_above-psi[n_])/hy_above
                        hy_avg = 0.5*(hy_below+hy_above)
                        F[3*n_] = eps_ox*lapx + (F_above - F_below)/hy_avg
                        terms[3*n_] = abs(eps_ox*lapx) + (abs(F_above)+abs(F_below))/hy_avg + 1e-20
                    # Pin phin/phip
                    F[3*n_+1] = phin[n_]
                    F[3*n_+2] = phip[n_]
                    terms[3*n_+1] = 1.0
                    terms[3*n_+2] = 1.0
                    continue

                # SILICON interior
                # Poisson with interface handling at top row
                if hx_arr is not None:
                    lapx = ((psi[idx(ix+1,iy)]-psi[n_])/hx_arr[ix]
                            - (psi[n_]-psi[idx(ix-1,iy)])/hx_arr[ix-1]) \
                           / self._hxc[ix]
                else:
                    lapx = (psi[idx(ix+1,iy)] - 2*psi[n_] + psi[idx(ix-1,iy)])/hx**2
                if iy == 0:
                    # Bottom Neumann
                    hy = self.hy_arr[0]
                    lapy = 2*(psi[idx(ix,1)] - psi[n_])/hy**2
                    F_poisson = eps_si*(lapx + lapy) + q*(p[n_]-n[n_]+self.C[n_])
                    terms_p = eps_si*(abs(lapx)+abs(lapy)) + q*(p[n_]+n[n_]+abs(self.C[n_]))
                elif iy == self.iy_interface:
                    # Top silicon (interface with oxide)
                    # d/dy(eps*dpsi/dy): below is Si, above is oxide
                    hy_below = self.hy_arr[iy-1]
                    hy_above = self.hy_arr[iy]
                    psi_below = psi[idx(ix, iy-1)]
                    psi_above = psi[idx(ix, iy+1)]  # oxide node
                    F_below = eps_si*(psi[n_]-psi_below)/hy_below
                    F_above = eps_ox*(psi_above-psi[n_])/hy_above
                    hy_avg = 0.5*(hy_below+hy_above)
                    ddy = (F_above - F_below)/hy_avg
                    F_poisson = eps_si*lapx + ddy + q*(p[n_]-n[n_]+self.C[n_])
                    terms_p = abs(eps_si*lapx) + (abs(F_above)+abs(F_below))/hy_avg + q*(p[n_]+n[n_]+abs(self.C[n_]))
                else:
                    # Interior silicon
                    hy_below = self.hy_arr[iy-1]
                    hy_above = self.hy_arr[iy]
                    # For uniform hy, simple Laplacian; for non-uniform, flux form
                    if abs(hy_below-hy_above) < 1e-20:
                        lapy = (psi[idx(ix,iy+1)] - 2*psi[n_] + psi[idx(ix,iy-1)])/hy_below**2
                    else:
                        psi_below = psi[idx(ix, iy-1)]
                        psi_above = psi[idx(ix, iy+1)]
                        F_below = eps_si*(psi[n_]-psi_below)/hy_below
                        F_above = eps_si*(psi_above-psi[n_])/hy_above
                        hy_avg = 0.5*(hy_below+hy_above)
                        lapy = (F_above - F_below)/hy_avg/eps_si
                    F_poisson = eps_si*(lapx + lapy) + q*(p[n_]-n[n_]+self.C[n_])
                    terms_p = eps_si*(abs(lapx)+abs(lapy)) + q*(p[n_]+n[n_]+abs(self.C[n_]))
                F[3*n_] = F_poisson
                terms[3*n_] = terms_p + 1e-20

                # Continuity (silicon only, no flux into oxide)
                hxc = self._hxc[ix] if hx_arr is not None else hx
                dJn = (Jn[jx(ix,iy)]-Jn[jx(ix-1,iy)])/hxc
                dJp = (Jp[jx(ix,iy)]-Jp[jx(ix-1,iy)])/hxc
                # y-flux divergence. Uniform mesh: original per-face
                # form (bit-identical). Graded mesh: proper finite-volume
                # form (J_top - J_bot)/h_cv with h_cv the control height
                # -- the per-face form is inconsistent when hy varies.
                if hx_arr is not None or self.y_grade != 1.0:
                    Jtn = Jtp = Jbn = Jbp = 0.0
                    if iy < self.iy_interface:
                        Jtn = Jn[jy(ix,iy)]; Jtp = Jp[jy(ix,iy)]
                    if iy > 0:
                        Jbn = Jn[jy(ix,iy-1)]; Jbp = Jp[jy(ix,iy-1)]
                    if iy == 0:
                        hcv = self.hy_arr[0]
                    elif iy == self.iy_interface:
                        hcv = self.hy_arr[iy-1]
                    else:
                        hcv = 0.5*(self.hy_arr[iy-1] + self.hy_arr[iy])
                    dJn += (Jtn - Jbn)/hcv
                    dJp += (Jtp - Jbp)/hcv
                else:
                    # y-flux: only include if neighbor in silicon
                    if iy < self.iy_interface:
                        # top neighbor in silicon
                        hy = self.hy_arr[iy]
                        dJn += Jn[jy(ix,iy)]/hy
                        dJp += Jp[jy(ix,iy)]/hy
                    # else: top is oxide, no flux (already zero in Jn/Jp)
                    if iy > 0:
                        hy = self.hy_arr[iy-1]
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

        5-point stencil (oxide rows included: they couple only to
        nearest neighbours) => columns of nodes with (ix%3, iy%3) equal
        never share a row: 9 colors x 3 vars = 27 residual evals per
        build, independent of mesh size (dense FD needs 3*N evals).
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
        """Fast mode: coarse-but-valid mesh preset (14 x (8+3)).

        This is the mesh the full test suite is validated on: same
        physics, textbook curves, roughly 3.3x fewer unknowns than the
        default 28 x (14+4) mesh. Use the default constructor for
        maximum accuracy.
        """
        kw.setdefault("Nx", 14)
        kw.setdefault("Ny_si", 8)
        kw.setdefault("Ny_ox", 3)
        return cls(**kw)

    def newton_solve(self, x0, hook=None, tol=1e-6, max_iter=60, jac_reuse=3):
        """Newton solver with hook interface (same as diode).

        Linear algebra (v0.6): sparse colored FD Jacobian + splu, with
        quasi-Newton reuse -- the factorization is reused for up to
        jac_reuse iterations and rebuilt early if the residual stalls.
        The iteration matrix never changes the converged solution:
        the fixed point is F(x)=0 and the convergence test is unchanged.
        """
        x = x0.copy()
        n = len(x)
        # Per-row floors
        Cmax = max(1.0, float(np.max(np.abs(self.C))))
        Ap = max((q/eps_si)*Cmax, Vt/self.hx**2)
        An = q*mun*Vt*Cmax/self.hx**2
        Ah = q*mup*Vt*Cmax/self.hx**2
        arow = np.ones(3*self.N)
        for iy in range(self.Ny):
            for ix in range(self.Nx):
                nn = iy*self.Nx + ix
                if self.is_contact(ix, iy):
                    continue
                if self.is_si[iy]:
                    arow[3*nn] = Ap
                    arow[3*nn+1] = An
                    arow[3*nn+2] = Ah
                else:
                    arow[3*nn] = Vt/self.hy_arr[min(iy, len(self.hy_arr)-1)]**2
                    # phin/phip pinned
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
            if update < 1e-10 and honest < 0.5:
                converged = True
                break
            state = {"version":"1","iteration":k,"x":x.copy(),"delta":delta.copy(),
                     "residual":F.copy(),"residual_norm":honest,"row_scale":s,
                     "mesh":{"Nx":self.Nx,"Ny":self.Ny},"edge":{},"solver":self}
            corr = hook(state) if hook else {}
            a = corr.get("alpha", 1.0)
            a = float(np.min(a)) if np.ndim(a) else float(a)
            x = x + a*delta
            prev_step = a*update
        rx = x if converged else best["x"]
        return {"converged":converged,"diverged":diverged,
                "iterations":len(hist),"residual_history":hist,"x":rx,
                "jac_builds":jac_builds}

    def drain_current(self, x):
        """Drain current (A/m) integrated over silicon depth at drain."""
        psi, phin, phip = x[0::3], x[1::3], x[2::3]
        n, p = densities(psi, phin, phip)
        Nx = self.Nx
        idx = lambda ix, iy: iy*Nx + ix
        hx = self.hx
        Itot = 0.0
        ix = Nx-2  # edge between Nx-2 and Nx-1 (drain)
        hxe = self.hx_arr[ix] if self._xgraded else hx
        for iy in range(self.Ny_si):
            # y-weight: hy around this row
            if iy == 0:
                wy = self.hy_arr[0]/2
            elif iy == self.Ny_si-1:
                wy = self.hy_arr[iy-1]/2
            else:
                wy = (self.hy_arr[iy-1]+self.hy_arr[iy])/2
            a, b = idx(ix,iy), idx(ix+1,iy)
            dv = (psi[b]-psi[a])/Vt
            Bp, Bm = bernoulli(dv), bernoulli(-dv)
            Jn = (q*Dn/hxe)*(n[b]*Bp - n[a]*Bm)
            Jp = (q*Dp/hxe)*(p[a]*Bp - p[b]*Bm)
            Itot += (Jn+Jp)*wy
        return float(Itot)
