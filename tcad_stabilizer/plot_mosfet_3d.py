"""Cross-section plots of the 3D MOSFET (x-y plane at mid-width).

Solves equilibrium, ramps to the on-state (Vd=1V, Vg=2V) by
continuation, and saves mosfet_3d_xsec.png with 4 panels:
potential (eq), potential (on), log10 electron density (on),
transfer curve Id/W vs Vg at Vd=0.1V.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, sys
sys.path.insert(0, "/home/hatch/workspace/tcad/package")
sys.path.insert(0, "/home/hatch/workspace/tcad/phase3_3d")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mosfet_3d import MOSFET3D
from tcad_stabilizer import make_stabilizer

STAB = make_stabilizer("classical")
m = MOSFET3D(Nx=14, Ny_si=8, Ny_ox=3, Nz=4)

m.set_bias(0, 0, 0)
x = m.newton_solve(m.initial_guess(), hook=STAB, max_iter=40)["x"]
psi_eq = m.cross_section(x)

# Continuation ramp; record transfer points at Vd=0.1
transfer = []
for (Vd, Vg) in [(0.1, 0.0), (0.1, 0.5), (0.1, 1.0), (0.1, 2.0),
                 (0.5, 2.0), (1.0, 2.0)]:
    m.set_bias(0, Vd, Vg)
    r = m.newton_solve(x, hook=STAB, max_iter=40)
    assert r["converged"], f"no conv at Vd={Vd} Vg={Vg}"
    x = r["x"]
    if Vd == 0.1:
        transfer.append((Vg, abs(m.drain_current_per_width(x))))
    print(f"Vd={Vd} Vg={Vg}: Id/W={abs(m.drain_current_per_width(x)):.3e} A/m",
          flush=True)

Id_on = abs(m.drain_current(x))
fields = m.get_fields(x)
iz = m.Nz // 2
psi_on = fields["psi"][iz]
logn = np.log10(np.maximum(fields["n"][iz], 1.0))
logn[m.Ny_si:, :] = np.nan  # no carriers in oxide - mask formal values

# Cell edges from node coords (correct non-uniform y geometry)
def edges(c):
    e = np.empty(len(c) + 1)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - 0.5 * (c[1] - c[0])
    e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return e

xe = edges(m.xc) * 1e6            # um
ye = edges(m.yc) * 1e9            # nm
yint = m.Hsi * 1e9                # Si/SiO2 interface (nm)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, C, title, lab in [
        (axes[0, 0], psi_eq, "Potential - equilibrium", "psi (V)"),
        (axes[0, 1], psi_on, "Potential - on-state (Vd=1V, Vg=2V)", "psi (V)"),
        (axes[1, 0], logn, "Electron density - on-state", "log10 n (m^-3)")]:
    im = ax.pcolormesh(xe, ye, C, shading="flat", cmap="RdYlBu_r")
    ax.axhline(yint, color="k", lw=1, ls="--", alpha=0.7)
    ax.text(0.03, 0.90, "gate oxide above line", transform=ax.transAxes,
            fontsize=8)
    ax.set_xlabel("x (um)  source -> drain")
    ax.set_ylabel("y (nm)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=lab)

ax = axes[1, 1]
Vg_t = [t[0] for t in transfer]
Id_t = [t[1] for t in transfer]
ax.semilogy(Vg_t, Id_t, "o-", lw=2)
ax.set_xlabel("Vgs (V)")
ax.set_ylabel("Id per width (A/m)")
ax.set_title("Transfer curve at Vd=0.1V (3D MOSFET)")
ax.grid(True, which="both", alpha=0.3)

fig.suptitle(f"3D MOSFET (mid-width x-y plane) - on-state Id = {Id_on:.2e} A "
             f"for W = {m.Lz*1e6:.0f} um")
fig.tight_layout()
out = "/home/hatch/workspace/tcad/phase3_3d/mosfet_3d_xsec.png"
fig.savefig(out, dpi=120)
print("saved", out)
