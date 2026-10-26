"""TCAD Stabilizer v0.6 — Newton stabilization hooks for drift-diffusion solvers.
1D/2D/3D diodes, 2D MOSFET, and 3D MOSFET.
v0.6: sparse colored Jacobians + gated reuse in the 2D solvers (20-70x
faster, identical results), MOSFET adaptive mesh options, fast modes.
See SPEED.md and SWEEP.md."""
from .stabilizer import make_stabilizer, solve_with_fallback
from .hook_interface import VERSION as HOOK_VERSION
try:
    from .dd_solver_2d import DDSolver2D as Diode2D
    _has_2d = True
except ImportError:
    _has_2d = False
try:
    from .mosfet_2d import MOSFET2D
    _has_mosfet = True
except ImportError:
    _has_mosfet = False
try:
    from .diode_3d import DDSolver3D
    _has_3d = True
except ImportError:
    _has_3d = False
try:
    from .mosfet_3d import MOSFET3D
    _has_mosfet3d = True
except ImportError:
    _has_mosfet3d = False

__version__ = "0.6.0"
__all__ = ["make_stabilizer", "solve_with_fallback", "HOOK_VERSION"]
if _has_2d:
    __all__.append("Diode2D")
if _has_mosfet:
    __all__.append("MOSFET2D")
if _has_3d:
    __all__.append("DDSolver3D")
if _has_mosfet3d:
    __all__.append("MOSFET3D")
