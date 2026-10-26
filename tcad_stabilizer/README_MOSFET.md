# 2D MOSFET Simulator

A 2D n-channel MOSFET drift-diffusion solver.

## Usage

```python
from mosfet_2d import MOSFET2D
from tcad_stabilizer import make_stabilizer

# Create MOSFET
m = MOSFET2D(Nx=14, Ny_si=8, Ny_ox=3)

# Solve equilibrium
m.set_bias(Vs=0, Vd=0, Vg=0)
x = m.newton_solve(m.initial_guess(),
                   hook=make_stabilizer("classical"))["x"]

# Sweep gate voltage
for Vgs in [0, 1, 2]:
    m.set_bias(Vs=0, Vd=0.1, Vg=Vgs)
    r = m.newton_solve(x, hook=make_stabilizer("classical"))
    x = r["x"]
    Id = m.drain_current(x)
    print(f"Vgs={Vgs}V: Id={Id:.3e} A/m")
```

## Device Structure

- Source (n+, left), Drain (n+, right), Gate (top, over oxide)
- Gate oxide: 8nm SiO2 (Poisson only, no carriers)
- Channel: p-type substrate (Na=3e22 m^-3)
- Source/drain: n+ (Nd=1e26 m^-3)

## Physics

- Drift-diffusion with Scharfetter-Gummel discretization
- Newton-Raphson with hook interface for stabilizer
- Si/SiO2 interface: displacement continuity, no carrier flux into oxide

## Tests

Run `python3 test_mosfet_2d.py` — all 6 tests must PASS:
1. Equilibrium converges
2. Threshold voltage physical (~0.9V)
3. Id-Vgs correct (off below Vt, on above)
4. Id-Vds linear then saturation
5. Subthreshold slope ~60-120 mV/dec
6. No NaN, physical values
