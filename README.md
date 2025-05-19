# TrainCritSpeed

A Python package for analyzing critical speeds in railway systems, focusing on soil and track dispersion analysis.

The methodology for the computation of the critical train speed is based on the work of [Mezher et al. (2016)](https://www.sciencedirect.com/science/article/abs/pii/S2214391215000239).
The dispersion curve for the layered soil is based on the Fast Delta Matrix method proposed by [Buchen and Ben-Hador (1996)](https://academic.oup.com/gji/article-lookup/doi/10.1111/j.1365-246X.1996.tb05642.x).

## Installation

To install the TrainCritSpeed you can use pip:

```bash
pip install git+https://github.com/PlatypusBytes/TrainCritSpeed
```

## Usage Examples

### Soil Dispersion Analysis

Analyze the dispersion characteristics of a multi-layered soil:

```python
import numpy as np
import matplotlib.pyplot as plt
from TrainCritSpeed.soil_dispersion import Layer, SoilDispersion

# Define soil layers with properties
soil_layers = [
    Layer(density=1900, young_modulus=2.67e7, poisson_ratio=0.33, thickness=5),
    Layer(density=1900, young_modulus=1.14e8, poisson_ratio=0.33, thickness=10),
    Layer(density=1900, young_modulus=2.63e8, poisson_ratio=0.33, thickness=15),
    Layer(density=1900, young_modulus=4.71e8, poisson_ratio=0.33, thickness=np.inf),
]

# Calculate dispersion across frequency range
omegas = np.linspace(1, 50 * 2 * np.pi, 100)
dispersion = SoilDispersion(soil_layers, omegas)
dispersion.soil_dispersion()

# Plot results
plt.plot(omegas / 2 / np.pi, dispersion.phase_velocity)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase velocity [m/s]')
plt.show()
```

### Track Dispersion Analysis

Compare dispersion characteristics of ballasted and slab tracks:

```python
from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.track_dispersion import SlabTrack, SlabTrackParameters

# Define frequency range
omega = np.linspace(0.1, 250, 100)

# Configure and analyze ballasted track
ballast_parameters = BallastTrackParameters(
    EI_rail=1.29e7,
    m_rail=120,
    k_rail_pad=5e8,
    c_rail_pad=2.5e5,
    m_sleeper=490,
    E_ballast=130e6,
    h_ballast=0.35,
    width_sleeper=1.25,
    soil_stiffness=0.0,
    rho_ballast=1700
)
ballast = BallastedTrack(ballast_parameters, omega)
ballast.track_dispersion()

# Configure and analyze slab track
slab_parameters = SlabTrackParameters(
    EI_rail=1.29e7,
    m_rail=120,
    k_rail_pad=5e8,
    c_rail_pad=2.5e5,
    EI_slab=30e9 * (1.25 * 0.35**3 / 12),
    m_slab=2500 * 1.25 * 0.35,
    soil_stiffness=0.0,
)
slab = SlabTrack(slab_parameters, omega)
slab.track_dispersion()

# Compare results
plt.plot(omega / 2 / np.pi, ballast.phase_velocity, label="Ballast Track")
plt.plot(omega / 2 / np.pi, slab.phase_velocity, label="Slab Track")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase speed [m/s]")
plt.legend()
plt.show()
```

### Critical Speed Calculation

Calculate the critical speed on a ballasted track:

```python
import numpy as np
import matplotlib.pyplot as plt
from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.soil_dispersion import Layer, SoilDispersion
from TrainCritSpeed.critical_Speed import CriticalSpeed

# Define frequency range
omega = np.linspace(0.1, 250, 100)

# Configure ballasted track parameters
ballast_parameters = BallastTrackParameters(
    EI_rail=1.29e7,
    m_rail=120,
    k_rail_pad=5e8,
    c_rail_pad=2.5e5,
    m_sleeper=490,
    E_ballast=130e6,
    h_ballast=0.35,
    width_sleeper=1.25,
    soil_stiffness=0.0,
    rho_ballast=1700
)

# Define soil layers with properties
soil_layers = [
    Layer(density=1900, young_modulus=2.67e7, poisson_ratio=0.33, thickness=5),
    Layer(density=1900, young_modulus=1.14e8, poisson_ratio=0.33, thickness=10),
    Layer(density=1900, young_modulus=2.63e8, poisson_ratio=0.33, thickness=15),
    Layer(density=1900, young_modulus=4.71e8, poisson_ratio=0.33, thickness=np.inf),
]

# Calculate dispersion for track and soil
ballast = BallastedTrack(ballast_parameters, omega)
dispersion = SoilDispersion(soil_layers, omega)

# Compute critical speed
cs = CriticalSpeed(omega, ballast, dispersion)
cs.compute()
print(f"Critical speed: {cs.critical_speed} m/s")

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(omega, ballast.phase_velocity, label="Ballast Track")
plt.plot(omega, dispersion.phase_velocity, label="Soil Layers")
plt.plot(cs.frequency, cs.critical_speed, "ro", label="Critical Speed")
plt.xlabel("Angular frequency [rad/s]")
plt.ylabel("Phase speed [m/s]")
plt.grid()
plt.legend()
plt.show()
```
### Stochastic analysis
You can perform a stochastic analysis using StochasticLayer

```
import numpy as np
from TrainCritSpeed.cs_stochastic import StochasticCriticalSpeed, StochasticSoilDispersion, StochasticLayer
from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.soil_dispersion import Layer

soil_layers = [
    StochasticLayer(1900,100,3e7,5e6,0.33,0.05,5,0.2),
    StochasticLayer(1900,200,1e8,1e7,0.33,0.05,10,1),
    Layer(1900,3e8,0.4,15),
    StochasticLayer(1900,400,5e8,2e8,0.33,0.1,np.inf,0)
]

omega = np.linspace(0.1, 250, 100)

#replace with import from file
ballast_parameters = BallastTrackParameters(EI_rail=1.29e7,
                                            m_rail=120,
                                            k_rail_pad=5e8,
                                            c_rail_pad=2.5e5,
                                            m_sleeper=490,
                                            E_ballast=130e6,
                                            h_ballast=0.35,
                                            width_sleeper=1.25,
                                            soil_stiffness=0.0,
                                            rho_ballast=1700)

ballast = BallastedTrack(ballast_parameters, omega)

dispersion = StochasticSoilDispersion(soil_layers, omega)

cs = StochasticCriticalSpeed(omega, ballast, dispersion)
cs.track.track_dispersion()

speeds=[]
frequencies=[]

for k in range(10):
    cs.soil.realise()
    cs.compute()
    print(f"Critical speed: {cs.critical_speed} m/s at a frequency of {cs.frequency}")
    speeds.append(cs.critical_speed)
    frequencies.append(cs.frequency)

print(f"Mean critical speed: {np.mean(speeds)} m/s with a standard deviation of {np.std(speeds)} m/s")

```