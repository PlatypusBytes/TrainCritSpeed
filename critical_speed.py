import numpy as np
import matplotlib.pyplot as plt

from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.soil_dispersion import Layer, SoilDispersion
from TrainCritSpeed.critical_speed import CriticalSpeed


omega = np.linspace(0.1, 250, 100)

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

soil_layers = [
    Layer(density=1900, young_modulus=2.67e7, poisson_ratio=0.33, thickness=5),
    Layer(density=1900, young_modulus=1.14e8, poisson_ratio=0.33, thickness=10),
    Layer(density=1900, young_modulus=2.63e8, poisson_ratio=0.33, thickness=15),
    Layer(density=1900, young_modulus=4.71e8, poisson_ratio=0.33, thickness=np.inf),
]

ballast = BallastedTrack(ballast_parameters, omega)
dispersion = SoilDispersion(soil_layers, omega)

cs = CriticalSpeed(omega, ballast, dispersion)
cs.compute()
print(f"Critical speed: {cs.critical_speed} m/s")
print(cs.frequency)
plt.figure(figsize=(10, 6))
plt.plot(omega, ballast.phase_velocity, label="Ballast Track")
plt.plot(omega, dispersion.phase_velocity, label="Soil Layers")
plt.plot(cs.frequency, cs.critical_speed, "ro", label="Critical Speed")
plt.xlabel("Angular frequency [rad/s]")
plt.ylabel("Phase speed [m/s]")
plt.grid()
plt.legend()
plt.show()