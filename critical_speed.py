import numpy as np
import matplotlib.pyplot as plt

from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.soil_dispersion import Layer, SoilDispersion
from TrainCritSpeed.critical_speed import CriticalSpeed

omega = np.linspace(0.001, 500, 100)

ballast_parameters = BallastTrackParameters(EI_rail=6.4e6,
                                                m_rail=60.21,
                                                k_rail_pad=6e8,
                                                c_rail_pad=2.5e5,
                                                m_sleeper=183.33333333333334,
                                                E_ballast=84098850.12359521,
                                                h_ballast=0.6,
                                                width_sleeper=1.25,
                                                soil_stiffness=0.0,
                                                rho_ballast=1881.6928724461318)

soil_layers = [
    Layer(density=1894.23409072766, young_modulus=11776.449940926, poisson_ratio=0.378098850322639, thickness=1.1),
    Layer(density=1021.25953873067, young_modulus=28784, poisson_ratio=0.495, thickness=0.6),
    Layer(density=1207.01327252653, young_modulus=25730, poisson_ratio=0.495, thickness=5),
    Layer(density=1701.92303296991, young_modulus=534796, poisson_ratio=0.495, thickness=3.3),
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
