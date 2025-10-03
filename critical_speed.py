from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.soil_dispersion import Layer, SoilDispersion
from TrainCritSpeed.critical_speed import CriticalSpeed

omega = np.linspace(0.1, 400, 100)
number_modes = 5

ballast_parameters = BallastTrackParameters(EI_rail=1.29e7,
                                            m_rail=120,
                                            k_rail_pad=5e8,
                                            c_rail_pad=2.5e5,
                                            m_sleeper=490,
                                            E_ballast=130e6,
                                            h_ballast=0.5,
                                            width_sleeper=1.25,
                                            soil_stiffness=0.0,
                                            rho_ballast=1700)

soil_layers = [
    Layer(density=2000, young_modulus=30e6, poisson_ratio=0.35, thickness=2),
    Layer(density=2000, young_modulus=40e6, poisson_ratio=0.35, thickness=4),
    Layer(density=2000, young_modulus=75e6, poisson_ratio=0.40, thickness=np.inf),
]

ballast = BallastedTrack(ballast_parameters, omega)
dispersion = SoilDispersion(soil_layers, omega, nb_modes=number_modes)
cs = CriticalSpeed(omega, ballast, dispersion)
cs.compute()
print(f"Critical speed: {cs.critical_speed} m/s")

# Optional: plot the 2D dispersion figure
print(f"Creating 2D dispersion figure")
dispersion.soil_dispersion_image(file_name=Path("results/soil_dispersion.png"))

# Plot the dispersion curves and critical speed
plt.figure(figsize=(10, 6))
plt.plot(omega/ 2 / np.pi, ballast.phase_velocity, label="Ballast Track")
for i in range(number_modes):
    plt.plot(omega/ 2 / np.pi, dispersion.phase_velocity[:, i], label=f"Soil Mode {i+1}")
plt.plot(cs.frequency/ 2 / np.pi, cs.critical_speed, "ro", label="Critical Speed")
plt.xlim(0, 60)
plt.ylim(0, 160)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase speed [m/s]")
plt.grid()
plt.legend()
plt.savefig("results/critical_speed.png")
plt.close()

