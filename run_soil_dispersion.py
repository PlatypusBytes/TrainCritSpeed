import numpy as np
import matplotlib.pyplot as plt

from TrainCritSpeed.soil_dispersion import Layer, SoilDispersion




def compute_elastic_parameters(vs, vp, density=1900):
    nu = (vp**2 - 2 * vs**2) / (2 * (vp**2 - vs**2))
    G = density * vs**2
    E = 2 * G * (1 + nu)
    return E, nu


E1, nu1 = compute_elastic_parameters(100, 200, 1900)
E2, nu2 = compute_elastic_parameters(200, 400, 1900)
E3, nu3 = compute_elastic_parameters(300, 600, 1900)
E4, nu4 = compute_elastic_parameters(400, 800, 1900)


soil_layers = [
    Layer(density=1900, young_modulus=E1, poisson_ratio=nu1, thickness=5),
    Layer(density=1900, young_modulus=E2, poisson_ratio=nu2, thickness=10),
    Layer(density=1900, young_modulus=E3, poisson_ratio=nu3, thickness=15),
    Layer(density=1900, young_modulus=E4, poisson_ratio=nu4, thickness=np.inf),
             ]


omegas = np.linspace(1, 50 * 2 * np.pi, 100)
dispersion = SoilDispersion(soil_layers, omegas)

dispersion.soil_dispersion()

plt.plot(omegas / 2 / np.pi, dispersion.phase_velocity, marker="o")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase velocity [m/s]')
plt.show()