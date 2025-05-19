import numpy as np
from TrainCritSpeed.cs_stochastic import StochasticCriticalSpeed, StochasticSoilDispersion, StochasticLayer
from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.soil_dispersion import Layer

soil_layers = [
    StochasticLayer(density_mean=1900,
                    young_mean=3e7,
                    poisson_mean=0.33,
                    thickness_mean=5,
                    density_std=100,
                    young_std=5e6,
                    poisson_std=0.05,
                    thickness_std=0.2,
                    distribution="normal",
                    seed=123),
    StochasticLayer(density_mean=1900,
                    young_mean=1e8,
                    poisson_mean=0.33,
                    thickness_mean=10,
                    density_std=200,
                    young_std=1e7,
                    poisson_std=0.05,
                    thickness_std=1,
                    distribution="normal",
                    seed=123),
    Layer(density=1900, young_modulus=3e8, poisson_ratio=0.4, thickness=15),
    StochasticLayer(density_mean=1900,
                    young_mean=5e8,
                    poisson_mean=0.33,
                    thickness_mean=np.inf,
                    density_std=400,
                    young_std=2e8,
                    poisson_std=0.1,
                    distribution="normal",
                    seed=123)
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

speeds = []
frequencies = []

for k in range(10):
    cs.compute()
    print(f"Critical speed: {cs.critical_speed} m/s at a frequency of {cs.frequency}")
    speeds.append(cs.critical_speed)
    frequencies.append(cs.frequency)
    cs.soil.realise()

print(f"Mean critical speed: {np.mean(speeds)} m/s with a standard deviation of {np.std(speeds)} m/s")
