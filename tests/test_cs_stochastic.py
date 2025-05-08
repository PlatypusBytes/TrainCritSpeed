from TrainCritSpeed.cs_stochastic import StochasticCriticalSpeed, StochasticSoilDispersion, StochasticLayer
import numpy as np
from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.soil_dispersion import Layer


def test_stochastic():
    #regular layer inbetween to test whether the functionality of mixing regular layers didn't break
    soil_layers = [
        StochasticLayer(density_mean=1900,
                        young_mean=3e7,
                        poisson_mean=0.33,
                        thickness_mean=5,
                        density_std=100,
                        young_std=5e6,
                        poisson_std=0.05,
                        thickness_std=0.2,
                        seed=2025),
        StochasticLayer(density_mean=1900,
                        young_mean=1e8,
                        poisson_mean=0.33,
                        thickness_mean=10,
                        density_std=200,
                        young_std=1e7,
                        poisson_std=0.05,
                        thickness_std=1,
                        seed=2025),
        Layer(density=1900, young_modulus=3e8, poisson_ratio=0.4, thickness=15),
        StochasticLayer(density_mean=1900,
                        young_mean=5e8,
                        poisson_mean=0.33,
                        thickness_mean=np.inf,
                        density_std=400,
                        young_std=2e8,
                        poisson_std=0.1,
                        seed=2025)
    ]

    omega = np.linspace(0.1, 250, 100)

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

    #run track dispersion only once because it's treated deterministically
    cs.track.track_dispersion()

    cs.compute()
    print(f"Critical speed: {cs.critical_speed} m/s at a frequency of {cs.frequency}")

    np.testing.assert_almost_equal(cs.critical_speed, [81.4493569])
    np.testing.assert_almost_equal(cs.frequency, [68.36840905])

    cs.soil.realise()
    cs.compute()
    print(f"Critical speed: {cs.critical_speed} m/s at a frequency of {cs.frequency}")

    np.testing.assert_almost_equal(cs.critical_speed, [85.47332512])
    np.testing.assert_almost_equal(cs.frequency, [75.39105381])


def test_stochastic_log():
    soil_layers = [
        StochasticLayer(density_mean=1900,
                        young_mean=3e7,
                        poisson_mean=0.33,
                        thickness_mean=5,
                        density_std=100,
                        young_std=5e6,
                        poisson_std=0.05,
                        thickness_std=0.2,
                        distribution="lognormal",
                        seed=123),
        StochasticLayer(density_mean=1900,
                        young_mean=1e8,
                        poisson_mean=0.33,
                        thickness_mean=10,
                        density_std=200,
                        young_std=1e7,
                        poisson_std=0.05,
                        thickness_std=1,
                        distribution="lognormal",
                        seed=123),
        Layer(density=1900, young_modulus=3e8, poisson_ratio=0.4, thickness=15),
        StochasticLayer(density_mean=1900,
                        young_mean=5e8,
                        poisson_mean=0.33,
                        thickness_mean=np.inf,
                        density_std=400,
                        young_std=2e8,
                        poisson_std=0.1,
                        distribution="lognormal",
                        seed=123)
    ]

    omega = np.linspace(0.1, 250, 100)

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

    #run track dispersion only once because it's treated deterministically
    cs.track.track_dispersion()

    cs.compute()
    print(f"Critical speed: {cs.critical_speed} m/s at a frequency of {cs.frequency}")

    np.testing.assert_almost_equal(cs.critical_speed, [76.28127655])
    np.testing.assert_almost_equal(cs.frequency, [59.88868051])

    cs.soil.realise()
    cs.compute()
    print(f"Critical speed: {cs.critical_speed} m/s at a frequency of {cs.frequency}")

    np.testing.assert_almost_equal(cs.critical_speed, [77.49354216])
    np.testing.assert_almost_equal(cs.frequency, [61.82650532])
