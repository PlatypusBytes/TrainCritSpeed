from TrainCritSpeed.cs_stochastic import StochasticCriticalSpeed, StochasticSoilDispersion, StochasticLayer
import numpy as np
from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.soil_dispersion import Layer

def test_stochastic():
    #regular layer inbetween to test whether the functionality of mixing regular layers didn't break
    soil_layers = [
        StochasticLayer(densitymean=1900, youngmean=3e7, poissonmean=0.33,thicknessmean=5,
                         densitystd=100, youngstd=5e6, poissonstd=0.05, thicknessstd=0.2),
        StochasticLayer(densitymean=1900, youngmean=1e8, poissonmean=0.33,thicknessmean=10,
                         densitystd=200, youngstd=1e7, poissonstd=0.05, thicknessstd=1),
        Layer(density=1900, young_modulus=3e8, poisson_ratio=0.4, thickness=15),
        StochasticLayer(densitymean=1900, youngmean=5e8, poissonmean=0.33,thicknessmean=np.inf,
                         densitystd=400, youngstd=2e8, poissonstd=0.1)
    ]

    #setting seed on every layer to provide consistently testable results
    for layer in soil_layers:
        layer.rng = np.random.default_rng(2025)


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




    cs.soil.realise()
    cs.compute()
    print(f"Critical speed: {cs.critical_speed} m/s at a frequency of {cs.frequency}")

    np.testing.assert_almost_equal(cs.critical_speed, [81.4493569])
    np.testing.assert_almost_equal(cs.frequency, [68.36840905])


    cs.soil.realise()
    cs.compute()
    print(f"Critical speed: {cs.critical_speed} m/s at a frequency of {cs.frequency}")

    np.testing.assert_almost_equal(cs.critical_speed, [85.47332512])
    np.testing.assert_almost_equal(cs.frequency, [75.39105381])
