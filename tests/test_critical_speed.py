import numpy as np

from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.soil_dispersion import Layer, SoilDispersion
from TrainCritSpeed.critical_speed import CriticalSpeed


def test_critical_speed():
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

    soil_layers = [
        Layer(density=1900, young_modulus=2.67e7, poisson_ratio=0.33, thickness=5),
        Layer(density=1900, young_modulus=1.14e8, poisson_ratio=0.33, thickness=10),
        Layer(density=1900, young_modulus=2.63e8, poisson_ratio=0.33, thickness=15),
        Layer(density=1900, young_modulus=4.71e8, poisson_ratio=0.33, thickness=np.inf),
    ]

    ballast = BallastedTrack(ballast_parameters, omega)
    dispersion = SoilDispersion(soil_layers, omega)

    cs = CriticalSpeed(omega, ballast, dispersion)
    assert cs.critical_speed == 0.0
    assert cs.frequency == 0.0
    cs.compute()

    np.testing.assert_almost_equal(cs.critical_speed, [74.5802565])
    np.testing.assert_almost_equal(cs.frequency, [57.22559056])
