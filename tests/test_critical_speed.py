import numpy as np

from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.soil_dispersion import Layer, SoilDispersion
from TrainCritSpeed.critical_speed import CriticalSpeed


def test_critical_speed():
    """
    Test based on the paper of Mezher et al. (2016).
    Soil test case with three layers and ballasted track.
    """

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
        Layer(density=2000, young_modulus=30e6, poisson_ratio=0.35, thickness=2),
        Layer(density=2000, young_modulus=40e6, poisson_ratio=0.35, thickness=4),
        Layer(density=2000, young_modulus=75e6, poisson_ratio=0.40, thickness=np.inf),
    ]


    ballast = BallastedTrack(ballast_parameters, omega)
    dispersion = SoilDispersion(soil_layers, omega)

    cs = CriticalSpeed(omega, ballast, dispersion)
    assert cs.critical_speed == 0.0
    assert cs.frequency == 0.0
    cs.compute()

    np.testing.assert_almost_equal(cs.critical_speed, [78.2246], decimal=4)
    np.testing.assert_almost_equal(cs.frequency, [63.0056], decimal=4)
