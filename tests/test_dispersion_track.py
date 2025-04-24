import pytest
import numpy as np
from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.track_dispersion import SlabTrack, SlabTrackParameters


@pytest.fixture
def ballast_parameters():
    """
    Ballast track parameters
    """
    params = BallastTrackParameters(EI_rail=1.29e7,
                                    m_rail=120,
                                    k_rail_pad=5e8,
                                    c_rail_pad=2.5e5,
                                    m_sleeper=490,
                                    E_ballast=130e6,
                                    h_ballast=0.35,
                                    width_sleeper=1.25,
                                    soil_stiffness=0.0,
                                    rho_ballast=1700)
    return params


@pytest.fixture
def slab_parameters():
    """
    Slab track parameters
    """
    params = SlabTrackParameters(
        EI_rail=1.29e7,
        m_rail=120,
        k_rail_pad=5e8,
        c_rail_pad=2.5e5,
        EI_slab=30e9 * (1.25 * 0.35**3 / 12),
        m_slab=2500 * 1.25 * 0.35,
        soil_stiffness=0.0,
    )
    return params


def test_ballast_track_dispersion(ballast_parameters):
    """
    Test for ballasted track
    """

    # Initialize and calculate dispersion for ballasted track
    ballast = BallastedTrack(ballast_parameters, np.linspace(0.1, 250, 100))
    ballast.track_dispersion()

    with open("./tests/data/ballast_dispersion.txt", "r") as f:
        data = f.read().splitlines()
    data = [line.split() for line in data]
    data = np.array(data, dtype=float)

    np.testing.assert_almost_equal(ballast.phase_velocity, data[:, 1], decimal=3)
    np.testing.assert_almost_equal(np.linspace(0.1, 250, 100), data[:, 0], decimal=3)


def test_slab_track_dispersion(slab_parameters):
    """
    Test for slab track
    """
    # Initialize and calculate dispersion for slab track
    ballast = SlabTrack(slab_parameters, np.linspace(0.1, 250, 100))
    ballast.track_dispersion()

    with open("./tests/data/slab_dispersion.txt", "r") as f:
        data = f.read().splitlines()
    data = [line.split() for line in data]
    data = np.array(data, dtype=float)

    np.testing.assert_almost_equal(ballast.phase_velocity, data[:, 1], decimal=3)
    np.testing.assert_almost_equal(np.linspace(0.1, 250, 100), data[:, 0], decimal=3)
