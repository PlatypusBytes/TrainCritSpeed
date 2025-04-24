import pytest
import numpy as np
from TrainCritSpeed.soil_dispersion import Layer, SoilDispersion


def test_shear_modulus_and_wave_velocities():
    """
    Test the computation of the shear modulus and wave velocities.
    """
    density = 1800.0
    young_modulus = 2e8
    poisson_ratio = 0.3
    thickness = 10.0

    expected_cs = np.sqrt(young_modulus / (2 * (1 + poisson_ratio)) / density)
    expected_cp = np.sqrt(young_modulus * (1 - poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio)) /
                          density)

    layer = Layer(density, young_modulus, poisson_ratio, thickness)

    # test
    np.testing.assert_almost_equal(layer.c_s, expected_cs)
    np.testing.assert_almost_equal(layer.c_p, expected_cp)

    # different props
    density = 2000.0
    young_modulus = 3e8
    poisson_ratio = 0.25
    thickness = 5.0

    shear_modulus = young_modulus / (2 * (1 + poisson_ratio))
    p_modulus = young_modulus * (1 - poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    expected_cs = np.sqrt(shear_modulus / density)
    expected_cp = np.sqrt(p_modulus / density)

    layer = Layer(density, young_modulus, poisson_ratio, thickness)

    # test
    np.testing.assert_almost_equal(layer.c_s, expected_cs)
    np.testing.assert_almost_equal(layer.c_p, expected_cp)

    # different props
    density = 1500.0
    young_modulus = 1e8
    poisson_ratio = 0.499
    thickness = 3.0

    # Expected values
    shear_modulus = young_modulus / (2 * (1 + poisson_ratio))
    p_modulus = young_modulus * (1 - poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    expected_cs = np.sqrt(shear_modulus / density)
    expected_cp = np.sqrt(p_modulus / density)

    # Act
    layer = Layer(density, young_modulus, poisson_ratio, thickness)

    # test
    np.testing.assert_almost_equal(layer.c_s, expected_cs)
    np.testing.assert_almost_equal(layer.c_p, expected_cp)


def test_soil_dispersion():
    """
    Test the computation of the soil dispersion.
    Test based on the example of: Foti et al. (2014). Surface Wave Methods for Near-Surface Site Characterization.
    CRC Press. pp 94 Fig 2.30

    """

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

    # Expected values
    with open("./tests/data/soil_dispersion_1.txt", "r") as f:
        data = f.read().splitlines()
    data = [line.split() for line in data]
    data = np.array(data, dtype=float)

    np.testing.assert_almost_equal(dispersion.phase_velocity, data[:, 1], decimal=3)
    np.testing.assert_almost_equal(omegas, data[:, 0], decimal=3)


def compute_elastic_parameters(vs, vp, density=1900):
    """
    Compute the elastic parameters of a soil layer.

    Args:
        vs (float): Shear wave velocity [m/s].
        vp (float): Compression wave velocity [m/s].
        density (float): Density [kg/m^3].

    Returns:
        E (float): Young's modulus [Pa].
        nu (float): Poisson's ratio.
    """
    nu = (vp**2 - 2 * vs**2) / (2 * (vp**2 - vs**2))
    G = density * vs**2
    E = 2 * G * (1 + nu)
    return E, nu
