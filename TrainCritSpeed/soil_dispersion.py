from dataclasses import dataclass
from typing import List
import warnings

import numpy as np
from tqdm import tqdm
from scipy import optimize

np.seterr(invalid='ignore')
warnings.simplefilter("ignore", category=RuntimeWarning)


@dataclass
class Layer:
    """
    Data class for the parameters of a layer.

    Attributes:
        density (float): Layer density [kg/m^3].
        young_modulus (float): Layer Young's modulus [Pa].
        poisson_ratio (float): Layer Poisson's ratio.
        thickness (float): Layer thickness [m].
    """
    density: float
    young_modulus: float
    poisson_ratio: float
    thickness: float
    shear_modulus: float = None
    c_s: float = None
    c_p: float = None

    def __post_init__(self):
        """
        Compute the shear and compression wave velocities.
        """
        shear_modulus = self.young_modulus / (2 * (1 + self.poisson_ratio))
        p_modulus = self.young_modulus * (1 - self.poisson_ratio) / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))
        self.c_s = np.sqrt(shear_modulus / self.density)
        self.c_p = np.sqrt(p_modulus / self.density)


class SoilDispersion:
    """"
    Compute the dispersion of a soil layer.

    It is based on the Fast Delta Matrix method:
    Buchen, P.W. and Ben-Hador, R. (1996). Free-mode surface-wave computations.
    Geophysical Journal International, 124(3), 869-887.

    The last layer is always assumed to be a halfspace.
    """

    def __init__(self, soil_layers: List[Layer], omegas: np.ndarray, resolution=100):
        """
        Initialize the soil dispersion model.

        Args:
            soil_layers (List[Layer]): List of soil layers.
            omegas (np.ndarray): Angular frequencies.
            resolution (int): Resolution of the phase velocity search space.
        """
        for layer in soil_layers:
            if not isinstance(layer, Layer):
                raise TypeError("All layers must be of type Layer.")
        self.soil_layers = soil_layers
        self.omega = omegas
        self.phase_velocity = np.zeros(len(omegas))
        self.resolution = resolution
        # define minimum and maximum values for the phase velocity iterative search
        self.min_c = 0.9 * np.min([layer.c_s for layer in soil_layers])
        self.max_c = 1.1 * np.max([layer.c_s for layer in soil_layers])


    def soil_dispersion(self):
        """
        Compute the dispersion of the soil layers.
        """

        # resample the phase velocity search space
        # the solution involved very large numbers, so the space needs to have high resolution
        c_list = np.linspace(self.min_c, self.max_c, int((self.max_c - self.min_c) * self.resolution + 1))

        for j, omega in enumerate(tqdm(self.omega)):

            # Find the first sign change to bracket the root
            d_1 = self.__compute_dispersion_fastdelta(c_list[0], omega, self.soil_layers)
            for i in range(len(c_list) - 1):
                d_2 = self.__compute_dispersion_fastdelta(c_list[i + 1], omega, self.soil_layers)
                if d_1 * d_2 < 0:  # Sign change detected
                    root_interval = (c_list[i], c_list[i + 1])
                    break
                d_1 = d_2

            # find the root within the bracket root
            solution = optimize.root_scalar(self.__compute_dispersion_fastdelta,
                                            args=(omega, self.soil_layers),
                                            bracket=root_interval,
                                            method='brentq')

            self.phase_velocity[j] = solution.root

    @staticmethod
    def __compute_dispersion_fastdelta(c: float, omega: float, layers: List[Layer]):
        """
        Compute the dispersion of the soil layers using the Fast Delta Matrix method.

        Args:
            c (float): Phase velocity.
            omega (float): Angular frequency.
            layers (List[Layer]): List of soil layers.

        Returns:
            float: Dispersion value.
        """
        wave_number = omega / c  # wavenumber
        num_layers = len(layers)

        # Pre-compute values for the first layer
        beta0 = layers[0].c_s
        t_value = 2 - c**2 / beta0**2
        mu0 = layers[0].density * beta0**2

        # Initialize X1
        X1 = mu0**2 * np.array([2 * t_value, -t_value**2, 0, 0, -4])

        # Compute terms for half-space (last layer)
        _, _, _, _, r_h, s_h = SoilDispersion.__compute_terms(
            c, wave_number, layers[-1].thickness, layers[-1].c_p, layers[-1].c_s
            )

        # Process intermediate layers
        for i in range(num_layers - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]

            # Calculate layer properties directly when needed
            gamma = (current_layer.c_s / c)**2
            gamma_next = (next_layer.c_s / c)**2

            C_alpha, S_alpha, C_beta, S_beta, r, s = SoilDispersion.__compute_terms(
                c, wave_number, current_layer.thickness, current_layer.c_p, current_layer.c_s
            )

            epsilon = next_layer.density / current_layer.density
            eta = 2 * (gamma - epsilon * gamma_next)

            a = epsilon + eta
            a_prime = a - 1
            b = 1 - eta
            b_prime = b - 1

            # Extract X1 components
            x1, x2, x3, x4, x5 = X1

            # Calculate intermediate values
            p1 = C_beta * x2 + s * S_beta * x3
            p2 = C_beta * x4 + s * S_beta * x5
            p3 = 1 / s * S_beta * x2 + C_beta * x3
            p4 = 1 / s * S_beta * x4 + C_beta * x5

            q1 = C_alpha * p1 - r * S_alpha * p2
            q2 = -1 / r * S_alpha * p3 + C_alpha * p4
            q3 = C_alpha * p3 - r * S_alpha * p4
            q4 = -1 / r * S_alpha * p1 + C_alpha * p2

            y1 = a_prime * x1 + a * q1
            y2 = a * x1 + a_prime * q2
            z1 = b * x1 + b_prime * q1
            z2 = b_prime * x1 + b * q2

            # Update X1 for next iteration
            X1 = np.array([
                b_prime * y1 + b * y2,        # x_hat_1
                a * y1 + a_prime * y2,        # x_hat_2
                epsilon * q3,                 # x_hat_3
                epsilon * q4,                 # x_hat_4
                b_prime * z1 + b * z2         # x_hat_5
            ])

        # Calculate final determinant
        D = X1[1] + s_h * X1[2] - r_h * (X1[3] + s_h * X1[4])

        return D.real

    @staticmethod
    def __compute_terms(c, k, d, c_p, c_s):
        """
        Compute C and S terms for P and S waves

        Args:
            c (float): Wave speed.
            k (float): Wavenumber.
            d (float): Layer thickness.
            c_p (float): Compression wave speed.
            c_s (float): Shear wave speed.

        Returns:
            tuple: C_alpha, S_alpha, C_beta, S_beta, r, s
        """
        if c < c_p:
            r = np.sqrt(1 - c**2 / c_p**2)
            C_alpha = np.cosh(k * r * d)
            S_alpha = np.sinh(k * r * d)
        else:
            r = np.sqrt(c**2 / c_s**2 - 1)
            C_alpha = np.cos(k * r * d)
            S_alpha = 1j * np.sin(k * r * d)

        if c < c_s:
            s = np.sqrt(1 - c**2 / c_s**2)
            C_beta = np.cosh(k * s * d)
            S_beta = np.sinh(k * s * d)
        else:
            s = np.sqrt(c**2 / c_s**2 - 1)
            C_beta = np.cos(k * s * d)
            S_beta = 1j * np.sin(k * s * d)

        return C_alpha, S_alpha, C_beta, S_beta, r, s