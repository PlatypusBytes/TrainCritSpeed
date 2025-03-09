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

    # compute the shear and compression wave velocities
    shear_modulus = young_modulus / (2 * (1 + poisson_ratio))
    p_modulus = young_modulus / (1 + poisson_ratio) / (1 - 2 * poisson_ratio)
    # shear wave velocity
    c_s = np.sqrt(shear_modulus / density)
    # compression wave velocity
    c_p = np.sqrt(p_modulus / density)


class SoilDispersion:
    """"
    Compute the dispersion of a soil layer.

    It is based on the Fast Delta Matrix method:
    Buchen, P.W. and Ben-Hador, R. (1996). Free-mode surface-wave computations.
    Geophysical Journal International, 124(3), 869-887.

    The last layer is always assumed to be a halfspace.
    """

    def __init__(self, soil_layers: List[Layer], omegas: np.ndarray):

        for layer in soil_layers:
            if not isinstance(layer, Layer):
                raise TypeError("All layers must be of type Layer.")
        self.soil_layers = soil_layers
        self.omega = omegas
        self.phase_valocity = np.zeros(len(omegas))
        # define minimum and maximum values for the phase velocity iterative search
        self.min_c = 0.9 * np.min([layer.c_s for layer in soil_layers])
        self.max_c = 1.1 * np.max([layer.c_s for layer in soil_layers])


    def soil_dispersion(self):

        c_list = np.linspace(self.min_c, self.max_c, int((self.max_c - self.min_c) * 10 + 1))

        for j, omega in enumerate(tqdm(self.omega)):
            D_aux = [self.__compute_dispersion_fastdelta(c, omega, self.soil_layers) for c in c_list]
            # find the fist sign change
            idx = np.where(np.diff(np.sign(D_aux)))[0]
            if len(idx) == 0:
                raise ValueError("Not possible to find the root of the dispersion function D(c, wave number).")

            solution = optimize.root_scalar(self.__compute_dispersion_fastdelta,
                                            args=(omega, self.soil_layers),
                                            bracket=[c_list[idx[0]], c_list[idx[0]+1]],
                                            method='brentq')
            self.phase_valocity[j] = solution.root



    def __compute_dispersion_fastdelta(self, c, omega, layers):

        # wavelength = (2 * np.pi * c) / omega
        k = omega / c  # wavenumber
        # k = 2 * np.pi / lambda_val

        t = np.zeros(len(layers))
        mu = np.zeros(len(layers))
        alpha = np.zeros(len(layers))
        beta = np.zeros(len(layers))
        rho = np.zeros(len(layers))
        thickness = np.zeros(len(layers))
        gamma = np.zeros(len(layers))

        for i, lay in enumerate(layers):
            thickness[i] = lay['d']
            alpha[i] = lay['alpha']
            beta[i] = lay['beta']
            rho[i] = lay['rho']
            t[i] = (c - c**2 / beta[i]**2)
            mu[i] = rho[i] * beta[i]**2
            gamma[i] = (beta[i] / c)**2

        t = 2 - c ** 2 / beta[0] ** 2

        # initialise
        X1 = mu[0] ** 2 * np.array([2 * t, -t**2, 0, 0, -4])

        # terms for half-space
        _, _, _, _, r_h, s_h = self.__compute_terms(c, k, thickness[-1], alpha[-1], beta[-1])


        for i, lay in enumerate(layers[:-1]):

            C_alpha, S_alpha, C_beta, S_beta, r, s = self.__compute_terms(c, k, thickness[i], alpha[i], beta[i])

            epsilon = rho[i+1] / rho[i]
            eta = 2 * (gamma[i] - epsilon * gamma[i+1])

            a = epsilon + eta
            a_prime = a - 1
            b = 1 - eta
            b_prime = b - 1

            x1 = X1[0]
            x2 = X1[1]
            x3 = X1[2]
            x4 = X1[3]
            x5 = X1[4]

            p1 = C_beta * x2 + s * S_beta * x3
            p2 = C_beta * x4 + s * S_beta * x5
            p3 = 1 / s * S_beta * x2 + C_beta * x3
            p4 = 1 / s * S_beta * x4 + C_beta * x5

            q1 = C_alpha * p1 - r * S_alpha * p2
            q2 = - 1 / r * S_alpha * p3 + C_alpha * p4
            q3 = C_alpha * p3 - r * S_alpha * p4
            q4 = - 1 / r * S_alpha * p1 + C_alpha * p2

            y1 = a_prime * x1 + a * q1
            y2 = a * x1 + a_prime * q2
            z1 = b * x1 + b_prime * q1
            z2 = b_prime * x1 + b * q2

            x_hat_1 = b_prime * y1 + b * y2
            x_hat_2 = a * y1 + a_prime * y2
            x_hat_3 = epsilon * q3
            x_hat_4 = epsilon * q4
            x_hat_5 = b_prime * z1 + b * z2
            # x_hat_6 = a * z1 + a_prime * z2 == x_hat_1

            X1 = np.array([x_hat_1, x_hat_2, x_hat_3, x_hat_4, x_hat_5])

        D = x_hat_2 + s_h * x_hat_3 - r_h * (x_hat_4 + s_h * x_hat_5)

        return D.real

    @staticmethod
    def __compute_terms(c, k, d, c_p, c_s):
        """
        Compute C and S terms for either P or S waves

        Args:
            c (float): Wave speed.
            k (float): Wavenumber.
            d (float): Layer thickness.
            c_p (float): Compression wave speed.
            c_s (float): Shear wave speed.
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