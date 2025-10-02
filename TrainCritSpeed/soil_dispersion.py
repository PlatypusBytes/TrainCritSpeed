from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import warnings

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from scipy import optimize
import matplotlib.pyplot as plt


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
    c_s: float = np.nan
    c_p: float = np.nan

    def __post_init__(self):
        """
        Compute the shear and compression wave velocities.
        """
        shear_modulus = self.young_modulus / (2 * (1 + self.poisson_ratio))
        p_modulus = self.young_modulus * (1 - self.poisson_ratio) / ((1 + self.poisson_ratio) *
                                                                     (1 - 2 * self.poisson_ratio))
        self.c_s = np.sqrt(shear_modulus / self.density)
        self.c_p = np.sqrt(p_modulus / self.density)


class SoilDispersion:
    """
    Compute the dispersion of a soil layer.

    It is based on the Fast Delta Matrix method:
    Buchen, P.W. and Ben-Hador, R. (1996). Free-mode surface-wave computations.
    Geophysical Journal International, 124(3), 869-887.

    The last layer is always assumed to be a halfspace.
    """

    def __init__(self, soil_layers: List[Layer], omegas: npt.NDArray[np.float64], nb_modes=5, step=0.001):
        """
        Initialize the soil dispersion model.

        Args:
            soil_layers (List[Layer]): List of soil layers.
            omegas (np.ndarray): Angular frequencies.
            nb_modes (int): Number of modes to compute (Optional: default is 1).
            step (float): Step size for the phase velocity search (Optional: default is 0.01).
        """
        for layer in soil_layers:
            if not isinstance(layer, Layer):
                raise TypeError("All layers must be of type Layer.")
        self.soil_layers = soil_layers
        self.omega = omegas
        self.phase_velocity = np.full((len(omegas), nb_modes), np.nan)
        self.nb_modes = nb_modes
        self.step = step
        # define minimum and maximum values for the phase velocity iterative search
        self.min_c = 0.5 * np.min([layer.c_s for layer in soil_layers])
        self.max_c = np.max([layer.c_s for layer in soil_layers])

    def soil_dispersion(self):
        """
        Compute the dispersion of the soil layers.
        """

        # resample the phase velocity search space
        # the solution involved very large numbers, so the space needs to have high resolution
        c_list = np.arange(self.min_c, self.max_c + self.step, self.step)

        for j, omega in enumerate(tqdm(self.omega)):
            if omega == 0:
                self.phase_velocity[j, :] = np.nan
                continue
            # Find the first sign change to bracket the root
            d = self.__compute_dispersion_fastdelta(c_list, omega, self.soil_layers)

            roots = []
            for i in range(len(c_list) - 1):
                if d[i] * d[i+1] < 0:  # Sign change detected
                    root_interval = (c_list[i], c_list[i + 1])

                    # find the root within the bracket root
                    solution = optimize.root_scalar(self.__compute_dispersion_fastdelta,
                                                    args=(omega, self.soil_layers),
                                                    bracket=root_interval,
                                                    method='brentq')
                    roots.append(solution.root)

            if not roots:
                self.phase_velocity[j, :nb_found_modes] = np.nan
                continue
            else:
                nb_found_modes = min(len(roots), self.nb_modes)
                self.phase_velocity[j, :nb_found_modes] = roots[:nb_found_modes]

    @staticmethod
    def __compute_dispersion_fastdelta(c: npt.NDArray[np.float64], omega: float,
                                       layers: List[Layer]) -> npt.NDArray[np.float64]:
        """
        Compute the dispersion of the soil layers using the Fast Delta Matrix method.

        Args:
            c (npt.NDArray[np.float64]): Phase velocity.
            omega (float): Angular frequency.
            layers (List[Layer]): List of soil layers.

        Returns:
            npt.NDArray[np.float64]: Dispersion value.
        """
        wave_number = omega / c  # wavenumber
        num_layers = len(layers)

        # Pre-compute values for the first layer
        beta0 = layers[0].c_s
        t_value = 2 - c**2 / beta0**2
        mu0 = layers[0].density * beta0**2

        # Initialize X1
        X1 = mu0**2 * np.array(
            [2 * t_value, -t_value**2,
             np.zeros_like(t_value),
             np.zeros_like(t_value),
             np.ones_like(t_value) * -4])

        # Compute terms for half-space (last layer)
        _, _, _, _, r_h, s_h = SoilDispersion.__compute_terms(c, wave_number, layers[-1].thickness, layers[-1].c_p,
                                                              layers[-1].c_s)

        # Process intermediate layers
        for i in range(num_layers - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]

            # Calculate layer properties directly when needed
            gamma = (current_layer.c_s / c)**2
            gamma_next = (next_layer.c_s / c)**2

            C_alpha, S_alpha, C_beta, S_beta, r, s = SoilDispersion.__compute_terms(c, wave_number,
                                                                                    current_layer.thickness,
                                                                                    current_layer.c_p,
                                                                                    current_layer.c_s)

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
                b_prime * y1 + b * y2,  # x_hat_1
                a * y1 + a_prime * y2,  # x_hat_2
                epsilon * q3,  # x_hat_3
                epsilon * q4,  # x_hat_4
                b_prime * z1 + b * z2  # x_hat_5
            ])

        # Calculate final determinant
        D = X1[1] + s_h * X1[2] - r_h * (X1[3] + s_h * X1[4])

        return np.asarray(D.real, dtype=np.float64)

    @staticmethod
    def __compute_terms(
        c: npt.NDArray[np.float64], k: npt.NDArray[np.float64], d: float, c_p: float, c_s: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64],
               npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute C and S terms for P and S waves

        Args:
            c (npt.NDArray[np.float64]): Wave speed.
            k (npt.NDArray[np.float64]): Wavenumber.
            d (float): Layer thickness.
            c_p (float): Compression wave speed.
            c_s (float): Shear wave speed.

        Returns:
            tuple: C_alpha, S_alpha, C_beta, S_beta, r, s
        """

        c = np.asarray(c, dtype=np.complex128)
        k = np.asarray(k, dtype=np.complex128)

        r = np.sqrt(1 - (c / c_p) ** 2 + 0j)
        s = np.sqrt(1 - (c / c_s) ** 2 + 0j)

        C_alpha = np.cosh(k * r * d)
        S_alpha = np.sinh(k * r * d)
        C_beta = np.cosh(k * s * d)
        S_beta = np.sinh(k * s * d)

        return C_alpha, S_alpha, C_beta, S_beta, r, s


    def soil_dispersion_image(self, c_min: float = None, c_max: float = None, c_step: float = 5.0,
                              file_name: Path = "./soil_dispersion.png"):
        """
        Computes and plots the dispersion function over a 2D grid of frequency vs. phase velocity.

        The resulting image reveals all dispersion modes as dark lines.

        Args:
            c_min (float, optional): The minimum phase velocity for the search grid.
                                     Defaults to 0.5 * min(layer_cs).
            c_max (float, optional): The maximum phase velocity for the search grid.
                                     Defaults to the half-space shear velocity.
            c_step (float): The velocity step [m/s] for the grid.
        """

        # check if critical speed has been computed
        if np.all(np.isnan(self.phase_velocity)):
            warnings.warn("Soil dispersion has not been computed yet, therefore the phase velocity will not be plotted."
            "Run the soil_dispersion() method first if you want to see the phase velocity on the plot.")

        c_list = np.arange(self.min_c, self.max_c + self.step, self.step)
        frequencies = self.omega / (2 * np.pi)

        # Initialize a matrix to store the determinant values
        determinant_matrix = np.zeros((len(c_list), len(frequencies)))

        # Compute the determinant for each combination of frequency and phase velocity
        for j, omega in enumerate(tqdm(self.omega, desc="Frequencies")):
            if omega == 0:
                determinant_matrix[:, j] = np.nan
            d_values = self.__compute_dispersion_fastdelta(c_list, omega, self.soil_layers)
            determinant_matrix[:, j] = d_values

        # Add a small epsilon to avoid log(0).
        log_determinant = np.log10(np.abs(determinant_matrix) + np.finfo(float).eps)

        # Create the plot
        _, ax = plt.subplots(figsize=(10, 6))
        ax.pcolormesh(
            frequencies,
            c_list,
            log_determinant,
            shading='auto',
            cmap='jet_r',
        )
        ax.plot(frequencies, self.phase_velocity, 'b', linewidth=1)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase Velocity (m/s)")
        plt.savefig(file_name)
        plt.close()
