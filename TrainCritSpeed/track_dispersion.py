from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from scipy import optimize


@dataclass
class BallastTrackParameters:
    """
    Data class for the parameters of ballasted track.

    Attributes:
        EI_rail (float): Rail bending stiffness [N·m^2].
        m_rail (float): Rail mass per unit length [kg/m].
        k_rail_pad (float): Railpad stiffness [N/m].
        c_rail_pad (float): Railpad damping [N·s/m].
        m_sleeper (float): Sleeper (distributed) mass [kg/m].
        E_ballast (float): Young's modulus of ballast [Pa].
        h_ballast (float): Ballast (layer) thickness [m].
        width_sleeper (float): Half-track width [m].
        rho_ballast (float): Ballast density [kg/m^3].
        alpha (float): Adimensional parameter.
        cp (float): Compression wave speed in ballast [m/s].
        soil_stiffness (float): Soil (spring) stiffness [N/m].
    """
    EI_rail: float
    m_rail: float
    k_rail_pad: float
    c_rail_pad: float
    m_sleeper: float
    E_ballast: float
    h_ballast: float
    width_sleeper: float
    rho_ballast: float
    alpha: float = 0.5
    cp: float = 0.0
    soil_stiffness: float = 0.0


@dataclass
class SlabTrackParameters:
    """
    Data class for the parameters of slab track.

    Attributes:
        EI_rail (float): Rail bending stiffness [N·m^2].
        m_rail (float): Rail mass per unit length [kg/m].
        EI_slab (float): Slab bending stiffness [N·m^2].
        m_slab (float): Slab mass per unit length [kg/m].
        k_rail_pad (float): Railpad stiffness [N/m].
        c_rail_pad (float): Railpad damping [N·s/m].
        soil_stiffness (float): Soil (spring) stiffness [N/m].
    """
    EI_rail: float
    m_rail: float
    EI_slab: float
    m_slab: float
    k_rail_pad: float
    c_rail_pad: float
    soil_stiffness: float = 0.0


class TrackDispersionAbc(ABC):
    """
    Abstract class for track dispersion models.
    """
    @abstractmethod
    def track_dispersion():
        """Abstract method to calculate track dispersion.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Method track_dispersion must be implemented")


class BallastedTrack(TrackDispersionAbc):
    """
    Ballasted track dispersion model.

    Based on the work of Mezher et al. (2016). Railway critical velocity - Analytical prediction and analysis.
    """
    def __init__(self, params: BallastTrackParameters, omega: npt.NDArray[np.float64],
                 initial_wave_number: float = 1e-3, end_wave_number: float = 1e3):
        """
        Initialize the ballasted track model with the given parameters.

        Args:
            params (BallasTrackParameters): Track parameters.
            omega (npt.NDArray[np.float64]): Angular frequency array.
            initial_wave_number (float): Initial wave number. Default is 1e-3.
            end_wave_number (float): End wave number. Default is 1e3.
        """
        self.parameters = params

        # Calculate the compression wave velocity in the ballast
        self.parameters.cp = np.sqrt(self.parameters.E_ballast / self.parameters.rho_ballast)

        # angular frequency
        self.omega = omega

        # frequency
        self.frequency = omega / (2 * np.pi)

        # railpad stiffness
        self.k_rail_pad = self.parameters.k_rail_pad # + 1j * omega * self.parameters.c_rail_pad

        # initial wave number for the root finding algorithm
        self._initial_wave_number = initial_wave_number
        self._end_wave_number = end_wave_number

        self.phase_velocity = np.zeros_like(omega)


    def __track_stiffness_matrix(self, wave_number: float, omega: float):
        """
        Calculate the determinant of the stiffness matrix for the given wave number.

        Args:
            wave_number (float): Wavenumber to evaluate.
            omega (float): Angular frequency

        Returns:
            float: Determinant of the stiffness matrix.
        """
        # auxiliar values
        tan_value = np.tan(omega * self.parameters.h_ballast / self.parameters.cp) * self.parameters.cp
        sin_value = np.sin(omega * self.parameters.h_ballast / self.parameters.cp) * self.parameters.cp

        # stiffness matrix
        k11 = self.parameters.EI_rail * wave_number ** 4 + self.k_rail_pad - omega ** 2 * self.parameters.m_rail
        k12 = self.k_rail_pad
        k22 = self.k_rail_pad + (2 * omega * self.parameters.E_ballast * self.parameters.width_sleeper * self.parameters.alpha) / tan_value - omega**2 * self.parameters.m_sleeper
        k23 = -2 * omega * self.parameters.E_ballast * self.parameters.width_sleeper * self.parameters.alpha / sin_value
        k33 = 2 * omega * self.parameters.E_ballast * self.parameters.width_sleeper * self.parameters.alpha / tan_value + self.parameters.soil_stiffness

        stiffness = np.array([[k11, k12, 0],
                              [k12, k22, k23],
                              [0, k23, k33],
                              ])

        return np.linalg.det(stiffness)


    def track_dispersion(self):
        """
        Find the wavenumber that causes the determinant of the stiffness matrix to be zero.

        Returns:
            float: Wave number solution.

        Raises:
            ValueError: If the solver fails to converge to a solution.
        """

        # Root finding algorithm to find the wavenumber that makes the determinant zero
        for i, om in enumerate(self.omega):
            solution = optimize.root_scalar(self.__track_stiffness_matrix,
                                            args=(om),
                                            bracket=[self._initial_wave_number, self._end_wave_number],
                                            method='brentq'
                                            )
            if not solution.converged:
                raise ValueError(f"Solver failed to converge for angular frequency {om}\n"
                                 "Please check the initial and end wavenumbers.")

            self.phase_velocity[i] = om / solution.root


class SlabTrack(TrackDispersionAbc):
    """
    Slab track dispersion model.

    Based on the work of Mezher et al. (2016). Railway critical velocity - Analytical prediction and analysis.
    """
    def __init__(self, params: SlabTrackParameters, omega: npt.NDArray[np.float64],
                 initial_wave_number: float = 1e-3, end_wave_number: float = 1e3):
        """
        Initialize the slab track model with the given parameters.

        Args:
            params (BallastTrackParameters): Track parameters.
            omega (npt.NDArray[np.float64]): Angular frequency array.
            initial_wave_number (float): Initial wave number. Default is 1e-3.
            end_wave_number (float): End wave number. Default is 1e3.
        """
        self.parameters = params

        # angular frequency
        self.omega = omega

        # frequency
        self.frequency = omega / (2 * np.pi)

        # railpad stiffness
        self.k_rail_pad = self.parameters.k_rail_pad # + 1j * omega * self.parameters.c_rail_pad

        # initial wave number for the root finding algorithm
        self._initial_wave_number = initial_wave_number
        self._end_wave_number = end_wave_number

        self.phase_velocity = np.zeros_like(omega)


    def __track_stiffness_matrix(self, wave_number: float, omega: float):
        """
        Calculate the determinant of the stiffness matrix for the given wave number.

        Args:
            wave_number (float): Wavenumber to evaluate.
            omega (float): Angular frequency

        Returns:
            float: Determinant of the stiffness matrix.
        """

        # stiffness matrix
        k11 = self.parameters.EI_rail * wave_number ** 4 + self.k_rail_pad - omega ** 2 * self.parameters.m_rail
        k12 = self.k_rail_pad
        k22 = self.k_rail_pad + self.parameters.EI_slab * wave_number ** 4 - omega**2 * self.parameters.m_slab + self.parameters.soil_stiffness

        stiffness = np.array([[k11, k12],
                              [k12, k22],
                              ])

        return np.linalg.det(stiffness)


    def track_dispersion(self):
        """
        Find the wavenumber that causes the determinant of the stiffness matrix to be zero.

        Returns:
            float: Wave number solution.

        Raises:
            ValueError: If the solver fails to converge to a solution.
        """

        # Root finding algorithm to find the wavenumber that makes the determinant zero
        for i, om in enumerate(self.omega):
            solution = optimize.root_scalar(self.__track_stiffness_matrix,
                                            args=(om),
                                            bracket=[self._initial_wave_number, self._end_wave_number],
                                            method='brentq'
                                            )
            if not solution.converged:
                raise ValueError(f"Solver failed to converge for angular frequency {om}\n"
                                 "Please check the initial and end wavenumbers.")
            self.phase_velocity[i] = om / solution.root