import numpy as np
from TrainCritSpeed.soil_dispersion import Layer, SoilDispersion
from TrainCritSpeed.critical_speed import CriticalSpeed


class StochasticLayer(Layer):
    """
    Stochastic extension to the Layer class that holds mean/std values and 'realises itself' when requested,
    can be used in the place of the regular Layer class
    Attributes:
        density_mean (float): Mean of layer density [kg/m^3].
        density_std (float): Standard deviation of layer density [kg/m^3].
        young_mean (float): Mean of layer Young's modulus [Pa].
        young_std (float): Standard deviation of layer Young's modulus [Pa].
        poisson_mean (float): Mean of layer Poisson's ratio.
        poisson_std (float): Standard deviation of layer Poisson's ratio.
        thickness_mean (float): Mean of layer thickness [m].
        thickness_std (float): Standard deviation of layer thickness [m].
    """

    def __init__(self,
                 density_mean: float,
                 young_mean: float,
                 poisson_mean: float,
                 thickness_mean: float,
                 density_std: float = 0,
                 young_std: float = 0,
                 poisson_std: float = 0,
                 thickness_std: float = 0,
                 distribution="normal",
                 seed=None):

        self.distribution = distribution

        if density_std < 0 or young_std < 0 or poisson_std < 0 or thickness_std < 0:
            raise ValueError("Initialisation of stochastic layer failed, a standard deviation can never be negative")

        if distribution == "lognormal":
            self.density_std = np.sqrt(np.log(1 + (density_std / density_mean)**2))
            self.density_mean = np.log(density_mean) - self.density_std**2 / 2
            self.young_std = np.sqrt(np.log(1 + (young_std / young_mean)**2))
            self.young_mean = np.log(young_mean) - self.young_std**2 / 2
            self.poisson_std = np.sqrt(np.log(1 + (poisson_std / poisson_mean)**2))
            self.poisson_mean = np.log(poisson_mean) - self.poisson_std**2 / 2
            self.thickness_std = np.sqrt(np.log(1 + (thickness_std / thickness_mean)**2))
            self.thickness_mean = np.log(thickness_mean) - self.thickness_std**2 / 2
        elif distribution == "normal":
            self.density_mean = density_mean
            self.density_std = density_std
            self.young_mean = young_mean
            self.young_std = young_std
            self.poisson_mean = poisson_mean
            self.poisson_std = poisson_std
            self.thickness_mean = thickness_mean
            self.thickness_std = thickness_std
        else:
            raise ValueError("Distribution can only be normal or lognormal")

        self.rng = np.random.default_rng(seed)

        #initial realisation
        self.realise()

    def realise(self):
        """
        Spits out a randomised layer using a (log)normal distribution 
        """
        if self.distribution == "normal":
            self.density = self.rng.normal(self.density_mean, self.density_std)
            self.young_modulus = self.rng.normal(self.young_mean, self.young_std)
            self.poisson_ratio = self.rng.normal(self.poisson_mean, self.poisson_std)
            self.thickness = self.rng.normal(self.thickness_mean, self.thickness_std)

            # check for the edge case that normal returns a negative value or poisson of/above 0.5:
            if self.density <= 0 or self.young_modulus <= 0 or self.poisson_ratio <= 0 or self.thickness <= 0 or self.poisson_ratio >= 0.5:
                self.realise()

        else:  #because we checked distribution at init and know it can only be normal or lognormal
            self.density = self.rng.lognormal(self.density_mean, self.density_std)
            self.young_modulus = self.rng.lognormal(self.young_mean, self.young_std)
            self.poisson_ratio = self.rng.lognormal(self.poisson_mean, self.poisson_std)
            self.thickness = self.rng.lognormal(self.thickness_mean, self.thickness_std)

            if self.poisson_ratio >= 0.5:
                self.realise()

        #Compute the shear and compression wave velocities.
        shear_modulus = self.young_modulus / (2 * (1 + self.poisson_ratio))
        p_modulus = self.young_modulus * (1 - self.poisson_ratio) / ((1 + self.poisson_ratio) *
                                                                     (1 - 2 * self.poisson_ratio))
        self.c_s = np.sqrt(shear_modulus / self.density)
        self.c_p = np.sqrt(p_modulus / self.density)


class StochasticSoilDispersion(SoilDispersion):
    """
    Extension to the SoilDispersion class which adds a method to realise all stochastic soil layers held inside of it
    """

    def realise(self):
        for soil in self.soil_layers:
            if isinstance(soil, StochasticLayer):
                soil.realise()


class StochasticCriticalSpeed(CriticalSpeed):
    """
    Extension to the CriticalSpeed class which changes the compute method to not calculate track dispersion every time
    """

    def compute(self):
        """
        Compute the critical speed of a train on a track-soil system.
        """
        self.soil.soil_dispersion()

        # intersection between track and soil dispersion curves
        self.frequency, self.critical_speed = self.intersection(self.omega, self.track.phase_velocity,
                                                                self.soil.phase_velocity)
