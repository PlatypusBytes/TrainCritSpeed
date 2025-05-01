import numpy as np
from TrainCritSpeed.soil_dispersion import Layer, SoilDispersion
from TrainCritSpeed.critical_speed import CriticalSpeed

class StochasticLayer(Layer):
    """
    Stochastic extension to the Layer class that holds mean/std values and 'realises itself' when requested,
    can be used in the place of the regular Layer class
    Attributes:
        densitymean (float): Mean of layer density [kg/m^3].
        densitystd (float): Standard deviation of layer density [kg/m^3].
        youngmean (float): Mean of layer Young's modulus [Pa].
        youngstd (float): Standard deviation of layer Young's modulus [Pa].
        poissonmean (float): Mean of layer Poisson's ratio.
        poissonstd (float): Standard deviation of layer Poisson's ratio.
        thicknessmean (float): Mean of layer thickness [m].
        thicknessstd (float): Standard deviation of layer thickness [m].
    """

    def __init__(self,densitymean:float,densitystd:float,youngmean:float,youngstd:float,poissonmean:float,poissonstd:float,thicknessmean:float,thicknessstd:float):
        self.densitymean=densitymean
        self.densitystd=densitystd
        self.youngmean=youngmean
        self.youngstd=youngstd
        self.poissonmean=poissonmean
        self.poissonstd=poissonstd
        self.thicknessmean=thicknessmean
        self.thicknessstd=thicknessstd
        self.rng=np.random.default_rng()

        if densitystd<0 or youngstd<0 or poissonstd<0 or thicknessstd<0:
            raise ValueError("Initialisation of stochastic layer failed, a standard deviation can never be negative")
        
        #initial realisation
        self.realise()
    
    def realise(self):
        """
        Spits out a randomised layer using a normal distribution 
        """
        self.density = self.rng.normal(self.densitymean, self.densitystd)
        self.young_modulus = self.rng.normal(self.youngmean, self.youngstd)
        self.poisson_ratio = self.rng.normal(self.poissonmean, self.poissonstd)
        self.thickness = self.rng.normal(self.thicknessmean, self.thicknessstd)

        
        """
        Compute the shear and compression wave velocities.
        """
        shear_modulus = self.young_modulus / (2 * (1 + self.poisson_ratio))
        p_modulus = self.young_modulus * (1 - self.poisson_ratio) / ((1 + self.poisson_ratio) *
                                                                     (1 - 2 * self.poisson_ratio))
        self.c_s = np.sqrt(shear_modulus / self.density)
        self.c_p = np.sqrt(p_modulus / self.density)

        # check for the edge case that normal returns a negative value or poisson of/above 0.5:
        # this should be replaced with lognormal if it becomes more than an edge case
        if self.density<=0 or self.young_modulus<=0 or self.poisson_ratio<=0 or self.thickness<=0 or self.poisson_ratio>=0.5: 
            self.realise()

class StochasticSoilDispersion(SoilDispersion):
    """
    Extension to the SoilDispersion class which adds a method to realise all stochastic soil layers held inside of it
    """
    def realise(self):
        for soil in self.soil_layers:
            if isinstance(soil,StochasticLayer):
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


