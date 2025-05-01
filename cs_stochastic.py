from typing import List
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
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
        Spits out a randomised layer using a lognormal distribution 
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

        # check for the edge case that normal returns a negative value:
        # this should be replaced with lognormal if it becomes more than an edge case
        if self.density<=0 or self.young_modulus<=0 or self.poisson_ratio<=0 or self.thickness<=0 or self.poisson_ratio>=0.5: 
            self.realise()

class StochasticSoilDispersion(SoilDispersion):
    def realise(self):
        for soil in self.soil_layers:
            if isinstance(soil,StochasticLayer):
                soil.realise()

class StochasticCriticalSpeed(CriticalSpeed):
    def compute(self):
        """
        Compute the critical speed of a train on a track-soil system.
        """
        self.soil.soil_dispersion()

        # intersection between track and soil dispersion curves
        self.frequency, self.critical_speed = self.intersection(self.omega, self.track.phase_velocity,
                                                                self.soil.phase_velocity)



soil_layers = [
    StochasticLayer(1900,100,3e7,5e6,0.33,0.05,5,0.2),
    StochasticLayer(1900,200,1e8,1e7,0.33,0.05,10,1),
    StochasticLayer(1900,300,3e8,1e8,0.4,0.1,15,5),
    StochasticLayer(1900,400,5e8,2e8,0.33,0.1,np.inf,0)
]


omega = np.linspace(0.1, 250, 100)

#replace with import from file
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

ballast = BallastedTrack(ballast_parameters, omega)

dispersion = StochasticSoilDispersion(soil_layers, omega)

cs = StochasticCriticalSpeed(omega, ballast, dispersion)
cs.track.track_dispersion()

for k in range(5):
    cs.soil.realise()
    cs.compute()
    print(f"Critical speed: {cs.critical_speed} m/s at a frequency of {cs.frequency}")
