import numpy as np
import matplotlib.pyplot as plt

from TrainCritSpeed.track_dispersion import BallastedTrack, BallastTrackParameters
from TrainCritSpeed.track_dispersion import SlabTrack, SlabTrackParameters

omega = np.linspace(0.1, 250, 100)

ballast_parameters = BallastTrackParameters(
        EI_rail=1.29e7,
        m_rail=120,
        k_rail_pad=5e8,
        c_rail_pad=2.5e5,
        m_sleeper=490,
        E_ballast=130e6,
        h_ballast=0.35,
        width_sleeper=1.25,
        soil_stiffness=0.0,
        rho_ballast=1700
    )

ballast = BallastedTrack(ballast_parameters, omega)
ballast.track_dispersion()

slab_parameters = SlabTrackParameters(
        EI_rail = 1.29e7,
        m_rail = 120,
        k_rail_pad = 5e8,
        c_rail_pad = 2.5e5,
        EI_slab = 30e9 * (1.25 * 0.35 ** 3 / 12),
        m_slab = 2500 * 1.25 * 0.35,
        soil_stiffness = 0.0,
    )

slab = SlabTrack(slab_parameters, omega)
slab.track_dispersion()


# make plot
plt.plot(omega / 2 / np.pi, ballast.phase_velocity, label="Ballast Track")
plt.plot(omega / 2 / np.pi, slab.phase_velocity, label="Slab Track")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase speed [m/s]")
plt.grid()
plt.legend()
plt.show()
