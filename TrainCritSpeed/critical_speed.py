from typing import Union
import numpy as np
from TrainCritSpeed.track_dispersion import BallastedTrack, SlabTrack
from TrainCritSpeed.soil_dispersion import SoilDispersion


class CriticalSpeed:
    """
    Computes the critical speed of a train on a track-soil system.
    Based on on the work of Mezher et al. (2016).
    """

    def __init__(self, omega, track: Union[BallastedTrack, SlabTrack], soil: SoilDispersion):
        """
        Constructor for the CriticalSpeed class.

        Args:
            omega (np.ndarray): Angular frequencies.
            track (Union[BallastedTrack, SlabTrack]): Track model.
            soil (SoilDispersion): Soil model.
        """
        self.omega = omega
        self.track = track
        self.soil = soil
        self.critical_speed = 0.0
        self.frequency = 0.0

    def compute(self):
        """
        Compute the critical speed of a train on a track-soil system.
        """

        self.track.track_dispersion()
        self.soil.soil_dispersion()

        # intersection between track and soil dispersion curves
        self.frequency, self.critical_speed = self.intersection(self.omega, self.track.phase_velocity,
                                                                self.soil.phase_velocity)

    @staticmethod
    def intersection(x, y1, y2):
        """
        Find the intersections between two curves defined by x and y1, y2.

        Args:
            x (np.ndarray): x-values.
            y1 (np.ndarray): y-values for the first curve.
            y2 (np.ndarray): y-values for the second curve.

        Returns:
            np.ndarray: x-values where the curves intersect.
            np.ndarray: y-values where the curves intersect.
        """
        # Find where the difference between curves changes sign
        diff = y1 - y2
        # Find indices where the sign changes (product of adjacent differences is <= 0)
        sign_change_indices = np.where((diff[:-1] * diff[1:]) <= 0)[0]

        intersections_x = []
        intersections_y = []

        # For each sign change, interpolate to find the exact intersection point
        for i in sign_change_indices:
            # Linear interpolation to find the x-value where y1 == y2
            if diff[i] == diff[i + 1]:  # Handle the case where both points are exactly equal
                x_intersect = x[i]
                y_intersect = y1[i]  # or y2[i], they're the same
            else:
                # Calculate the fraction between samples where the intersection occurs
                fraction = -diff[i] / (diff[i + 1] - diff[i])
                x_intersect = x[i] + fraction * (x[i + 1] - x[i])
                # Interpolate y-value at the intersection
                y_intersect = y1[i] + fraction * (y1[i + 1] - y1[i])

            intersections_x.append(x_intersect)
            intersections_y.append(y_intersect)

        return np.array(intersections_x), np.array(intersections_y)
