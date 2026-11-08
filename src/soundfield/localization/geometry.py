"""Microphone positions and far-field plane-wave responses."""

from dataclasses import dataclass

import numpy as np

from ..contracts import finite_scalar, positive_integer, real_array


@dataclass(frozen=True)
class ArrayGeometry:
    positions_meters: tuple[tuple[float, float, float], ...]
    sound_speed_meters_per_second: float = 340.0

    def __post_init__(self):
        positions = real_array(self.positions_meters, "positions_meters", 2)
        if positions.shape[1] != 3 or positions.shape[0] < 2:
            raise ValueError("positions_meters must have shape (microphones >= 2, 3)")
        finite_scalar(self.sound_speed_meters_per_second, "sound_speed_meters_per_second", minimum=1)
        object.__setattr__(self, "positions_meters", tuple(map(tuple, positions)))

    @property
    def microphone_count(self):
        return len(self.positions_meters)


def linear_array(microphone_count=4, spacing_meters=.035):
    positive_integer(microphone_count, "microphone_count")
    finite_scalar(spacing_meters, "spacing_meters", minimum=np.finfo(float).tiny)
    return ArrayGeometry(tuple((-index * spacing_meters, 0.0, 0.0) for index in range(microphone_count)))


def circular_array(microphone_count=4, radius_meters=.035):
    positive_integer(microphone_count, "microphone_count")
    finite_scalar(radius_meters, "radius_meters", minimum=np.finfo(float).tiny)
    angles = np.arange(microphone_count) * 2 * np.pi / microphone_count
    return ArrayGeometry(tuple(zip(radius_meters * np.cos(angles), radius_meters * np.sin(angles), np.zeros(microphone_count))))


def plane_wave_responses(geometry, frequencies_hz, azimuths_degrees, *, polar_angle_radians=np.pi / 2, propagation_sign=-1):
    """Return (frequencies, microphones, directions), with azimuth from +x toward +y."""
    frequencies = real_array(frequencies_hz, "frequencies_hz", 1)
    azimuths = real_array(azimuths_degrees, "azimuths_degrees", 1)
    finite_scalar(polar_angle_radians, "polar_angle_radians")
    if propagation_sign not in (-1, 1) or np.any(frequencies < 0):
        raise ValueError("propagation_sign must be -1 or 1 and frequencies must be nonnegative")
    angles = np.deg2rad(azimuths)
    directions = np.stack((np.sin(polar_angle_radians) * np.cos(angles),
                           np.sin(polar_angle_radians) * np.sin(angles),
                           np.full_like(angles, np.cos(polar_angle_radians))))
    delays = propagation_sign * np.asarray(geometry.positions_meters) @ directions / geometry.sound_speed_meters_per_second
    return np.exp(-2j * np.pi * frequencies[:, None, None] * delays[None])
