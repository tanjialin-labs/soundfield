"""Event-based beam choice using constrained-band and full-band energy."""

from dataclasses import dataclass
import numpy as np
from ..contracts import spectral_array, unit_interval_array, positive_integer, nonnegative_integer, finite_scalar


@dataclass(frozen=True)
class EventBeamConfig:
    excluded_tail_frames: int = 20
    energy_frames: int = 40
    lower_band: tuple[int, int] = (47, 147)
    upper_band: tuple[int, int] = (147, 217)
    full_band: tuple[int, int] = (2, 256)
    lower_band_weight: float = 1.
    minimum_full_band_ratio: float = .2

    def __post_init__(self):
        nonnegative_integer(self.excluded_tail_frames, 'excluded_tail_frames')
        positive_integer(self.energy_frames, 'energy_frames')
        for band in (self.lower_band, self.upper_band, self.full_band):
            if not 0 <= band[0] < band[1] <= 257:
                raise ValueError('frequency intervals must fit 257 bins')
        for name in ('lower_band_weight', 'minimum_full_band_ratio'):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)


def select_event_beam(spectrum, beam_activity, config=EventBeamConfig()):
    values = spectral_array(spectrum)
    activity = unit_interval_array(beam_activity, 'beam_activity', (values.shape[2],))
    if values.shape[0] != 257 or np.any((activity != 0)&(activity != 1)):
        raise ValueError('event selection requires 257 bins and binary beam activity')
    active = np.flatnonzero(activity)
    if len(active) == 1:
        return int(active[0])
    if not len(active):
        active = np.arange(values.shape[2])
    stop = values.shape[1]-config.excluded_tail_frames
    start = stop-config.energy_frames
    if start < 0:
        raise ValueError('audio does not cover the event energy history')
    power = np.abs(values[:, start:stop])**2
    def energy(band):
        return np.sum(power[slice(*band)], axis=(0, 1))[active]
    constrained = config.lower_band_weight*energy(config.lower_band)+(1-config.lower_band_weight)*energy(config.upper_band)
    chosen = int(np.argmax(constrained))
    full = energy(config.full_band)
    maximum = int(np.argmax(full))
    selected = maximum if full[chosen] < config.minimum_full_band_ratio * full[maximum] else chosen
    return int(active[selected])
