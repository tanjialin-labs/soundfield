"""Windowed speech-to-noise ratios for event-triggered beam comparison."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from ..contracts import nonnegative_integer, positive_integer, spectral_frame


@dataclass(frozen=True)
class WindowedSnrConfig:
    frequency_bin_count: int = 257
    channel_count: int = 3
    start_bin: int = 63
    stop_bin: int = 160
    noise_window_frames: int = 100
    speech_window_frames: int = 40

    def __post_init__(self):
        for name in ("frequency_bin_count", "channel_count", "stop_bin", "noise_window_frames", "speech_window_frames"):
            positive_integer(getattr(self, name), name)
        nonnegative_integer(self.start_bin, "start_bin")
        if not self.start_bin + 1 < self.stop_bin <= self.frequency_bin_count:
            raise ValueError("frequency interval must contain at least two bins")


class WindowedSnrEstimator:
    def __init__(self, config: WindowedSnrConfig | None = None):
        self.config = config if config is not None else WindowedSnrConfig()
        self.reset()

    def reset(self):
        shape = (self.config.stop_bin - self.config.start_bin, self.config.channel_count)
        self._minimum = np.zeros(shape)
        self._recent_minimum = np.zeros(shape)
        self._maximum = np.full(shape, 32768.0)
        self._recent_maximum = np.full(shape, 32768.0)
        self._noise_power = np.zeros(shape[1])
        self._speech_power = np.zeros(shape[1])
        self._noise_age = self._speech_age = 0
        self._update_count = 0
        self._selected_beam = 0
        self._wake_active = False

    @property
    def noise_power(self):
        return self._noise_power.copy()

    @property
    def speech_power(self):
        return self._speech_power.copy()

    @property
    def selected_beam(self):
        return self._selected_beam

    @property
    def wake_active(self):
        return self._wake_active

    def evaluate(self, spectrum_frame, *, wake_active=False, beam_index=None):
        config = self.config
        spectrum = spectral_frame(spectrum_frame, config.frequency_bin_count, config.channel_count)
        if not isinstance(wake_active, (bool, np.bool_)):
            raise ValueError("wake_active must be boolean")
        selected = self._selected_beam if beam_index is None else beam_index
        nonnegative_integer(selected, "beam_index")
        if selected >= config.channel_count:
            raise ValueError("beam_index is outside the input channels")
        self._selected_beam, self._wake_active = int(selected), bool(wake_active)
        if wake_active:
            if not self._update_count:
                return 0.0
            return float(self._speech_power[selected] / self._noise_power[selected])
        power = np.abs(spectrum[config.start_bin:config.stop_bin]) ** 2 + 1
        smooth = lfilter([.25, .5, .25], [1], power, axis=0)
        smooth[1:-1] = smooth[2:]
        smooth[[0, -1]] = power[[0, -1]]
        if not self._update_count:
            self._minimum = smooth.copy()
            self._maximum = smooth.copy()
            self._recent_minimum = smooth.copy()
            self._recent_maximum = smooth.copy()
            self._noise_age = self._speech_age = 1
        else:
            self._minimum = np.minimum(self._minimum, smooth)
            self._recent_minimum = np.minimum(self._recent_minimum, smooth)
            self._maximum = np.maximum(self._maximum, smooth)
            self._recent_maximum = np.maximum(self._recent_maximum, smooth)
            self._noise_age += 1
            self._speech_age += 1
        if self._noise_age == config.noise_window_frames:
            self._minimum = np.minimum(self._recent_minimum, smooth)
            self._recent_minimum = smooth.copy()
            self._noise_age = 0
        if self._speech_age == config.speech_window_frames:
            self._maximum = np.maximum(self._recent_maximum, smooth)
            self._recent_maximum = smooth.copy()
            self._speech_age = 0
        self._noise_power = np.mean(self._minimum, axis=0)
        strongest = np.sort(self._maximum, axis=0)[::-1]
        count = len(strongest) // 2
        self._speech_power = np.mean(strongest[1:count + 1], axis=0)
        self._update_count += 1
        return 0.0
