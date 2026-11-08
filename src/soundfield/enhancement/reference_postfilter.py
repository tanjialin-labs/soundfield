"""Noise-reference spectral enhancement using minimum-statistics estimates."""

from dataclasses import dataclass, field

import numpy as np

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, spectral_frame
from ..numerics import spectral_amplitude_gain
from .minimum_statistics import NoiseTrackingConfig, _MinimumStatistics, _posterior_ratio, _speech_probability


@dataclass(frozen=True)
class ReferencePostfilterConfig:
    frequency_bin_count: int = 257
    amplitude_gain_floor: float = .5
    detection_ratio: float = 1.67
    spatial_ratio_low: float = 1.0
    spatial_ratio_high: float = 3.0
    speech_start_bin: int = 8
    speech_stop_bin: int = 113
    noise_tracking: NoiseTrackingConfig = field(default_factory=NoiseTrackingConfig)

    def __post_init__(self):
        positive_integer(self.frequency_bin_count, "frequency_bin_count")
        nonnegative_integer(self.speech_start_bin, "speech_start_bin")
        positive_integer(self.speech_stop_bin, "speech_stop_bin")
        if not self.speech_start_bin < self.speech_stop_bin <= self.frequency_bin_count:
            raise ValueError("speech band must fit the frequency bins")
        finite_scalar(self.amplitude_gain_floor, "amplitude_gain_floor", minimum=0, maximum=1)
        finite_scalar(self.detection_ratio, "detection_ratio", minimum=0)
        finite_scalar(self.spatial_ratio_low, "spatial_ratio_low", minimum=0)
        finite_scalar(self.spatial_ratio_high, "spatial_ratio_high", minimum=0)
        if self.spatial_ratio_low >= self.spatial_ratio_high:
            raise ValueError("spatial_ratio_high must exceed spatial_ratio_low")
        if not isinstance(self.noise_tracking, NoiseTrackingConfig):
            raise TypeError("noise_tracking must be a NoiseTrackingConfig")


class NoiseReferencePostfilter:
    """Enhance a single target spectrum using an explicit noise reference.

    Both spectra have shape (bins, 1). The first frame initializes the final
    noise estimate and passes through unchanged. Power units are squared
    spectral amplitude; all returned gains multiply amplitude.
    """

    def __init__(self, config: ReferencePostfilterConfig | None = None):
        self._config = config if config is not None else ReferencePostfilterConfig()
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def gain(self):
        return self._gain.copy()

    @property
    def speech_probability(self):
        return self._speech_probability.copy()

    def reset(self):
        shape = (self.config.frequency_bin_count, 1)
        self._target_tracker = _MinimumStatistics(shape, self.config.noise_tracking)
        self._reference_tracker = _MinimumStatistics(shape, self.config.noise_tracking)
        self._noise_power = np.zeros(shape)
        self._posterior_snr = np.ones(shape)
        self._conditional_gain = np.ones(shape)
        self._gain = np.ones(shape)
        self._speech_probability = np.zeros(shape)
        self._initialized = False

    def process_frame(self, target_spectrum, noise_spectrum):
        config = self.config
        target = spectral_frame(target_spectrum, config.frequency_bin_count, 1)
        reference = spectral_frame(noise_spectrum, config.frequency_bin_count, 1)
        power = np.abs(target)**2
        self._target_tracker.update(power)
        self._reference_tracker.update(np.abs(reference)**2)
        if not self._initialized:
            self._noise_power = power.copy()
            self._initialized = True
            return target.copy()
        target_track, reference_track = self._target_tracker, self._reference_tracker
        target_ratio = target_track.power / np.maximum(target_track.noise, 1e-8)
        reference_ratio = reference_track.power / np.maximum(reference_track.noise, 1e-8)
        spatial_ratio = (target_track.power - target_track.noise) / np.maximum(reference_track.power - reference_track.noise, 1e-8)
        presence = np.clip((spatial_ratio - config.spatial_ratio_low)
                           / (config.spatial_ratio_high - config.spatial_ratio_low), 0, 1)
        presence[reference_ratio <= config.detection_ratio] = 1
        presence[target_ratio <= config.detection_ratio] = 0
        band_presence = np.mean(presence[config.speech_start_bin:config.speech_stop_bin], axis=0)
        local_snr = power / (target_track.noise + 1e-10)
        absence = np.where((local_snr > 1) & (band_presence > .25),
                           np.maximum((4.6 - local_snr) / 3.6, 1 - presence), 1)
        absence = np.clip(absence, 0, 1)
        posterior = _posterior_ratio(power, self._noise_power)
        prior = np.maximum(.92 * self._conditional_gain**2 * self._posterior_snr
                           + .08 * np.maximum(posterior - 1, 0), 10**(-2.5))
        exponent = posterior * prior / (1 + prior)
        self._conditional_gain = np.maximum(config.amplitude_gain_floor, prior * spectral_amplitude_gain(exponent) / (1 + prior))
        probability = _speech_probability(absence, prior, exponent)
        memory = config.noise_tracking.noise_memory + (1 - config.noise_tracking.noise_memory) * probability
        self._noise_power = memory * self._noise_power + 1.98 * (1 - memory) * power
        self._posterior_snr = posterior
        self._speech_probability = probability
        self._gain = self._conditional_gain**probability * (10**(-.5))**(1 - probability)
        return target * self._gain
