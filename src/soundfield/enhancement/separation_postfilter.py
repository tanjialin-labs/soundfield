"""Multichannel enhancement guided by external source-separation gains."""

from dataclasses import dataclass, field

import numpy as np

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, spectral_frame, unit_interval_array
from ..numerics import spectral_amplitude_gain
from .minimum_statistics import NoiseTrackingConfig, _MinimumStatistics, _posterior_ratio, _speech_probability


@dataclass(frozen=True)
class SeparationPostfilterConfig:
    frequency_bin_count: int = 257
    channel_count: int = 2
    gain_memory: float = .4
    prior_memory: float = .92
    gain_evidence_low: float = 0.0
    gain_evidence_high: float = .8
    prior_snr_floor: float = .0031623
    amplitude_gain_floor: float = .5
    posterior_snr_low: float = 1.0
    posterior_snr_high: float = 4.6
    detection_ratio: float = 1.67
    speech_start_bin: int = 8
    speech_stop_bin: int = 113
    speech_band_threshold: float = .25
    noise_bias: float = 1.98
    absence_gain: float = .056234
    power_epsilon: float = 1e-8
    noise_tracking: NoiseTrackingConfig = field(default_factory=NoiseTrackingConfig)

    def __post_init__(self):
        for name in ("frequency_bin_count", "channel_count", "speech_stop_bin"):
            positive_integer(getattr(self, name), name)
        nonnegative_integer(self.speech_start_bin, "speech_start_bin")
        if self.channel_count < 2:
            raise ValueError("at least two channels are required")
        if not self.speech_start_bin < self.speech_stop_bin <= self.frequency_bin_count:
            raise ValueError("speech band must fit the frequency bins")
        for name in ("gain_memory", "prior_memory", "gain_evidence_low", "gain_evidence_high",
                     "amplitude_gain_floor", "speech_band_threshold", "absence_gain"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        for name in ("prior_snr_floor", "posterior_snr_low", "posterior_snr_high", "detection_ratio", "noise_bias", "power_epsilon"):
            finite_scalar(getattr(self, name), name, minimum=0)
        if self.power_epsilon == 0 or self.prior_snr_floor == 0:
            raise ValueError("power_epsilon and prior_snr_floor must be positive")
        if self.gain_evidence_low >= self.gain_evidence_high or self.posterior_snr_low >= self.posterior_snr_high:
            raise ValueError("upper evidence bounds must exceed lower bounds")
        if not isinstance(self.noise_tracking, NoiseTrackingConfig):
            raise TypeError("noise_tracking must be a NoiseTrackingConfig")


class SeparationGuidedPostfilter:
    """Combine channel spectra using supplied per-source amplitude gains.

    Spectrum, separation gains, and output have shape (bins, channels).
    Separation is supplied by the caller; the filter does not estimate it.
    """

    def __init__(self, config: SeparationPostfilterConfig | None = None):
        self._config = config if config is not None else SeparationPostfilterConfig()
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def mixing_gain(self):
        return self._mixing_gain.copy()

    @property
    def speech_probability(self):
        return self._speech_probability.copy()

    def reset(self):
        shape = (self.config.frequency_bin_count, self.config.channel_count)
        self._other_tracker = _MinimumStatistics(shape, self.config.noise_tracking)
        self._target_power = np.zeros(shape)
        self._smoothed_separation_gain = np.zeros(shape)
        self._noise_power = np.zeros(shape)
        self._posterior_snr = np.ones(shape)
        self._conditional_gain = np.ones(shape)
        self._mixing_gain = np.ones(shape)
        self._speech_probability = np.zeros(shape)

    def process_frame(self, spectrum_frame, separation_gain):
        config = self.config
        spectrum = spectral_frame(spectrum_frame, config.frequency_bin_count, config.channel_count)
        gain = unit_interval_array(separation_gain, "separation_gain", spectrum.shape)
        summed = np.sum(spectrum, axis=1, keepdims=True)
        other_power = np.abs(summed - spectrum)**2
        power = np.abs(spectrum)**2
        memory = config.noise_tracking.power_memory
        self._target_power = memory * self._target_power + (1 - memory) * power
        self._other_tracker.update(other_power)
        tracker = self._other_tracker
        local_ratio = tracker.power / np.maximum(tracker.noise, config.power_epsilon)
        evidence = np.divide(self._target_power, tracker.power, out=np.zeros_like(power), where=tracker.power > 0)
        evidence_sum = np.sum(evidence, axis=1, keepdims=True)
        evidence = np.divide(evidence, evidence_sum, out=np.zeros_like(evidence), where=evidence_sum > 0)
        self._smoothed_separation_gain = config.gain_memory * self._smoothed_separation_gain + (1 - config.gain_memory) * gain
        weighted_gain = np.sum(evidence * self._smoothed_separation_gain, axis=1, keepdims=True)
        presence = np.broadcast_to(np.clip((config.gain_evidence_high - weighted_gain)
                                          / (config.gain_evidence_high - config.gain_evidence_low), 0, 1), spectrum.shape).copy()
        presence[np.any(tracker.power == 0, axis=1)] = 0
        presence[local_ratio <= config.detection_ratio] = 0
        band_presence = np.mean(presence[config.speech_start_bin:config.speech_stop_bin], axis=0)
        local_snr = other_power / (tracker.noise + 1e-10)
        absence = np.where((local_snr > 1) & (band_presence > config.speech_band_threshold),
                           np.maximum((config.posterior_snr_high - local_snr)
                                      / (config.posterior_snr_high - config.posterior_snr_low), 1 - presence), 1)
        absence = np.clip(absence, 0, 1)
        posterior = _posterior_ratio(other_power, self._noise_power)
        prior = np.maximum(config.prior_memory * self._conditional_gain**2 * self._posterior_snr
                           + (1 - config.prior_memory) * np.maximum(posterior - 1, 0), config.prior_snr_floor)
        exponent = posterior * prior / (1 + prior)
        self._conditional_gain = np.maximum(prior * spectral_amplitude_gain(exponent) / (1 + prior), config.amplitude_gain_floor)
        probability = _speech_probability(absence, prior, exponent)
        self._mixing_gain = self._conditional_gain**probability * config.absence_gain**(1 - probability)
        memory = config.noise_tracking.noise_memory + (1 - config.noise_tracking.noise_memory) * probability
        self._noise_power = memory * self._noise_power + config.noise_bias * (1 - memory) * other_power
        self._posterior_snr = posterior
        self._speech_probability = probability
        return ((1 - self._mixing_gain) * spectrum + self._mixing_gain * summed) * gain
