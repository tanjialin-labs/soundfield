"""Residual suppression using power leakage or spectral coherence."""

from dataclasses import dataclass, field

import numpy as np

from ..contracts import finite_scalar, positive_integer, real_array, spectral_frame
from .residual_postfilter import ResidualEchoConfig, ResidualEchoPostfilter


@dataclass(frozen=True)
class PlaybackPostfilterConfig:
    residual: ResidualEchoConfig = field(default_factory=lambda: ResidualEchoConfig(
        maximum_snr=100, enable_energy_gate=False, enable_level_detection=False))
    leakage_method: str = 'coherence'
    minimum_leakage: float = .005
    leakage_scale: float = 2.0
    subtraction_strength: float = 0.0
    subtraction_probability_offset: float = 1.8
    enable_noise_tracking: bool = False
    noise_scale: float = 12.0
    power_memory: float = .8
    noise_memory: float = .95
    presence_memory: float = .2
    minimum_window_frames: int = 50
    noise_detection_ratio: float = 3.0

    def __post_init__(self):
        if not isinstance(self.residual, ResidualEchoConfig) or self.residual.method == 'mask':
            raise ValueError('residual must configure critical-band or spectral gains')
        if self.leakage_method not in ('coherence', 'power'):
            raise ValueError('leakage_method must be coherence or power')
        for name in ('minimum_leakage', 'power_memory', 'noise_memory', 'presence_memory'):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        for name in ('leakage_scale', 'subtraction_strength', 'noise_scale', 'noise_detection_ratio'):
            finite_scalar(getattr(self, name), name, minimum=0)
        finite_scalar(self.subtraction_probability_offset, 'subtraction_probability_offset', minimum=1)
        positive_integer(self.minimum_window_frames, 'minimum_window_frames')


class PlaybackResidualSuppressor(ResidualEchoPostfilter):
    def __init__(self, config=PlaybackPostfilterConfig()):
        self.playback_config = config
        super().__init__(config.residual)

    def reset(self):
        super().reset()
        self.leakage_cross = self.leakage_auto = 1.0
        for name in ('tracked_power', 'minimum_power', 'recent_minimum', 'noise_power', 'presence'):
            setattr(self, name, np.zeros(self.config.frequency_bin_count))

    def _estimate_residual_echo(self, residual, estimated, power, echo):
        config = self.playback_config
        if config.leakage_method == 'coherence':
            residual_echo = super()._estimate_residual_echo(residual, estimated, power, echo)-self.config.noise_floor
        else:
            error_difference, echo_difference = power-self.residual_power, echo-self.echo_power
            cross = 1+np.dot(error_difference, echo_difference)
            auto = np.sqrt(1+np.dot(echo_difference, echo_difference))
            period = (len(power)-1)/self.config.sample_rate_hz
            self.residual_power += period*error_difference
            self.echo_power += period*echo_difference
            memory = min(2*period*echo.sum()/power.sum(), .5*period) if power.sum() else 0
            self.leakage_cross = (1-memory)*self.leakage_cross+memory*cross/auto
            self.leakage_auto = max((1-memory)*self.leakage_auto+memory*auto, 1)
            self.leakage_cross = np.clip(self.leakage_cross, config.minimum_leakage*self.leakage_auto, self.leakage_auto)
            leakage = self.leakage_cross/self.leakage_auto
            residual_echo = echo*(1 if leakage > .5 else config.leakage_scale*leakage)
        if self.config.method == 'spectral':
            if config.enable_noise_tracking:
                if not self.frame_count:
                    self.tracked_power = power.copy()
                    self.minimum_power = power.copy()
                    self.recent_minimum = power.copy()
                    self.noise_power = power.copy()
                else:
                    self.tracked_power = config.power_memory*self.tracked_power+(1-config.power_memory)*power
                    if (self.frame_count+1) % config.minimum_window_frames == 0:
                        self.minimum_power = np.minimum(self.recent_minimum, self.tracked_power)
                        self.recent_minimum = self.tracked_power.copy()
                    else:
                        self.minimum_power = np.minimum(self.minimum_power, self.tracked_power)
                        self.recent_minimum = np.minimum(self.recent_minimum, self.tracked_power)
                    detected = self.tracked_power > config.noise_detection_ratio*self.minimum_power
                    self.presence = config.presence_memory*self.presence+(1-config.presence_memory)*detected
                    memory = config.noise_memory+(1-config.noise_memory)*self.presence
                    self.noise_power = memory*self.noise_power+(1-memory)*power
                residual_echo = config.noise_scale*residual_echo+self.noise_power
            residual_echo += self.config.noise_floor
            if not self.frame_count:
                self.previous_output_power = residual_echo.copy()
        return residual_echo

    def process_frame(self, residual_spectrum, echo_spectrum, *, unfiltered_spectrum=None,
                      reference_power=None, normalized_residual=None):
        config = self.playback_config
        bins = self.config.frequency_bin_count
        if config.subtraction_strength:
            unfiltered = spectral_frame(unfiltered_spectrum, bins, 1)
            normalized = spectral_frame(normalized_residual, bins, 1)
            power = real_array(reference_power, 'reference_power', 1)
            if power.shape != (bins,) or np.any(power < 0):
                raise ValueError('reference_power must be a nonnegative vector matching the spectrum')
        output = super().process_frame(residual_spectrum, echo_spectrum)
        if config.subtraction_strength and self.config.method == 'critical_band':
            residual = unfiltered-config.subtraction_strength*(config.subtraction_probability_offset-self.frame_probability)*power[:, None]*normalized
            output = residual*self.gains[:, None]
        return output
