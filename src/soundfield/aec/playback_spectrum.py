"""Playback-mode echo spectra with continuous input conditioning."""

from dataclasses import dataclass, field

import numpy as np

from ..contracts import finite_scalar
from .input_conditioning import EchoInputConditioner, InputConditioningConfig
from .playback_adaptation import PlaybackAdaptiveEchoFilter, PlaybackEchoConfig


@dataclass(frozen=True)
class PlaybackSpectrumConfig:
    linear_filter: PlaybackEchoConfig = field(default_factory=PlaybackEchoConfig)
    preemphasis: float = .9
    residual_suppression: float = .5
    normalization_scale: float = 2
    normalization_ceiling: float = 2097.152
    power_attack: float = .016
    power_release: float = 9.92e-5

    def __post_init__(self):
        for name in ('preemphasis', 'residual_suppression', 'power_attack', 'power_release'):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        finite_scalar(self.normalization_scale, 'normalization_scale', minimum=0)
        finite_scalar(self.normalization_ceiling, 'normalization_ceiling', minimum=np.finfo(float).eps)


@dataclass(frozen=True)
class PlaybackSpectrumResult:
    residual_spectrum: np.ndarray
    echo_spectrum: np.ndarray
    reference_spectrum: np.ndarray
    unfiltered_spectrum: np.ndarray
    normalized_residual: np.ndarray
    reference_power: np.ndarray
    linear_audio: np.ndarray


class PlaybackSpectrumAnalyzer:
    def __init__(self, config=PlaybackSpectrumConfig()):
        self.config = config
        linear = config.linear_filter
        self.filter = PlaybackAdaptiveEchoFilter(linear)
        self.conditioner = EchoInputConditioner(InputConditioningConfig(
            hop_length_samples=linear.hop_length_samples, reference_count=linear.reference_count,
            preemphasis=config.preemphasis))
        self.window = np.sqrt(np.hanning(2*linear.hop_length_samples))
        self.reset()

    def reset(self):
        self.filter.reset()
        self.conditioner.reset()
        hop = self.config.linear_filter.hop_length_samples
        self.residual_history = np.zeros(2*hop)
        self.echo_history = np.zeros(2*hop)
        self.previous_residual_power = np.zeros(hop+1)
        self.smoothed_reference_power = np.zeros(hop+1)

    def process_block(self, microphone_audio, reference_audio, *, playback_mode=0):
        microphone, reference = self.conditioner.process_block(microphone_audio, reference_audio)
        result = self.filter.process_block(microphone, reference, playback_mode=playback_mode,
                                           is_clipped=self.conditioner.is_clipped)
        hop = self.config.linear_filter.hop_length_samples
        for history, audio in ((self.residual_history, result.residual_audio), (self.echo_history, result.estimated_echo_audio)):
            history[:hop] = history[hop:]
            history[hop:] = audio[:, 0]
        residual = np.fft.rfft(self.residual_history*self.window)
        echo = np.fft.rfft(self.echo_history*self.window)
        reference_spectrum = self.filter.partition_spectra[:, :reference.shape[1]].copy()
        config = self.config
        normalization = np.clip((self.previous_residual_power+self.smoothed_reference_power)*config.normalization_scale,
                                self.filter.update_floor, config.normalization_ceiling)
        power = np.sum(np.abs(self.filter.partition_spectra)**2, axis=1)
        rate = np.where(power > self.smoothed_reference_power, config.power_attack, config.power_release)
        self.smoothed_reference_power += rate*(power-self.smoothed_reference_power)
        normalized = residual/(power+normalization)
        suppressed = residual-config.residual_suppression*power*normalized
        self.previous_residual_power = np.abs(suppressed)**2
        return PlaybackSpectrumResult(suppressed[:, None], echo[:, None], reference_spectrum,
                                      residual[:, None], normalized[:, None], power,
                                      result.residual_audio)
