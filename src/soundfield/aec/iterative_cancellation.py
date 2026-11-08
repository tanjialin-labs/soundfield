"""Iterative single-reference echo adaptation with speech-presence suppression."""

from dataclasses import dataclass

import numpy as np

from ..contracts import positive_integer, finite_scalar, audio_array
from .partitioned_filter import EchoSpectrumResult
from .communication_postfilter import CommunicationEchoPostfilter, CommunicationPostfilterConfig
from .noise_tracking import EchoNoiseConfig


@dataclass(frozen=True)
class IterativeEchoConfig:
    hop_length_samples: int = 256
    partition_count: int = 12
    adaptation_rate: float = 2.0
    power_smoothing: float = .98
    activity_amplitude_threshold: float = .0015
    normalization_scale: float = 12.0
    iterations: int = 2
    constrain_impulse_response: bool = True

    def __post_init__(self):
        for name in ('hop_length_samples', 'partition_count', 'iterations'):
            positive_integer(getattr(self, name), name)
        finite_scalar(self.power_smoothing, 'power_smoothing', minimum=0, maximum=1)
        for name in ('adaptation_rate', 'activity_amplitude_threshold', 'normalization_scale'):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).eps)


class IterativeEchoCanceller:
    def __init__(self, config=IterativeEchoConfig(), *, postfilter_config=None):
        self.config = config
        hop, bins = config.hop_length_samples, config.hop_length_samples+1
        self.window = np.sqrt(np.hanning(2*hop))
        self.normalization_floor = np.maximum(1-np.arange(bins)/bins, 2/config.partition_count)*config.partition_count*7e-9*(2*hop)**2
        if postfilter_config is None:
            postfilter_config = CommunicationPostfilterConfig(
                noise=EchoNoiseConfig(frequency_bin_count=bins, speech_bins=(min(3, bins-1), min(127, bins)), absence_bias=.2),
                leakage_scale=4, minimum_leakage=0, leakage_regularization=1e-5,
                activity_history_frames=40, reference_energy_threshold=config.activity_amplitude_threshold,
                reference_energy_smoothing=0, initial_reference_energy=0, enable_residual_gate=False,
                residual_energy_intervals=((0, bins),), reference_threshold_bin_count=min(66, bins-1),
                noise_output_scale=1, highpass_bin_count=0, noise_feedback='combined', inactive_gain_policy='bypass')
        if postfilter_config.noise.frequency_bin_count != bins:
            raise ValueError('postfilter bins must match the echo filter')
        self.postfilter = CommunicationEchoPostfilter(postfilter_config)
        self.reset()

    def reset(self):
        hop, partitions = self.config.hop_length_samples, self.config.partition_count
        self.reference_history = np.zeros(2*hop)
        self.partition_spectra = np.zeros((hop+1, partitions), complex)
        self.adaptive_weights = np.zeros_like(self.partition_spectra)
        self.smoothed_reference_power = np.zeros(hop+1)
        self.previous_residual, self.previous_echo = np.zeros(hop), np.zeros(hop)
        self.postfilter.reset()
        self.last_linear_result = None

    def process_block(self, microphone_audio, reference_audio):
        microphone, reference = audio_array(microphone_audio), audio_array(reference_audio)
        config, hop = self.config, self.config.hop_length_samples
        if microphone.shape != (hop, 1) or reference.shape != (hop, 1):
            raise ValueError('iterative cancellation requires one microphone and one reference hop')
        self.reference_history[:hop], self.reference_history[hop:] = self.reference_history[hop:].copy(), reference[:, 0]
        reference_spectrum = np.fft.rfft(self.reference_history)
        self.partition_spectra[:, 1:] = self.partition_spectra[:, :-1]
        self.partition_spectra[:, 0] = reference_spectrum
        self.smoothed_reference_power = config.power_smoothing*self.smoothed_reference_power+(1-config.power_smoothing)*np.abs(reference_spectrum)**2
        activity_energy = float(np.mean(np.abs(reference[:, 0]-reference[:, 0].mean())))
        estimated = np.fft.irfft(np.sum(self.partition_spectra*self.adaptive_weights, axis=1), n=2*hop)[hop:]
        residual = microphone[:, 0]-estimated
        if activity_energy > config.activity_amplitude_threshold:
            for _ in range(config.iterations):
                error_spectrum = np.fft.rfft(np.r_[np.zeros(hop), residual])
                update = self.partition_spectra.conj()*error_spectrum[:, None]/(
                    self.smoothed_reference_power+self.normalization_floor*config.normalization_scale)[:, None]
                self.adaptive_weights += config.adaptation_rate*(1-config.power_smoothing)*update
                if config.constrain_impulse_response:
                    impulse = np.fft.irfft(self.adaptive_weights, n=2*hop, axis=0)
                    impulse[hop:] = 0
                    self.adaptive_weights = np.fft.rfft(impulse, axis=0)
                estimated = np.fft.irfft(np.sum(self.partition_spectra*self.adaptive_weights, axis=1), n=2*hop)[hop:]
                residual = microphone[:, 0]-estimated
        residual_spectrum = np.fft.rfft(np.r_[self.previous_residual, residual]*self.window)[:, None]
        echo_spectrum = np.fft.rfft(np.r_[self.previous_echo, estimated]*self.window)[:, None]
        self.previous_residual, self.previous_echo = residual.copy(), estimated.copy()
        self.last_linear_result = EchoSpectrumResult(residual_spectrum.copy(), echo_spectrum.copy())
        return self.postfilter.process_frame(residual_spectrum, echo_spectrum, reference_spectrum[:, None],
                                             reference_activity_energy=activity_energy)
