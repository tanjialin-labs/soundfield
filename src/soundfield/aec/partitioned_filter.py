"""Frequency-domain echo estimation with constrained or spectral adaptation."""

from dataclasses import dataclass

import numpy as np

from ..contracts import real_array
from .reference_spectrum import EchoFilterConfig, ReferenceSpectrumBuffer


@dataclass(frozen=True)
class EchoSpectrumResult:
    residual_spectrum: np.ndarray
    echo_spectrum: np.ndarray


class PartitionedEchoFilter:
    def __init__(self, config=EchoFilterConfig()):
        self.config = config
        bins = config.frequency_bin_count
        self.normalization_floor = np.maximum(1-np.arange(bins)/bins, 2/config.partition_count)*config.partition_count*config.normalization_floor
        self.reset()

    def reset(self):
        config = self.config
        self.adaptive_weights = np.zeros((config.frequency_bin_count, config.reference_count*config.partition_count), complex)
        self.microphone_history = np.zeros(2*config.hop_length_samples)
        self.residual_history = np.zeros_like(self.microphone_history)
        self.echo_history = np.zeros_like(self.microphone_history)
        self.residual_power = np.zeros(config.frequency_bin_count)
        self.estimated_delay_samples = 0

    def process_block(self, microphone_audio, reference: ReferenceSpectrumBuffer, *, is_clipped=False):
        microphone = real_array(microphone_audio, 'microphone_audio', 1)
        config = self.config
        if microphone.shape != (config.hop_length_samples,) or reference.config != config:
            raise ValueError('microphone block and reference buffer must match the filter configuration')
        hop = config.hop_length_samples
        old_power = self.residual_power.copy()
        if config.method == 'spectral':
            self.microphone_history[:hop] = self.microphone_history[hop:]
            self.microphone_history[hop:] = microphone
        if not reference.is_active:
            if config.method == 'spectral':
                self.residual_history = self.microphone_history.copy()
                self.echo_history.fill(0)
            else:
                self.residual_history[:hop] = self.residual_history[hop:]
                self.residual_history[hop:] = microphone
                self.echo_history[:hop] = self.echo_history[hop:]
                self.echo_history[hop:] = 0
            self.residual_power.fill(0)
            return EchoSpectrumResult(np.fft.rfft(reference.window*self.residual_history)[:, None],
                                      np.fft.rfft(reference.window*self.echo_history)[:, None])
        estimated = np.sum(reference.partition_spectra*self.adaptive_weights, axis=1)
        normalization = np.minimum(np.maximum((old_power+reference.smoothed_power)*config.normalization_scale,
                                               self.normalization_floor), config.normalization_ceiling)
        denominator = reference.summed_power+normalization
        if config.method == 'constrained':
            echo = np.fft.irfft(estimated, n=2*hop)[hop:]
            residual = microphone-echo
            padded = np.r_[np.zeros(hop), residual]
            if not is_clipped:
                error_spectrum = np.fft.rfft(padded)
                self.residual_power = np.abs(error_spectrum)**2
                cross = reference.partition_spectra.conj().reshape(config.frequency_bin_count, config.partition_count, config.reference_count)
                update = np.einsum('bij,btj->bti', reference.inverse_covariance, cross)
                self.adaptive_weights += config.adaptation_rate*update.reshape(self.adaptive_weights.shape)*error_spectrum[:, None]
                impulse = np.fft.irfft(self.adaptive_weights, n=2*hop, axis=0)
                impulse[hop:] = 0
                delay_candidates = impulse[:hop, ::config.reference_count]
                self.estimated_delay_samples = int(np.argmax(delay_candidates.ravel(order='F')))
                self.adaptive_weights = np.fft.rfft(impulse, axis=0)
            self.residual_history[:hop], self.residual_history[hop:] = self.residual_history[hop:].copy(), residual
            self.echo_history[:hop], self.echo_history[hop:] = self.echo_history[hop:].copy(), echo
            raw_residual = np.fft.rfft(reference.window*self.residual_history)
            echo_spectrum = np.fft.rfft(reference.window*self.echo_history)
            output = raw_residual-config.residual_suppression*reference.summed_power*raw_residual/denominator
        else:
            microphone_spectrum = np.fft.rfft(reference.window*self.microphone_history)
            raw_residual = microphone_spectrum-estimated
            normalized = raw_residual/denominator
            output = raw_residual-config.residual_suppression*reference.summed_power*normalized
            mask = old_power > np.abs(microphone_spectrum)**2
            output[mask] = microphone_spectrum[mask]
            if not is_clipped:
                self.adaptive_weights += config.spectral_adaptation_rate*normalized[:, None]*reference.partition_spectra.conj()
            self.residual_power = np.abs(output)**2
            echo_spectrum = estimated
        return EchoSpectrumResult(output[:, None], echo_spectrum[:, None])
