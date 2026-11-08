"""Partitioned reference spectra and regularized cross-channel inverse power."""

from dataclasses import dataclass

import numpy as np

from ..contracts import positive_integer, nonnegative_integer, finite_scalar, audio_array, real_array


@dataclass(frozen=True)
class EchoFilterConfig:
    hop_length_samples: int = 512
    reference_count: int = 1
    partition_count: int = 8
    method: str = 'constrained'
    adaptation_rate: float = 0.03
    spectral_adaptation_rate: float = 0.4
    inverse_time_constant_seconds: float = 0.5
    cross_power_scale: float = 0.99
    inverse_regularization_floor: float = 0.01
    residual_regularization_scale: float = 1.0
    power_attack: float = 0.016
    power_release: float = 9.92e-5
    residual_suppression: float = 0.5
    normalization_scale: float = 2.0
    normalization_floor: float = 0.007340032
    normalization_ceiling: float = 2097.152
    ignore_reference_activity: bool = False
    refinement_partition_count: int = 3
    refinement_reference_indices: tuple = (0,)

    def __post_init__(self):
        for name in ('hop_length_samples', 'reference_count', 'partition_count'):
            positive_integer(getattr(self, name), name)
        nonnegative_integer(self.refinement_partition_count, 'refinement_partition_count')
        if self.method not in ('constrained', 'spectral'):
            raise ValueError('method must be constrained or spectral')
        for name in ('adaptation_rate', 'spectral_adaptation_rate', 'cross_power_scale', 'power_attack',
                     'power_release', 'residual_suppression'):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        for name in ('inverse_time_constant_seconds', 'inverse_regularization_floor', 'normalization_floor',
                     'normalization_ceiling'):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).eps)
        for name in ('residual_regularization_scale', 'normalization_scale'):
            finite_scalar(getattr(self, name), name, minimum=0)
        if self.inverse_time_constant_seconds < self.hop_length_samples/16000:
            raise ValueError('inverse time constant must span at least one hop')
        if any(index < 0 or index >= self.reference_count for index in self.refinement_reference_indices):
            raise ValueError('refinement reference index is out of range')

    @property
    def frequency_bin_count(self):
        return self.hop_length_samples+1


class ReferenceSpectrumBuffer:
    def __init__(self, config=EchoFilterConfig()):
        self.config = config
        self.window = np.sqrt(np.hanning(config.hop_length_samples*2))
        self.reset()

    def reset(self):
        config = self.config
        bins, references = config.frequency_bin_count, config.reference_count
        self.audio_history = np.zeros((2*config.hop_length_samples, references))
        self.spectrum = np.zeros((bins, references), complex)
        self.partition_spectra = np.zeros((bins, references*config.partition_count), complex)
        self.windowed_mono_history = np.zeros((bins, config.partition_count), complex)
        self.refinement_history = np.zeros((bins, len(config.refinement_reference_indices)*config.refinement_partition_count), complex)
        self.partition_power = np.zeros(self.partition_spectra.shape)
        self.summed_power = np.zeros(bins)
        self.smoothed_power = np.zeros(bins)
        self.covariance = np.tile(np.eye(references, dtype=complex), (bins, 1, 1))
        self.inverse_covariance = self.covariance.copy()
        self.is_active = False
        self.active_frame_count = 0

    def process_block(self, reference_audio, is_reference_active, previous_residual_power):
        config = self.config
        reference = audio_array(reference_audio)
        residual_power = real_array(previous_residual_power, 'previous_residual_power', 1)
        if (reference.shape != (config.hop_length_samples, config.reference_count)
                or residual_power.shape != (config.frequency_bin_count,) or np.any(residual_power < 0)):
            raise ValueError('reference block or residual power has an invalid shape or value')
        hop, references = config.hop_length_samples, config.reference_count
        self.audio_history[:hop], self.audio_history[hop:] = self.audio_history[hop:].copy(), reference
        self.is_active = bool(is_reference_active or config.ignore_reference_activity)
        if not self.is_active:
            self.refinement_history.fill(0)
            return
        values = self.audio_history if config.method == 'constrained' else self.audio_history*self.window[:, None]
        self.spectrum = np.fft.rfft(values, axis=0)
        self.partition_spectra[:, references:] = self.partition_spectra[:, :-references]
        self.partition_spectra[:, :references] = self.spectrum
        self.windowed_mono_history[:, 1:] = self.windowed_mono_history[:, :-1]
        self.windowed_mono_history[:, 0] = np.fft.rfft(self.audio_history.mean(axis=1)*self.window)
        selected = config.refinement_reference_indices
        if selected and config.refinement_partition_count:
            count = len(selected)
            self.refinement_history[:, count:] = self.refinement_history[:, :-count]
            self.refinement_history[:, :count] = np.fft.rfft(self.audio_history[:, selected]*self.window[:, None], axis=0)
        if self.active_frame_count:
            step = np.where(self.summed_power > self.smoothed_power, config.power_attack, config.power_release)
            self.smoothed_power = (1-step)*self.smoothed_power+step*self.summed_power
        self.partition_power[:, references:] = self.partition_power[:, :-references]
        self.partition_power[:, :references] = np.abs(self.spectrum)**2
        self.summed_power = self.partition_power.sum(axis=1)
        if config.method == 'constrained':
            smoothing = 1-hop/(16000*config.inverse_time_constant_seconds)
            current = self.spectrum.conj()[:, :, None]*self.spectrum[:, None, :]
            self.covariance = smoothing*self.covariance+(1-smoothing)*current
            regularized = self.covariance*config.cross_power_scale
            indices = np.arange(references)
            diagonal = self.covariance[:, indices, indices]
            floor = np.maximum(config.inverse_regularization_floor, config.residual_regularization_scale*residual_power)
            regularized[:, indices, indices] = diagonal+floor[:, None]*np.exp(-diagonal)
            self.inverse_covariance = np.linalg.inv(regularized)
        self.active_frame_count += 1
