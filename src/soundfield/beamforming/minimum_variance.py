"""Minimum-variance beamforming from recursively estimated spatial statistics."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, spectral_frame


@dataclass(frozen=True)
class MvdrConfig:
    frequency_bin_count: int = 257
    microphone_count: int = 2
    transition_start_bin: int = 33
    transition_stop_bin: int = 52
    speech_memory_low: float = .98
    speech_memory_high: float = .9799
    noise_memory_low: float = .98
    noise_memory_high: float = .85
    diagonal_loading: float = 1e-8
    relative_loading: float = 0.0
    reference_microphone: int | None = None

    def __post_init__(self):
        positive_integer(self.frequency_bin_count, "frequency_bin_count")
        positive_integer(self.microphone_count, "microphone_count")
        nonnegative_integer(self.transition_start_bin, "transition_start_bin")
        positive_integer(self.transition_stop_bin, "transition_stop_bin")
        if not self.transition_start_bin + 1 < self.transition_stop_bin <= self.frequency_bin_count:
            raise ValueError("smoothing transition must contain at least two frequency bins")
        for name in ("speech_memory_low", "speech_memory_high", "noise_memory_low", "noise_memory_high"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        finite_scalar(self.diagonal_loading, "diagonal_loading", minimum=np.finfo(float).tiny)
        finite_scalar(self.relative_loading, "relative_loading", minimum=0)
        if self.reference_microphone is not None:
            nonnegative_integer(self.reference_microphone, "reference_microphone")
            if self.reference_microphone >= self.microphone_count:
                raise ValueError("reference_microphone must identify an input channel")


class MvdrBeamformer:
    """Process (bins, microphones) spectra and return one channel.

    Speech and noise estimates are explicit inputs in the same amplitude units.
    With a reference microphone, the output preserves the principal speech
    component on that channel. An unresolved speech direction passes through
    the reference channel. Steering vectors are unit length in both modes.
    """

    def __init__(self, config: MvdrConfig | None = None):
        self._config = config if config is not None else MvdrConfig()
        self._speech_memory = self._memory(self.config.speech_memory_low, self.config.speech_memory_high)
        self._noise_memory = self._memory(self.config.noise_memory_low, self.config.noise_memory_high)
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def beamforming_coefficients(self):
        return self._coefficients.copy()

    @property
    def steering_vectors(self):
        return self._steering.copy()

    def _memory(self, low, high):
        start, stop = self.config.transition_start_bin, self.config.transition_stop_bin
        return np.concatenate((np.full(start, low), np.linspace(low, high, stop - start),
                               np.full(self.config.frequency_bin_count - stop, high)))

    def reset(self):
        bins, microphones = self.config.frequency_bin_count, self.config.microphone_count
        self._speech_covariance = np.broadcast_to(np.eye(microphones), (bins, microphones, microphones)).astype(complex).copy()
        self._noise_covariance = self._speech_covariance.copy()
        self._steering = np.zeros((bins, microphones), dtype=np.complex128)
        self._coefficients = np.zeros_like(self._steering)

    def process_frame(self, spectrum_frame, speech_spectrum, noise_spectrum):
        config = self.config
        spectrum, speech, noise = [spectral_frame(value, config.frequency_bin_count, config.microphone_count)
                                   for value in (spectrum_frame, speech_spectrum, noise_spectrum)]
        speech_memory = self._speech_memory[:, None, None]
        noise_memory = self._noise_memory[:, None, None]
        speech_covariance = speech_memory * self._speech_covariance + (1 - speech_memory) * speech[:, :, None] * speech[:, None, :].conj()
        noise_covariance = noise_memory * self._noise_covariance + (1 - noise_memory) * noise[:, :, None] * noise[:, None, :].conj()
        loading = config.diagonal_loading * np.eye(config.microphone_count)
        reference = config.reference_microphone
        if reference is None:
            steering = np.empty_like(self._steering)
            for frequency in range(config.frequency_bin_count):
                loaded_speech = speech_covariance[frequency] + loading
                left_vectors, _, _ = svd(loaded_speech, lapack_driver="gesvd", check_finite=False)
                direction = left_vectors[:, 0]
                orientation = np.sign(np.real(np.vdot(direction, loaded_speech[:, 0])))
                if orientation != 0:
                    direction = orientation * direction
                steering[frequency] = direction
        else:
            eigenvalues, vectors = np.linalg.eigh(speech_covariance)
            steering = vectors[:, :, -1]
            unresolved = np.zeros(config.frequency_bin_count, dtype=bool)
            if config.microphone_count > 1:
                gap = eigenvalues[:, -1] - eigenvalues[:, -2]
                unresolved = gap <= 32 * np.finfo(float).eps * eigenvalues[:, -1]
        # Keep the diagonal increment representable even for coherent high-level inputs.
        mean_power = np.diagonal(noise_covariance, axis1=1, axis2=2).real.mean(axis=1)
        relative_floor = max(config.relative_loading, 32 * config.microphone_count * np.finfo(float).eps)
        increment = np.maximum(config.diagonal_loading, relative_floor * mean_power)
        loaded_noise = noise_covariance + increment[:, None, None] * np.eye(config.microphone_count)
        inverse_direction = np.linalg.solve(loaded_noise, steering[..., None])[..., 0]
        coefficients = inverse_direction / np.sum(steering.conj()*inverse_direction, axis=1, keepdims=True)
        if reference is not None:
            # Back-project the unit-vector response to the reference microphone.
            coefficients *= steering[:, reference:reference + 1].conj()
            coefficients[unresolved] = 0
            coefficients[unresolved, reference] = 1
            steering[unresolved] = 0
            steering[unresolved, reference] = 1
        output = np.sum(spectrum * coefficients.conj(), axis=1, keepdims=True)
        self._speech_covariance = speech_covariance
        self._noise_covariance = noise_covariance
        self._steering, self._coefficients = steering, coefficients
        return output
