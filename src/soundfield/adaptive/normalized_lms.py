"""Frequency-domain normalized least-mean-square cancellation."""

from dataclasses import dataclass

import numpy as np

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, spectral_frame, unit_interval_array


@dataclass(frozen=True)
class NlmsConfig:
    frame_length_samples: int = 512
    sample_rate_hz: int = 16000
    reference_count: int = 1
    filter_length_frames: int = 16
    step_size: float = 0.8
    reference_regularization: float = 2.0
    error_regularization: float = 16.0
    posterior_error_scale: float = 0.5
    frequency_dependent_floor: bool = True
    coefficient_limit: float | None = None
    limit_start_bin: int = 0
    limit_stop_bin: int | None = None

    def __post_init__(self):
        for name in ("frame_length_samples", "sample_rate_hz", "reference_count", "filter_length_frames"):
            positive_integer(getattr(self, name), name)
        if self.frame_length_samples < 4 or self.frame_length_samples % 2:
            raise ValueError("frame_length_samples must be even and at least 4")
        if self.frame_length_samples > 2 * self.sample_rate_hz:
            raise ValueError("frame duration exceeds the power smoothing range")
        for name in ("step_size", "reference_regularization", "error_regularization", "posterior_error_scale"):
            finite_scalar(getattr(self, name), name, minimum=0)
        if not isinstance(self.frequency_dependent_floor, bool):
            raise ValueError("frequency_dependent_floor must be boolean")
        if self.coefficient_limit is not None:
            finite_scalar(self.coefficient_limit, "coefficient_limit", minimum=0)
        stop = self.frequency_bin_count if self.limit_stop_bin is None else self.limit_stop_bin
        nonnegative_integer(self.limit_start_bin, "limit_start_bin")
        positive_integer(stop, "limit_stop_bin")
        if not self.limit_start_bin < stop <= self.frequency_bin_count:
            raise ValueError("coefficient limiting interval is outside the frequency bins")

    @property
    def frequency_bin_count(self):
        return self.frame_length_samples // 2 + 1


class NormalizedLmsCanceller:
    """Cancel reference components from a single target spectrum.

    Inputs and output retain their channel axes. Update strength has shape
    (bins, 1); zero freezes coefficients while input and power history advance.
    """

    def __init__(self, config: NlmsConfig | None = None):
        self._config = config if config is not None else NlmsConfig()
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def coefficients(self):
        return self._coefficients.copy()

    def reset(self):
        config = self.config
        shape = (config.frequency_bin_count, config.reference_count, config.filter_length_frames)
        self._coefficients = np.zeros(shape, dtype=np.complex128)
        self._reference_history = np.zeros(shape, dtype=np.complex128)
        self._power_history = np.zeros(shape)
        self._smoothed_reference_power = np.zeros(shape[:2])
        self._error_power = np.zeros(shape[:2])
        self._regularization_floor = np.full(shape[:2], 0.02)
        self._maximum_regularization = 2e-3 * (2 * config.frame_length_samples)**2
        if config.frequency_dependent_floor:
            minimum = 7e-9 * (2 * config.frame_length_samples)**2
            ramp = np.maximum(1 - np.arange(shape[0]) / shape[0], 2 / shape[2])
            self._regularization_floor[:] = (ramp * shape[2] * minimum)[:, None]

    def process_frame(self, target_spectrum, reference_spectrum, update_strength):
        config = self.config
        target = spectral_frame(target_spectrum, config.frequency_bin_count, 1)
        reference = spectral_frame(reference_spectrum, config.frequency_bin_count, config.reference_count)
        strength = unit_interval_array(update_strength, "update_strength", target.shape)
        regularization = np.minimum(np.maximum(
            self._error_power * config.error_regularization
            + self._smoothed_reference_power * config.reference_regularization,
            self._regularization_floor), self._maximum_regularization)
        self._reference_history[:, :, 1:] = self._reference_history[:, :, :-1].copy()
        self._reference_history[:, :, 0] = reference
        estimate = np.sum(np.sum(self._reference_history * self._coefficients, axis=1), axis=1)[:, None]
        residual = target - estimate
        self._power_history[:, :, 1:] = self._power_history[:, :, :-1].copy()
        self._power_history[:, :, 0] = np.abs(reference)**2
        reference_power = np.sum(self._power_history, axis=2)
        normalized_residual = residual / (reference_power + regularization)
        self._coefficients += (config.step_size * strength[:, :, None]
                               * normalized_residual[:, :, None] * self._reference_history.conj())
        if config.coefficient_limit is not None:
            section = self._coefficients[config.limit_start_bin:config.limit_stop_bin]
            magnitudes = np.abs(section)
            scale = np.ones_like(magnitudes)
            np.divide(config.coefficient_limit, magnitudes, out=scale,
                      where=magnitudes > config.coefficient_limit)
            section *= scale
        posterior = residual - config.posterior_error_scale * reference_power * normalized_residual
        self._error_power = np.abs(posterior)**2
        increasing = reference_power > self._smoothed_reference_power
        rate = np.where(increasing, 0.5, 0.0031) * config.frame_length_samples / config.sample_rate_hz
        self._smoothed_reference_power = (1 - rate) * self._smoothed_reference_power + rate * reference_power
        return residual
