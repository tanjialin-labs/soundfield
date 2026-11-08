"""Recursive weighted prediction error dereverberation."""

from dataclasses import dataclass, field

import numpy as np

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, spectral_frame
from .prediction_residual import PredictionResidualFilter, ResidualSuppressionConfig


@dataclass(frozen=True)
class PredictionPowerConfig:
    strategy: str = "smoothed_error"
    power_memory: float = .3
    reverberation_time_seconds: float = .6
    decay_interval_seconds: float = .05
    history_frames: int = 5
    error_blend: float = .95
    power_floor: float = 1e-6

    def __post_init__(self):
        if self.strategy not in ("input", "prediction_error", "decay_compensated", "smoothed_error"):
            raise ValueError("unknown prediction power strategy")
        positive_integer(self.history_frames, "history_frames")
        if self.history_frames < 2:
            raise ValueError("history_frames must be at least two")
        for name in ("power_memory", "error_blend"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        for name in ("reverberation_time_seconds", "decay_interval_seconds", "power_floor"):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).tiny)


@dataclass(frozen=True)
class AdaptiveForgettingConfig:
    output_power_memory: float = .98
    update_error_memory: float = .99
    sensitivity: float = .05
    minimum_factor: float = .998
    rapid_change_factor: float = .990
    history_frames: int = 35
    change_threshold: float = 2.0

    def __post_init__(self):
        positive_integer(self.history_frames, "history_frames")
        for name in ("output_power_memory", "update_error_memory"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        for name in ("minimum_factor", "rapid_change_factor"):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).tiny, maximum=1)
        for name in ("sensitivity", "change_threshold"):
            finite_scalar(getattr(self, name), name, minimum=0)


@dataclass(frozen=True)
class WpeConfig:
    frame_length_samples: int = 512
    channel_count: int = 2
    prediction_delay_frames: int = 3
    filter_length_frames: int = 7
    output_delay_frames: int = 10
    covariance_update: str = "full"
    forgetting_factor: float = .99
    wiener_memory: float = .63
    power: PredictionPowerConfig = field(default_factory=PredictionPowerConfig)
    adaptive_forgetting: AdaptiveForgettingConfig | None = field(default_factory=AdaptiveForgettingConfig)
    postfilter: ResidualSuppressionConfig | None = None

    def __post_init__(self):
        for name in ("frame_length_samples", "channel_count", "prediction_delay_frames", "filter_length_frames"):
            positive_integer(getattr(self, name), name)
        nonnegative_integer(self.output_delay_frames, "output_delay_frames")
        if self.frame_length_samples < 4 or self.frame_length_samples % 2:
            raise ValueError("frame_length_samples must be even and at least four")
        if self.covariance_update not in ("full", "diagonal"):
            raise ValueError("covariance_update must be full or diagonal")
        finite_scalar(self.forgetting_factor, "forgetting_factor", minimum=np.finfo(float).tiny, maximum=1)
        finite_scalar(self.wiener_memory, "wiener_memory", minimum=0, maximum=1)
        if not isinstance(self.power, PredictionPowerConfig):
            raise TypeError("power must be a PredictionPowerConfig")
        if self.adaptive_forgetting is not None and not isinstance(self.adaptive_forgetting, AdaptiveForgettingConfig):
            raise TypeError("adaptive_forgetting must be an AdaptiveForgettingConfig or None")
        if self.postfilter is not None and not isinstance(self.postfilter, ResidualSuppressionConfig):
            raise TypeError("postfilter must be a ResidualSuppressionConfig or None")

    @property
    def frequency_bin_count(self):
        return self.frame_length_samples // 2 + 1


class WeightedPredictionDereverberator:
    """Remove predictable late components from (bins, channels) frames.

    Prediction history is ordered oldest to newest, with microphones adjacent
    within each frame. A positive output delay uses the updated coefficients
    to render a past frame; zero delay returns the current pre-update residual.
    """

    def __init__(self, config: WpeConfig | None = None):
        self._config = config if config is not None else WpeConfig()
        self._postfilter = PredictionResidualFilter(self.config.frequency_bin_count, self.config.channel_count,
                                                    self.config.postfilter) if self.config.postfilter else None
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def prediction_coefficients(self):
        return self._coefficients.copy()

    @property
    def forgetting_factor(self):
        return self._forgetting_factor

    @property
    def frame_index(self):
        return self._frame_index

    @property
    def latency_frames(self):
        return self.config.output_delay_frames

    def reset(self):
        config = self.config
        bins, channels = config.frequency_bin_count, config.channel_count
        history_length = config.filter_length_frames + config.prediction_delay_frames
        coefficients = channels * config.filter_length_frames
        self._history = np.zeros((bins, history_length, channels), dtype=np.complex128)
        self._delayed_history = np.zeros((bins, history_length + config.output_delay_frames, channels), dtype=np.complex128)
        self._coefficients = np.zeros((bins, coefficients, channels), dtype=np.complex128)
        covariance = np.diag(np.repeat(.5 ** np.arange(config.filter_length_frames), channels))
        self._inverse_covariance = np.broadcast_to(covariance, (bins, coefficients, coefficients)).astype(complex).copy()
        self._power_history = np.zeros((bins, config.power.history_frames, channels))
        self._clean_power = np.zeros((bins, channels))
        self._psd = np.zeros((bins, 1))
        self._previous_output_power = np.zeros((bins, 1))
        self._wiener_gain = np.zeros((bins, 1))
        self._output = np.zeros((bins, channels), dtype=np.complex128)
        self._forgetting_factor = config.forgetting_factor
        self._update_error = 0.0
        self._smoothed_output_power = 0.0
        self._error_history = np.zeros(config.adaptive_forgetting.history_frames if config.adaptive_forgetting else 0)
        self._frame_index = 0
        self._flushed = False
        if self._postfilter is not None:
            self._postfilter.reset()

    def process_frame(self, spectrum_frame):
        if self._flushed:
            raise RuntimeError("reset is required before processing after flush")
        config = self.config
        spectrum = spectral_frame(spectrum_frame, config.frequency_bin_count, config.channel_count)
        self._history[:, :-1] = self._history[:, 1:].copy()
        self._history[:, -1] = spectrum
        history = self._history[:, :config.filter_length_frames].reshape(config.frequency_bin_count, -1)
        prediction = np.einsum("fkc,fk->fc", self._coefficients.conj(), history)
        residual = spectrum - prediction
        power_config = config.power
        floor = power_config.power_floor
        if power_config.strategy == "input":
            psd = np.mean(np.abs(spectrum)**2, axis=1, keepdims=True) + floor
        elif power_config.strategy == "prediction_error":
            psd = np.mean(np.abs(residual)**2, axis=1, keepdims=True) + floor
        elif power_config.strategy == "decay_compensated":
            self._power_history[:, :-1] = self._power_history[:, 1:].copy()
            self._power_history[:, -1] = (power_config.power_memory * self._power_history[:, -2]
                                         + (1 - power_config.power_memory) * np.abs(spectrum)**2)
            decay = 3 * np.log(10) * power_config.decay_interval_seconds / power_config.reverberation_time_seconds
            late_power = np.exp(-2 * decay) * self._power_history[:, 0]
            instantaneous_clean = np.maximum(np.abs(spectrum)**2 - late_power, 0)
            self._clean_power = power_config.power_memory * self._clean_power + (1 - power_config.power_memory) * instantaneous_clean
            psd = np.mean(self._clean_power, axis=1, keepdims=True) + floor
        else:
            psd = ((1 - power_config.error_blend) * self._previous_output_power
                   + power_config.error_blend * np.mean(np.abs(residual)**1.5, axis=1, keepdims=True) + floor)
        factor = self._forgetting_factor
        numerator = np.einsum("fkl,fl->fk", self._inverse_covariance, history)
        denominator = factor * psd[:, 0] + np.sum(history.conj() * numerator, axis=1)
        kalman_gain = numerator / denominator[:, None]
        coefficient_update = kalman_gain[:, :, None] * residual[:, None, :].conj()
        self._coefficients += coefficient_update
        if config.covariance_update == "full":
            row = np.einsum("fk,fkl->fl", history.conj(), self._inverse_covariance)
            self._inverse_covariance = (self._inverse_covariance - kalman_gain[:, :, None] * row[:, None, :]) / factor
        else:
            diagonal_update = kalman_gain * history.conj()
            self._inverse_covariance = (self._inverse_covariance - diagonal_update[:, :, None] * self._inverse_covariance) / factor
        if config.output_delay_frames:
            self._delayed_history[:, :-1] = self._delayed_history[:, 1:].copy()
            self._delayed_history[:, -1] = spectrum
            delayed_taps = self._delayed_history[:, :config.filter_length_frames].reshape(config.frequency_bin_count, -1)
            target = self._delayed_history[:, config.filter_length_frames + config.prediction_delay_frames - 1]
            prediction = np.einsum("fkc,fk->fc", self._coefficients.conj(), delayed_taps)
            residual = target - prediction
        else:
            target = spectrum
        ratio = np.minimum(np.mean(np.abs(residual)**2, axis=1, keepdims=True)
                           / np.maximum(np.mean(np.abs(target)**2, axis=1, keepdims=True), floor), 1)
        self._wiener_gain = config.wiener_memory * self._wiener_gain + (1 - config.wiener_memory) * ratio
        output = residual * self._wiener_gain
        if self._postfilter is not None:
            output = self._postfilter.process_frame(output, prediction)
        self._previous_output_power = np.mean(np.abs(output)**2, axis=1, keepdims=True)
        if config.adaptive_forgetting is not None:
            adaptation = config.adaptive_forgetting
            update_energy = np.mean(np.abs(coefficient_update)**2, axis=(1, 2))
            coefficient_energy = np.mean(np.abs(self._coefficients)**2, axis=(1, 2))
            error = np.sqrt(psd[:, 0]) * update_energy / np.maximum(coefficient_energy, floor)
            self._update_error = adaptation.update_error_memory * self._update_error + (1 - adaptation.update_error_memory) * np.sum(error)
            self._error_history[:-1] = self._error_history[1:]
            self._error_history[-1] = self._update_error
            if self._update_error / max(np.min(self._error_history), floor) > adaptation.change_threshold:
                self._forgetting_factor = adaptation.rapid_change_factor
            else:
                self._smoothed_output_power = (adaptation.output_power_memory * self._smoothed_output_power
                                               + (1 - adaptation.output_power_memory) * np.sum(self._previous_output_power))
                scale = max(np.sqrt(self._smoothed_output_power), floor)
                self._forgetting_factor = max(1 - adaptation.sensitivity * self._update_error / scale, adaptation.minimum_factor)
        self._psd = psd
        self._output = output.copy()
        self._frame_index += 1
        return output

    def flush(self):
        """Drain delayed output with zero input frames, then close the stream.

        The returned cube has shape (bins, delay_frames, channels). Empty or
        already drained streams return zero frames. reset() starts a new stream.
        """
        shape = (self.config.frequency_bin_count, 0, self.config.channel_count)
        if self._flushed or self._frame_index == 0:
            self._flushed = True
            return np.empty(shape, dtype=np.complex128)
        if not self.latency_frames:
            self._flushed = True
            return np.empty(shape, dtype=np.complex128)
        zeros = np.zeros((self.config.frequency_bin_count, self.config.channel_count), dtype=complex)
        tail = np.stack([self.process_frame(zeros) for _ in range(self.latency_frames)], axis=1)
        self._flushed = True
        return tail
