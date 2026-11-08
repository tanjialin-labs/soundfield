"""Residual reverberation suppression from prediction/error correlation."""

from dataclasses import dataclass

import numpy as np

from ..contracts import finite_scalar, positive_integer, spectral_frame
from ..numerics import spectral_amplitude_gain


@dataclass(frozen=True)
class ResidualSuppressionConfig:
    correlation_memory: float = .984
    leakage_scale: float = 1.7
    posterior_weight: float = .02
    prior_snr_floor: float = .0032
    noise_power_floor: float = 1e-7

    def __post_init__(self):
        for name in ("correlation_memory", "posterior_weight"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        finite_scalar(self.leakage_scale, "leakage_scale", minimum=0)
        for name in ("prior_snr_floor", "noise_power_floor"):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).tiny)


class PredictionResidualFilter:
    """Suppress residuals using their correlation with the predicted component."""

    def __init__(self, frequency_bin_count: int, channel_count: int, config: ResidualSuppressionConfig | None = None):
        positive_integer(frequency_bin_count, "frequency_bin_count")
        positive_integer(channel_count, "channel_count")
        self._shape = (frequency_bin_count, channel_count)
        self._config = config if config is not None else ResidualSuppressionConfig()
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def gain(self):
        return self._gain.copy()

    def reset(self):
        self._initialized = False
        self._cross_power = np.zeros(self._shape, dtype=np.complex128)
        self._prediction_power = np.zeros(self._shape)
        self._residual_power = np.zeros(self._shape)
        self._previous_output_power = np.zeros(self._shape)
        self._gain = np.ones(self._shape)

    def process_frame(self, residual_spectrum, prediction_spectrum):
        residual, prediction = [spectral_frame(value, *self._shape) for value in (residual_spectrum, prediction_spectrum)]
        config = self.config
        memory = config.correlation_memory if self._initialized else 0.0
        prediction_power, residual_power = np.abs(prediction)**2, np.abs(residual)**2
        self._cross_power = memory * self._cross_power + (1 - memory) * prediction * residual.conj()
        self._prediction_power = memory * self._prediction_power + (1 - memory) * prediction_power
        self._residual_power = memory * self._residual_power + (1 - memory) * residual_power
        with np.errstate(divide="ignore", invalid="ignore"):
            power_ratio = np.fmin(1, np.abs(self._residual_power / self._prediction_power))
            coherence = np.abs(self._cross_power / np.sqrt(self._residual_power * self._prediction_power))
            leakage = np.fmin(1, config.leakage_scale * power_ratio * coherence)
        noise = prediction_power * leakage + config.noise_power_floor
        posterior = np.minimum(residual_power / noise, 40)
        prior = np.maximum(config.posterior_weight * np.maximum(posterior - 1, 0)
                           + (1 - config.posterior_weight) * self._previous_output_power / noise, config.prior_snr_floor)
        ratio = prior / (1 + prior)
        self._gain = np.minimum(1, ratio * spectral_amplitude_gain(ratio * posterior))
        output = residual * self._gain
        self._previous_output_power = np.abs(output)**2
        self._initialized = True
        return output
