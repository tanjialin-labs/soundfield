"""Selected-channel suppression using correlation with other channels."""

from dataclasses import dataclass

import numpy as np

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, spectral_frame
from ..numerics import exponential_integral_approximation


@dataclass(frozen=True)
class CorrelationSuppressionConfig:
    frequency_bin_count: int = 513
    channel_count: int = 4
    correlation_memory: float = .8
    prior_memory: float = .8
    energy_start_bin: int = 19
    energy_stop_bin: int = 513
    absence_gain: float = .1

    def __post_init__(self):
        for name in ("frequency_bin_count", "channel_count", "energy_stop_bin"):
            positive_integer(getattr(self, name), name)
        nonnegative_integer(self.energy_start_bin, "energy_start_bin")
        if not self.energy_start_bin < self.energy_stop_bin <= self.frequency_bin_count:
            raise ValueError("energy band must fit the frequency bins")
        for name in ("correlation_memory", "prior_memory", "absence_gain"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)


class CorrelationNoiseSuppressor:
    """Maintain independent statistics for each selected channel.

    Every call updates only channel_index, which is zero-based. The interference
    estimate is the complex sum of the remaining channels. Output is (bins, 1).
    """

    def __init__(self, config: CorrelationSuppressionConfig | None = None):
        self._config = config if config is not None else CorrelationSuppressionConfig()
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def gains(self):
        return self._gains.copy()

    def reset(self):
        shape = (self.config.frequency_bin_count, self.config.channel_count)
        self._cross_power = np.zeros(shape, dtype=np.complex128)
        self._interference_power = np.zeros(shape)
        self._prior_history = np.zeros(shape)
        self._selected_energy = np.zeros(shape[1])
        self._total_energy = np.zeros(shape[1])
        self._gains = np.ones(shape)

    def process_frame(self, spectrum_frame, channel_index: int):
        config = self.config
        spectrum = spectral_frame(spectrum_frame, config.frequency_bin_count, config.channel_count)
        nonnegative_integer(channel_index, "channel_index")
        if channel_index >= config.channel_count:
            raise ValueError("channel_index must identify an input channel")
        combined = np.sum(spectrum, axis=1)
        selected = spectrum[:, channel_index]
        interference = combined - selected
        band = slice(config.energy_start_bin, config.energy_stop_bin)
        total_energy = .8 * self._total_energy[channel_index] + .2 * np.sum(np.abs(combined[band])**2)
        selected_energy = .8 * self._selected_energy[channel_index] + .2 * np.sum(np.abs(selected[band])**2)
        memory = config.correlation_memory
        cross = memory * self._cross_power[:, channel_index] + (1 - memory) * selected * interference.conj()
        interference_power = memory * self._interference_power[:, channel_index] + (1 - memory) * np.abs(interference)**2
        leakage = np.minimum(.3, np.abs(cross) / (np.abs(interference_power) + 1e-9))
        residual_power = leakage * np.abs(interference)**2
        posterior = np.abs(selected)**2 / np.maximum(residual_power, 1e-10)
        prior = np.maximum(config.prior_memory * self._prior_history[:, channel_index]
                           + (1 - config.prior_memory) * np.maximum(posterior - 1, 0), 0)
        exponent = posterior * prior / (1 + prior)
        energy_absence = max(0, (total_energy / (selected_energy + 1e-6) - 30) / 30)
        absence = np.minimum(np.maximum((4.6 - posterior) / 3.6, energy_absence), 1)
        probability = np.zeros_like(posterior)
        active = absence < .9
        probability[active] = 1 / (1 + absence[active] / (1 - absence[active]) * (1 + prior[active]) * np.exp(-exponent[active]))
        conditional = np.ones_like(posterior)
        high = exponent > 5
        conditional[high] = prior[high] / (1 + prior[high])
        moderate = (exponent > 0) & (exponent <= 5)
        conditional[moderate] = prior[moderate] / (1 + prior[moderate]) * np.exp(.5 * exponential_integral_approximation(exponent[moderate]))
        gain = conditional**probability * config.absence_gain**(1 - probability)
        self._cross_power[:, channel_index] = cross
        self._interference_power[:, channel_index] = interference_power
        self._prior_history[:, channel_index] = conditional**2 * posterior
        self._selected_energy[channel_index] = selected_energy
        self._total_energy[channel_index] = total_energy
        self._gains[:, channel_index] = gain
        return (gain * selected)[:, None]
