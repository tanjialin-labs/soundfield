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
    channel_model: str = "interference_reference"

    def __post_init__(self):
        if self.channel_model not in ("interference_reference", "coherent_speech"):
            raise ValueError("unknown channel_model")
        for name in ("frequency_bin_count", "channel_count", "energy_stop_bin"):
            positive_integer(getattr(self, name), name)
        nonnegative_integer(self.energy_start_bin, "energy_start_bin")
        if not self.energy_start_bin < self.energy_stop_bin <= self.frequency_bin_count:
            raise ValueError("energy band must fit the frequency bins")
        if self.channel_model == "coherent_speech" and not 0 < self.correlation_memory < 1:
            raise ValueError("coherent speech requires correlation_memory between zero and one, exclusively")
        for name in ("correlation_memory", "prior_memory", "absence_gain"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)


class CorrelationNoiseSuppressor:
    """Maintain independent statistics for each selected channel.

    Every call updates only channel_index, which is zero-based. The interference
    estimate in reference mode is the complex sum of the remaining channels.
    Coherent-speech mode retains components shared across microphones. Output is (bins, 1).
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
        self._gains = np.ones(shape)
        if self.config.channel_model == "coherent_speech":
            pair_shape = (*shape, self.config.channel_count)
            self._pair_cross = np.zeros(pair_shape, complex)
            self._pair_power = np.zeros(pair_shape)
        else:
            self._cross_power = np.zeros(shape, dtype=np.complex128)
            self._interference_power = np.zeros(shape)
            self._prior_history = np.zeros(shape)
            self._selected_energy = np.zeros(shape[1])
            self._total_energy = np.zeros(shape[1])

    def process_frame(self, spectrum_frame, channel_index: int):
        config = self.config
        spectrum = spectral_frame(spectrum_frame, config.frequency_bin_count, config.channel_count)
        nonnegative_integer(channel_index, "channel_index")
        if channel_index >= config.channel_count:
            raise ValueError("channel_index must identify an input channel")
        if config.channel_model == "coherent_speech":
            return self._coherent_output(spectrum, channel_index)
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

    def _coherent_output(self, spectrum, channel):
        if self.config.channel_count == 1:
            return spectrum.copy()
        memory = self.config.correlation_memory
        cross = self._pair_cross[:, channel]
        power = self._pair_power[:, channel]
        cross *= memory
        cross += (1 - memory) * spectrum[:, channel:channel+1] * spectrum.conj()
        power *= memory
        power += (1 - memory) * np.abs(spectrum)**2
        denominator = power[:, channel:channel+1] * power
        coherence = np.divide(np.abs(cross)**2, denominator, out=np.zeros_like(power), where=denominator > 0)
        coherence[:, channel] = 0
        observed = denominator > 0
        observed[:, channel] = False
        count = observed.sum(axis=1)
        shared = np.divide(coherence.sum(axis=1), count, out=np.ones(len(count)), where=count > 0)
        bias = (1 - memory) / (1 + memory)
        corrected = np.clip((shared - bias) / (2 * memory / (1 + memory)), 0, 1)
        gain = self.config.absence_gain + (1 - self.config.absence_gain) * np.sqrt(corrected)
        self._gains[:, channel] = gain
        return spectrum[:, channel:channel+1] * gain[:, None]
