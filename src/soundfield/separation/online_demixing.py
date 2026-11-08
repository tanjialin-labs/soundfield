"""Online multichannel demixing and source-dependent spectral masks."""

from dataclasses import dataclass

import numpy as np

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, real_array, spectral_frame
from ..enhancement import SeparationGuidedPostfilter, SeparationPostfilterConfig


@dataclass(frozen=True)
class NearfieldControlConfig:
    suppressed_channel_index: int = 2
    gain_sum_threshold: float = 10.0
    initial_gain_sum: float = 0.0

    def __post_init__(self):
        nonnegative_integer(self.suppressed_channel_index, "suppressed_channel_index")
        finite_scalar(self.gain_sum_threshold, "gain_sum_threshold", minimum=0)
        finite_scalar(self.initial_gain_sum, "initial_gain_sum", minimum=0)


@dataclass(frozen=True)
class DemixingConfig:
    frequency_bin_count: int = 2049
    channel_count: int = 2
    iterations_per_frame: int = 1
    source_distribution: str = "gaussian"
    covariance_memory: float = .98
    energy_blend: float = .5
    weighting_strategy: str = "source_energy"
    energy_start_bin: int = 0
    energy_stop_bin: int = 257
    output_mode: str = "reference_mask"
    reference_channel_index: int = 1
    mask_floor: float | None = None
    nearfield_control: NearfieldControlConfig | None = None
    postfilter: SeparationPostfilterConfig | None = None

    def __post_init__(self):
        for name in ("frequency_bin_count", "channel_count", "iterations_per_frame", "energy_stop_bin"):
            positive_integer(getattr(self, name), name)
        nonnegative_integer(self.energy_start_bin, "energy_start_bin")
        nonnegative_integer(self.reference_channel_index, "reference_channel_index")
        if self.channel_count < 2 or self.reference_channel_index >= self.channel_count:
            raise ValueError("at least two channels and a valid reference channel are required")
        if not self.energy_start_bin < self.energy_stop_bin <= self.frequency_bin_count:
            raise ValueError("energy band must fit the frequency bins")
        if self.source_distribution not in ("gaussian", "laplace"):
            raise ValueError("unknown source distribution")
        if self.weighting_strategy not in ("spectral_energy", "source_energy", "blended_energy"):
            raise ValueError("unknown covariance weighting strategy")
        if self.output_mode not in ("channel_mask", "reference_mask", "mixture_mask", "postfilter"):
            raise ValueError("unknown output mode")
        for name in ("covariance_memory", "energy_blend"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        if self.covariance_memory == 1:
            raise ValueError("covariance_memory must be less than one")
        if self.mask_floor is not None:
            finite_scalar(self.mask_floor, "mask_floor", minimum=0, maximum=1)
        if self.nearfield_control is not None:
            if not isinstance(self.nearfield_control, NearfieldControlConfig):
                raise TypeError("nearfield_control must be a NearfieldControlConfig")
            if self.nearfield_control.suppressed_channel_index >= self.channel_count:
                raise ValueError("suppressed channel must fit the input channels")
        if self.output_mode == "postfilter":
            if not isinstance(self.postfilter, SeparationPostfilterConfig):
                raise ValueError("postfilter output requires a SeparationPostfilterConfig")
            if (self.postfilter.frequency_bin_count, self.postfilter.channel_count) != (self.frequency_bin_count, self.channel_count):
                raise ValueError("postfilter dimensions must agree with the demixer")
        elif self.postfilter is not None:
            raise ValueError("postfilter configuration requires postfilter output mode")


class OnlineSourceSeparator:
    """Estimate source masks and render multichannel audio spectra.

    Demixing coefficients have shape (bins, microphones, sources). Weighted
    covariance has shape (bins, microphones, microphones, sources). Input and
    output frames retain shape (bins, channels) and source order.
    """

    def __init__(self, config: DemixingConfig | None = None, *, channel_energy_weights=None):
        self._config = config if config is not None else DemixingConfig()
        count = self.config.channel_count
        if channel_energy_weights is None:
            channel_energy_weights = np.ones(count)
        weights = real_array(channel_energy_weights, "channel_energy_weights", 1)
        if weights.shape != (count,) or np.any(weights < 0):
            raise ValueError("channel_energy_weights must be nonnegative and match channels")
        self._channel_energy_weights = weights.copy()
        self._postfilter = SeparationGuidedPostfilter(self.config.postfilter) if self.config.postfilter else None
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def demixing_coefficients(self):
        return self._coefficients.copy()

    @property
    def separation_gain(self):
        return self._gain.copy()

    def reset(self):
        config = self.config
        identity = np.eye(config.channel_count, dtype=np.complex128)
        self._coefficients = np.broadcast_to(identity, (config.frequency_bin_count, *identity.shape)).copy()
        self._covariance = np.broadcast_to(.001 * identity[None, :, :, None],
            (config.frequency_bin_count, config.channel_count, config.channel_count, config.channel_count)).copy()
        self._gain = np.zeros((config.frequency_bin_count, config.channel_count))
        self._previous_gain_sum = self.config.nearfield_control.initial_gain_sum if self.config.nearfield_control else 0.0
        if self._postfilter is not None:
            self._postfilter.reset()

    def process_frame(self, spectrum_frame, *, nearfield_power=None, nearfield_factors=None):
        config = self.config
        spectrum = spectral_frame(spectrum_frame, config.frequency_bin_count, config.channel_count)
        if config.nearfield_control is not None:
            if nearfield_power is None or nearfield_factors is None:
                raise ValueError("nearfield control requires explicit power and factors for every frame")
            nearfield_power = real_array(nearfield_power, "nearfield_power", 1)
            nearfield_factors = real_array(nearfield_factors, "nearfield_factors", 1)
            if any(value.shape != (config.channel_count,) or np.any(value < 0) for value in (nearfield_power, nearfield_factors)):
                raise ValueError("nearfield arrays must be nonnegative and match channels")
        elif nearfield_power is not None or nearfield_factors is not None:
            raise ValueError("nearfield evidence requires nearfield_control")
        identity = np.eye(config.channel_count)
        input_covariance = spectrum[:, :, None] * spectrum[:, None, :].conj()
        coefficients = self._coefficients.copy()
        for _ in range(config.iterations_per_frame):
            estimates = np.einsum("fms,fm->fs", coefficients.conj(), spectrum)
            if config.source_distribution == "gaussian":
                source_energy = np.maximum(np.mean(np.abs(estimates)**2, axis=0), 1e-4)
            else:
                source_energy = 2 * np.maximum(np.sqrt(np.sum(np.abs(estimates)**2, axis=0)), 1e-4)
            if config.weighting_strategy == "spectral_energy":
                spectral_power = np.abs(spectrum)**2
                spectral_power[:config.energy_start_bin + 1] = 0
                spectral_power[config.energy_stop_bin - 1:] = 0
                inverse_energy = 1 / (spectral_power + source_energy)
            elif config.weighting_strategy == "source_energy":
                inverse_energy = np.broadcast_to(1 / source_energy, spectrum.shape)
            else:
                if config.nearfield_control is None:
                    channel_energy = np.mean(np.abs(spectrum[config.energy_start_bin:config.energy_stop_bin])**2, axis=0) * self._channel_energy_weights
                    evidence = channel_energy / np.mean(channel_energy + 1e-10)
                else:
                    evidence = nearfield_power * nearfield_factors / np.mean(nearfield_power + 1e-10)
                blended = config.energy_blend * source_energy + (1 - config.energy_blend) * evidence * np.mean(source_energy)
                inverse_energy = np.broadcast_to(np.divide(1, blended, out=np.zeros_like(blended), where=blended > 0), spectrum.shape)
            covariance = (config.covariance_memory * self._covariance
                          + (1 - config.covariance_memory) * (input_covariance[:, :, :, None] * inverse_energy[:, None, None, :]
                                                             + 1e-4 * identity[None, :, :, None]))
            previous_adjoint = coefficients.conj().swapaxes(1, 2)
            updated = np.empty_like(coefficients)
            for source in range(config.channel_count):
                weighted = covariance[:, :, :, source]
                system = previous_adjoint @ weighted
                rhs = np.broadcast_to(identity[:, source][None, :, None], (config.frequency_bin_count, config.channel_count, 1))
                direction = np.linalg.solve(system, rhs)[..., 0]
                normalizer = np.sum(direction.conj() * np.einsum("fmn,fn->fm", weighted, direction), axis=1)
                updated[:, :, source] = direction / np.sqrt(normalizer)[:, None]
            coefficients = updated
        estimates = np.einsum("fms,fm->fs", coefficients.conj(), spectrum)
        power = np.abs(estimates)**2
        total_power = np.sum(power, axis=1, keepdims=True)
        gain = np.divide(power, total_power, out=np.zeros_like(power), where=total_power > 0)
        if config.nearfield_control is not None:
            control = config.nearfield_control
            gain_sum = float(np.sum(gain[:, control.suppressed_channel_index]))
            if gain_sum < control.gain_sum_threshold and self._previous_gain_sum < control.gain_sum_threshold:
                gain[:, control.suppressed_channel_index] = 0
            self._previous_gain_sum = gain_sum
        if config.mask_floor is not None:
            gain = np.maximum(gain, config.mask_floor)
        if config.output_mode == "channel_mask":
            output = spectrum * gain
        elif config.output_mode == "reference_mask":
            output = spectrum[:, config.reference_channel_index:config.reference_channel_index + 1] * gain
        elif config.output_mode == "mixture_mask":
            output = np.sum(spectrum, axis=1, keepdims=True) * gain
        else:
            output = self._postfilter.process_frame(spectrum, gain)
        self._coefficients = coefficients
        self._covariance = covariance
        self._gain = gain
        return output
