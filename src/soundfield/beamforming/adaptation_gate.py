"""Spatial evidence controlling adaptive interference cancellation."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..contracts import spectral_frame, unit_interval_array
from .fixed_beams import FixedBeamBank


@dataclass(frozen=True)
class AdaptationGateConfig:
    strategy: str = "spatial_amplitude"
    exclude_adjacent_beams: bool = False
    comparison_beam_count: int | None = None

    def __post_init__(self):
        if self.strategy not in ("continuous", "spatial_power", "spatial_amplitude", "diffuse_amplitude", "speech_mask"):
            raise ValueError("unknown adaptation strategy")
        if not isinstance(self.exclude_adjacent_beams, bool):
            raise ValueError("exclude_adjacent_beams must be boolean")
        if self.comparison_beam_count is not None and (isinstance(self.comparison_beam_count, bool)
                or not isinstance(self.comparison_beam_count, Integral) or self.comparison_beam_count < 1):
            raise ValueError("comparison_beam_count must be a positive integer")


class AdaptationGate:
    """Estimate update strength (bins, 1) for one zero-based beam index.

    A supplied speech mask has shape (bins, frames). Each successful call
    consumes one mask frame. Exhaustion raises IndexError without advancing.
    """

    def __init__(self, bank: FixedBeamBank, beam_index: int, config: AdaptationGateConfig | None = None,
                 *, speech_mask=None):
        if isinstance(beam_index, bool) or not isinstance(beam_index, Integral) or not 0 <= beam_index < bank.beam_count:
            raise ValueError("beam_index must identify a beam in the bank")
        self._bank = bank
        self._beam_index = int(beam_index)
        self._config = config if config is not None else AdaptationGateConfig()
        self._competitors = list(range(beam_index + 1, bank.beam_count)) + list(range(beam_index))
        if self.config.strategy == "spatial_amplitude" and self.config.comparison_beam_count is not None:
            if not beam_index < self.config.comparison_beam_count <= bank.beam_count:
                raise ValueError("comparison beam range must contain the target beam")
            self._competitors = [index for index in self._competitors if index < self.config.comparison_beam_count]
        if self.config.strategy == "spatial_amplitude" and self.config.exclude_adjacent_beams:
            self._competitors = [index for index in self._competitors if abs(index - beam_index) > 1]
        if self.config.strategy in ("spatial_power", "spatial_amplitude") and not self._competitors:
            raise ValueError("spatial gating requires at least one competing beam")
        self._speech_mask = None
        if self.config.strategy == "speech_mask":
            if speech_mask is None or np.ndim(speech_mask) != 2:
                raise ValueError("speech_mask must have shape (bins, frames)")
            self._speech_mask = unit_interval_array(speech_mask, "speech_mask", (bank.frequency_bin_count, np.shape(speech_mask)[1])).copy()
            self._speech_mask.flags.writeable = False
        elif speech_mask is not None:
            raise ValueError("speech_mask requires the speech_mask strategy")
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def frame_index(self):
        return self._frame_index

    def reset(self):
        self._frame_index = 0
        self._target_power = np.zeros((self._bank.frequency_bin_count, 1))
        self._competing_power = np.zeros((self._bank.frequency_bin_count, len(self._competitors)))

    def process_frame(self, spectrum_frame):
        frame = spectral_frame(spectrum_frame, self._bank.frequency_bin_count, self._bank.microphone_count)
        mode = self.config.strategy
        if mode == "speech_mask":
            if self._frame_index >= self._speech_mask.shape[1]:
                raise IndexError("speech_mask has no remaining frames")
            strength = 1 - self._speech_mask[:, self._frame_index:self._frame_index + 1]
        elif mode == "continuous":
            strength = np.ones((self._bank.frequency_bin_count, 1))
        else:
            beams, _ = self._bank.project(frame)
            target = beams[:, self._beam_index:self._beam_index + 1]
            if mode == "spatial_power":
                powers = np.abs(beams[:, [self._beam_index] + self._competitors])**2
                smoothed = np.apply_along_axis(lambda column: np.convolve(column, [.25, .5, .25], "full")[1:-1], 0, powers)
                memory = 0.9 if self._frame_index else 0.0
                self._target_power = memory * self._target_power + (1 - memory) * smoothed[:, :1]
                self._competing_power = memory * self._competing_power + (1 - memory) * smoothed[:, 1:]
                denominator = np.max(self._competing_power, axis=1, keepdims=True)
                ratio = np.divide(self._target_power, denominator, out=np.full_like(denominator, np.inf), where=denominator > 0)
                strength = np.maximum((1.5 - ratio) / 1.2, 0)
                strength[strength <= .5] = 0
                strength[strength >= .75] = 1
            elif mode == "spatial_amplitude":
                competing = np.max(np.abs(beams[:, self._competitors]), axis=1, keepdims=True)
                strength = np.maximum(1 - np.abs(target) / (competing + 1e-10), .01)
                strength[strength > .5] = 1
            else:
                mean_amplitude = np.maximum(np.mean(np.abs(frame), axis=1, keepdims=True), 1e-10)
                strength = 1 - np.minimum(1, np.abs(target) / mean_amplitude)
                strength[strength < .2] = .01
        self._frame_index += 1
        return strength.copy()
