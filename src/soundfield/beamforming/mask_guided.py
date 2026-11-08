"""Spatial filtering guided by an explicit speech-amplitude mask."""

from dataclasses import dataclass, field

from ..contracts import finite_scalar, spectral_frame, unit_interval_array
from .minimum_variance import MvdrBeamformer, MvdrConfig
import numpy as np


@dataclass(frozen=True)
class MaskGuidedConfig:
    beamformer: MvdrConfig = field(default_factory=MvdrConfig)
    speech_gain_floor: float = 0.0
    noise_gain_floor: float = 0.0

    def __post_init__(self):
        if not isinstance(self.beamformer, MvdrConfig):
            raise TypeError("beamformer must be an MvdrConfig")
        for name in ("speech_gain_floor", "noise_gain_floor"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)


class MaskGuidedBeamformer:
    """Split speech/noise estimates with a (bins, 1) gain and apply MVDR."""

    def __init__(self, config: MaskGuidedConfig | None = None):
        self._config = config if config is not None else MaskGuidedConfig()
        self._beamformer = MvdrBeamformer(self.config.beamformer)

    @property
    def config(self):
        return self._config

    @property
    def beamforming_coefficients(self):
        return self._beamformer.beamforming_coefficients

    def reset(self):
        self._beamformer.reset()

    def process_frame(self, spectrum_frame, speech_gain):
        dimensions = self.config.beamformer
        spectrum = spectral_frame(spectrum_frame, dimensions.frequency_bin_count, dimensions.microphone_count)
        gain = unit_interval_array(speech_gain, "speech_gain", (dimensions.frequency_bin_count, 1))
        speech = np.maximum(self.config.speech_gain_floor, gain) * spectrum
        noise = np.maximum(self.config.noise_gain_floor, 1 - gain) * spectrum
        return self._beamformer.process_frame(spectrum, speech, noise)
