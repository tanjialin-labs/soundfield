"""Fixed beams, spatial update control, and adaptive sidelobe cancellation."""

from dataclasses import dataclass, field

import numpy as np

from ..adaptive import NlmsConfig, NormalizedLmsCanceller
from ..contracts import finite_scalar, spectral_frame
from ..enhancement.reference_postfilter import NoiseReferencePostfilter, ReferencePostfilterConfig
from .adaptation_gate import AdaptationGate, AdaptationGateConfig
from .fixed_beams import FixedBeamBank


@dataclass(frozen=True)
class SidelobeConfig:
    adaptive_filter: NlmsConfig = field(default_factory=NlmsConfig)
    adaptation_gate: AdaptationGateConfig = field(default_factory=AdaptationGateConfig)
    enable_gate_attenuation: bool = True
    attenuation_offset: float = .2
    attenuation_floor: float = 0.0
    postfilter: ReferencePostfilterConfig | None = None

    def __post_init__(self):
        if not isinstance(self.adaptive_filter, NlmsConfig) or not isinstance(self.adaptation_gate, AdaptationGateConfig):
            raise TypeError("adaptive_filter and adaptation_gate must be configuration objects")
        if not isinstance(self.enable_gate_attenuation, bool):
            raise ValueError("enable_gate_attenuation must be boolean")
        for name in ("attenuation_offset", "attenuation_floor"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        if self.postfilter is not None and not isinstance(self.postfilter, ReferencePostfilterConfig):
            raise TypeError("postfilter must be a ReferencePostfilterConfig or None")


class GeneralizedSidelobeCanceller:
    """Enhance one beam with independent state for its adaptive components."""

    def __init__(self, bank: FixedBeamBank, beam_index: int, config: SidelobeConfig | None = None, *, speech_mask=None):
        self._config = config if config is not None else SidelobeConfig()
        adaptive = self.config.adaptive_filter
        if adaptive.frequency_bin_count != bank.frequency_bin_count or adaptive.reference_count != bank.reference_count:
            raise ValueError("adaptive filter dimensions must agree with the beam bank")
        if self.config.postfilter is not None and self.config.postfilter.frequency_bin_count != bank.frequency_bin_count:
            raise ValueError("postfilter dimensions must agree with the beam bank")
        self._bank = bank
        self._gate = AdaptationGate(bank, beam_index, self.config.adaptation_gate, speech_mask=speech_mask)
        self._beam_index = beam_index
        self._canceller = NormalizedLmsCanceller(adaptive)
        self._postfilter = NoiseReferencePostfilter(self.config.postfilter) if self.config.postfilter is not None else None
        self._active_references = bank.active_references[:, :, beam_index]

    @property
    def config(self):
        return self._config

    def reset(self):
        self._gate.reset()
        self._canceller.reset()
        if self._postfilter is not None:
            self._postfilter.reset()

    def process_frame(self, spectrum_frame):
        frame = spectral_frame(spectrum_frame, self._bank.frequency_bin_count, self._bank.microphone_count)
        beams, references = self._bank.project(frame)
        strength = self._gate.process_frame(frame)
        reference = references[:, :, self._beam_index]
        output = self._canceller.process_frame(beams[:, self._beam_index:self._beam_index + 1], reference, strength)
        if self.config.enable_gate_attenuation:
            attenuation = np.maximum(self.config.attenuation_floor, np.minimum(1 - strength + self.config.attenuation_offset, 1))
            output *= attenuation
        if self._postfilter is not None:
            mean_reference = np.sum(reference * self._active_references, axis=1, keepdims=True) / (np.sum(self._active_references, axis=1, keepdims=True) + 1e-10)
            output = self._postfilter.process_frame(output, mean_reference)
        return output
