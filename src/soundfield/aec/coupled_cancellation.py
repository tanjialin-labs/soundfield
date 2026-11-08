"""Multichannel echo cancellation with coupled postfiltering and speech decisions."""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

from ..contracts import finite_scalar
from .input_conditioning import EchoInputConditioner, InputConditioningConfig, ReferenceActivityConfig, ReferenceActivityDetector
from .reference_spectrum import EchoFilterConfig, ReferenceSpectrumBuffer
from .partitioned_filter import PartitionedEchoFilter
from .coupled_postfilter import CoupledEchoPostfilter, CoupledPostfilterConfig
from .farend_speech import FarendSpeechConfig, FarendSpeechDetector


@dataclass(frozen=True)
class CoupledEchoConfig:
    input: InputConditioningConfig = field(default_factory=lambda: InputConditioningConfig(microphone_count=3, reference_count=6))
    linear_filter: EchoFilterConfig = field(default_factory=lambda: EchoFilterConfig(reference_count=6, residual_suppression=0))
    postfilter: CoupledPostfilterConfig = field(default_factory=CoupledPostfilterConfig)
    speech_detection: FarendSpeechConfig = field(default_factory=FarendSpeechConfig)
    reference_activity: ReferenceActivityConfig = field(default_factory=lambda: ReferenceActivityConfig(amplitude_threshold=15/32768))
    output_gain: float = 1

    def __post_init__(self):
        if (self.linear_filter.method != 'constrained'
                or self.input.reference_count != self.linear_filter.reference_count
                or any(config.hop_length_samples != self.input.hop_length_samples
                       for config in (self.linear_filter, self.postfilter, self.speech_detection))):
            raise ValueError('coupled processing requires matching blocks and constrained reference filters')
        finite_scalar(self.output_gain, 'output_gain', minimum=0)


@dataclass(frozen=True)
class CoupledEchoResult:
    audio: np.ndarray
    linear_audio: np.ndarray
    residual_spectrum: np.ndarray
    echo_power: np.ndarray
    speech_decisions: tuple


class CoupledEchoCanceller:
    def __init__(self, config=CoupledEchoConfig()):
        self.config = config
        self.conditioner = EchoInputConditioner(config.input)
        self.activity = ReferenceActivityDetector(config.reference_activity)
        self.reference = ReferenceSpectrumBuffer(config.linear_filter)
        count = config.input.microphone_count
        self.filters = [PartitionedEchoFilter(config.linear_filter) for _ in range(count)]
        self.postfilters = [CoupledEchoPostfilter(config.postfilter) for _ in range(count)]
        self.detectors = [FarendSpeechDetector(config.speech_detection) for _ in range(count)]
        self.reset()

    @property
    def latency_samples(self):
        return self.config.input.hop_length_samples

    def reset(self):
        for processor in (self.conditioner, self.activity, self.reference, *self.filters, *self.postfilters, *self.detectors):
            processor.reset()
        self.linear_memory = np.zeros((1, self.config.input.microphone_count))
        self.output_memory = np.zeros_like(self.linear_memory)

    def _finish(self, audio, name):
        if self.config.input.enable_preemphasis and len(audio):
            audio, memory = lfilter([1], [1, -self.config.input.preemphasis], audio,
                                    axis=0, zi=getattr(self, name))
            setattr(self, name, memory)
        return audio*self.config.output_gain

    def process_block(self, microphone_audio, reference_audio):
        microphone, reference = self.conditioner.process_block(microphone_audio, reference_audio)
        self.reference.process_block(reference, self.activity.process_block(reference), self.filters[0].residual_power)
        hop = self.latency_samples
        linear, output, spectra, powers, decisions = [], [], [], [], []
        for channel, (processor, postfilter, detector) in enumerate(zip(self.filters, self.postfilters, self.detectors)):
            processor.process_block(microphone[:, channel], self.reference, is_clipped=self.conditioner.is_clipped)
            residual = processor.residual_history[hop:, None]
            echo = processor.echo_history[hop:, None]
            linear.append(residual)
            output.append(postfilter.process_block(residual, echo))
            spectra.append(postfilter.output_spectrum)
            powers.append(postfilter.estimated_echo_power[:, None])
            decisions.append(detector.process_block(microphone[:, channel, None], postfilter.output_spectrum,
                                                    postfilter.estimated_echo_power))
        return CoupledEchoResult(self._finish(np.concatenate(output, axis=1), 'output_memory'),
                                 self._finish(np.concatenate(linear, axis=1), 'linear_memory'),
                                 np.concatenate(spectra, axis=1), np.concatenate(powers, axis=1), tuple(decisions))

    def flush(self):
        return self._finish(np.concatenate([processor.flush() for processor in self.postfilters], axis=1), 'output_memory')
