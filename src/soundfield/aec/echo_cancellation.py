"""Multichannel acoustic echo cancellation with explicit reference inputs."""

from dataclasses import dataclass, field

import numpy as np

from ..contracts import audio_array, unit_interval_array, finite_scalar
from .input_conditioning import InputConditioningConfig, EchoInputConditioner, ReferenceActivityConfig, ReferenceActivityDetector
from .reference_spectrum import EchoFilterConfig, ReferenceSpectrumBuffer
from .partitioned_filter import PartitionedEchoFilter
from .echo_refinement import EchoRefinementConfig, CovarianceEchoRefiner
from .residual_postfilter import ResidualEchoConfig, ResidualEchoPostfilter
from .communication_postfilter import CommunicationPostfilterConfig, CommunicationEchoPostfilter
from .audio_output import EchoAudioSynthesizer
from .delay_estimation import estimate_filter_delay


@dataclass(frozen=True)
class EchoCancellationConfig:
    input: InputConditioningConfig = field(default_factory=InputConditioningConfig)
    linear_filter: EchoFilterConfig = field(default_factory=EchoFilterConfig)
    reference_activity: ReferenceActivityConfig | None = field(default_factory=ReferenceActivityConfig)
    residual_filter: ResidualEchoConfig | None = field(default_factory=ResidualEchoConfig)
    communication_filter: CommunicationPostfilterConfig | None = field(default_factory=CommunicationPostfilterConfig)
    refinement: EchoRefinementConfig | None = None
    output_gain: float = 1.0
    enable_delay_estimation: bool = False

    def __post_init__(self):
        if (self.input.hop_length_samples != self.linear_filter.hop_length_samples
                or self.input.reference_count != self.linear_filter.reference_count):
            raise ValueError('input and linear filter configurations must agree')
        bins = self.linear_filter.frequency_bin_count
        if self.residual_filter is not None and self.residual_filter.frequency_bin_count != bins:
            raise ValueError('residual filter bins must match the linear filter')
        if self.communication_filter is not None and self.communication_filter.noise.frequency_bin_count != bins:
            raise ValueError('communication filter bins must match the linear filter')
        if self.refinement is not None:
            count = self.linear_filter.refinement_partition_count*len(self.linear_filter.refinement_reference_indices)
            if self.refinement.frequency_bin_count != bins or self.refinement.reference_component_count != count:
                raise ValueError('refinement dimensions must match reference preparation')
        finite_scalar(self.output_gain, 'output_gain', minimum=0)


@dataclass(frozen=True)
class EchoAudioResult:
    audio: np.ndarray
    communication_audio: np.ndarray
    residual_spectrum: np.ndarray
    echo_spectrum: np.ndarray


class AcousticEchoCanceller:
    """One stream per instance; audio blocks use normalized amplitude units."""

    def __init__(self, config=EchoCancellationConfig()):
        self.config = config
        self.input_conditioner = EchoInputConditioner(config.input)
        self.reference_detector = ReferenceActivityDetector(config.reference_activity) if config.reference_activity else None
        self.reference_buffer = ReferenceSpectrumBuffer(config.linear_filter)
        count = config.input.microphone_count
        self.filters = [PartitionedEchoFilter(config.linear_filter) for _ in range(count)]
        self.refiners = [CovarianceEchoRefiner(config.refinement) for _ in range(count)] if config.refinement else []
        self.residual_filters = [ResidualEchoPostfilter(config.residual_filter) for _ in range(count)] if config.residual_filter else []
        self.communication_filters = [CommunicationEchoPostfilter(config.communication_filter) for _ in range(count)] if config.communication_filter else []
        output_arguments = dict(hop_length_samples=config.input.hop_length_samples, channel_count=count,
                                preemphasis=config.input.preemphasis, enable_deemphasis=config.input.enable_preemphasis,
                                output_gain=config.output_gain)
        self.audio_synthesizer = EchoAudioSynthesizer(**output_arguments)
        self.communication_synthesizer = EchoAudioSynthesizer(**output_arguments)
        self.estimated_delay_samples = 0

    @property
    def latency_samples(self):
        return self.config.input.hop_length_samples

    def reset(self):
        for processor in (self.input_conditioner, self.reference_buffer, self.audio_synthesizer,
                          self.communication_synthesizer, *self.filters, *self.refiners,
                          *self.residual_filters, *self.communication_filters):
            processor.reset()
        if self.reference_detector:
            self.reference_detector.reset()
        self.estimated_delay_samples = 0

    def process_block(self, microphone_audio, reference_audio, *, speech_mask=None, communication_mask=None):
        bins, channels = self.config.linear_filter.frequency_bin_count, self.config.input.microphone_count
        if speech_mask is not None:
            speech_mask = unit_interval_array(speech_mask, 'speech_mask', (bins, channels))
        if communication_mask is not None:
            communication_mask = unit_interval_array(communication_mask, 'communication_mask', (bins, channels))
        if self.config.residual_filter and self.config.residual_filter.method == 'mask' and speech_mask is None:
            raise ValueError('mask-based residual filtering requires speech_mask')
        microphone, reference = self.input_conditioner.process_block(microphone_audio, reference_audio)
        active = self.reference_detector.process_block(reference) if self.reference_detector else True
        silent_frames = self.reference_detector.silent_frames if self.reference_detector else 0
        self.reference_buffer.process_block(reference, active, self.filters[0].residual_power)
        if self.config.enable_delay_estimation:
            self.estimated_delay_samples = (self.filters[0].estimated_delay_samples if self.config.linear_filter.method == 'constrained'
                else estimate_filter_delay(self.filters[0].adaptive_weights, self.latency_samples, self.config.input.reference_count))
        residual, estimated = np.zeros((bins, channels), complex), np.zeros((bins, channels), complex)
        primary, communication = residual.copy(), residual.copy()
        for channel, processor in enumerate(self.filters):
            result = processor.process_block(microphone[:, channel], self.reference_buffer, is_clipped=self.input_conditioner.is_clipped)
            error, echo = result.residual_spectrum, result.echo_spectrum
            if self.refiners:
                refined = self.refiners[channel].process_frame(error, echo, self.reference_buffer.refinement_history,
                                                              is_active=self.reference_buffer.is_active, silent_frames=silent_frames)
                error, echo = refined.residual_spectrum, echo+refined.echo_spectrum
            residual[:, channel], estimated[:, channel] = error[:, 0], echo[:, 0]
            filtered = error
            if self.residual_filters and active:
                filtered = self.residual_filters[channel].process_frame(error, echo,
                    speech_mask=None if speech_mask is None else speech_mask[:, channel])
            primary[:, channel] = filtered[:, 0]
            if self.communication_filters:
                communication[:, channel] = self.communication_filters[channel].process_frame(error, echo,
                    self.reference_buffer.spectrum, speech_mask=None if communication_mask is None else communication_mask[:, channel])[:, 0]
        output = self.audio_synthesizer.process_frame(primary)
        communication_output = (self.communication_synthesizer.process_frame(communication) if self.communication_filters else np.zeros_like(output))
        return EchoAudioResult(output, communication_output, residual, estimated)

    def flush(self):
        """Emit the final synthesis hop; no new adaptive-filter update is performed."""
        primary = self.audio_synthesizer.flush()
        communication = self.communication_synthesizer.flush() if self.communication_filters else np.zeros_like(primary)
        return primary, communication


def cancel_echo(microphone_audio, reference_audio, config=None):
    """Process an entire recording and compensate the analysis hop delay."""
    microphone, reference = audio_array(microphone_audio), audio_array(reference_audio)
    if len(microphone) != len(reference):
        raise ValueError('microphone and reference recordings must have equal sample counts')
    config = config or EchoCancellationConfig()
    if microphone.shape[1] != config.input.microphone_count or reference.shape[1] != config.input.reference_count:
        raise ValueError('recording channels must agree with the configuration')
    if config.residual_filter and config.residual_filter.method == 'mask':
        raise ValueError('external masks require the block interface')
    if not len(microphone):
        return microphone.copy(), microphone.copy()
    processor = AcousticEchoCanceller(config)
    hop = processor.latency_samples
    padded = ((len(microphone)+hop-1)//hop)*hop
    microphone = np.pad(microphone, ((0, padded-len(microphone)+hop), (0, 0)))
    reference = np.pad(reference, ((0, padded-len(reference)+hop), (0, 0)))
    outputs, communication = [], []
    for start in range(0, len(microphone), hop):
        result = processor.process_block(microphone[start:start+hop], reference[start:start+hop])
        outputs.append(result.audio)
        communication.append(result.communication_audio)
    length = len(microphone_audio)
    return np.concatenate(outputs)[hop:hop+length], np.concatenate(communication)[hop:hop+length]
