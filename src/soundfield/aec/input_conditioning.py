"""Reference activity detection and continuous pre-emphasis."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from ..contracts import positive_integer, finite_scalar, audio_array


def nonlinear_playback_reference(playback_audio, *, transition_amplitude):
    """Expand mono playback into linear and bounded odd nonlinear reference channels.

    The second channel changes polarity at the specified absolute playback
    amplitude. Both channels can be processed by one two-reference echo filter.
    """
    playback = audio_array(playback_audio)
    if playback.shape[1] != 1:
        raise ValueError('nonlinear playback expansion requires a mono reference')
    finite_scalar(transition_amplitude, 'transition_amplitude', minimum=np.finfo(float).eps)
    amplitude = np.abs(playback)
    gain = (transition_amplitude - amplitude) / (transition_amplitude + amplitude)
    return np.column_stack((playback, playback * gain))


@dataclass(frozen=True)
class InputConditioningConfig:
    hop_length_samples: int = 512
    microphone_count: int = 1
    reference_count: int = 1
    preemphasis: float = 0.9
    enable_preemphasis: bool = True
    reference_gain: float = 1.0
    clipping_threshold: float = 32766/32768

    def __post_init__(self):
        for name in ('hop_length_samples', 'microphone_count', 'reference_count'):
            positive_integer(getattr(self, name), name)
        finite_scalar(self.preemphasis, 'preemphasis', minimum=0, maximum=1)
        finite_scalar(self.reference_gain, 'reference_gain', minimum=0)
        finite_scalar(self.clipping_threshold, 'clipping_threshold', minimum=np.finfo(float).eps)


class EchoInputConditioner:
    def __init__(self, config=InputConditioningConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.microphone_memory = np.zeros(self.config.microphone_count)
        self.reference_memory = np.zeros(self.config.reference_count)
        self.is_clipped = False

    def process_block(self, microphone_audio, reference_audio):
        microphone, reference = audio_array(microphone_audio), audio_array(reference_audio)
        config = self.config
        if (microphone.shape != (config.hop_length_samples, config.microphone_count)
                or reference.shape != (config.hop_length_samples, config.reference_count)):
            raise ValueError('microphone and reference blocks must match the configured shapes')
        microphone, reference = microphone.copy(), reference*config.reference_gain
        self.is_clipped = bool(max(np.max(np.abs(microphone)), np.max(np.abs(reference))) >= config.clipping_threshold)
        if config.enable_preemphasis:
            for values, memory in ((microphone, self.microphone_memory), (reference, self.reference_memory)):
                final = values[-1].copy()
                values[1:] -= config.preemphasis*values[:-1]
                values[0] -= config.preemphasis*memory
                memory[:] = final
        return microphone, reference


@dataclass(frozen=True)
class ReferenceActivityConfig:
    amplitude_threshold: float = 4.5776e-4
    activation_frames: int = 1
    release_frames: int = 30

    def __post_init__(self):
        finite_scalar(self.amplitude_threshold, 'amplitude_threshold', minimum=0)
        positive_integer(self.activation_frames, 'activation_frames')
        positive_integer(self.release_frames, 'release_frames')


class ReferenceActivityDetector:
    def __init__(self, config=ReferenceActivityConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.is_active = False
        self.active_frames = self.silent_frames = 0
        self.amplitude_deviation = 0.0

    def process_block(self, reference_audio):
        reference = audio_array(reference_audio)
        if not len(reference):
            raise ValueError('reference block must not be empty')
        envelope = np.abs(reference).sum(axis=1)
        self.amplitude_deviation = float(np.mean(np.abs(envelope-envelope.mean())))
        if not self.is_active:
            self.active_frames = self.active_frames+1 if self.amplitude_deviation > self.config.amplitude_threshold else 0
            if self.active_frames >= self.config.activation_frames:
                self.is_active, self.active_frames = True, 0
        else:
            self.silent_frames = self.silent_frames+1 if self.amplitude_deviation < self.config.amplitude_threshold else 0
            if self.silent_frames >= self.config.release_frames:
                self.is_active, self.silent_frames = False, 0
        return self.is_active
