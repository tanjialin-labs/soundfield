"""Causal alignment of playback to microphone time using normalized correlation."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

from ..contracts import audio_array, finite_scalar, positive_integer


@dataclass(frozen=True)
class ReferenceAlignmentConfig:
    maximum_delay_samples: int = 4096
    observation_samples: int = 2048
    update_interval_blocks: int = 4
    minimum_correlation: float = .35

    def __post_init__(self):
        for name in ('maximum_delay_samples', 'observation_samples', 'update_interval_blocks'):
            positive_integer(getattr(self, name), name)
        finite_scalar(self.minimum_correlation, 'minimum_correlation', minimum=np.finfo(float).eps, maximum=1.)


class PlaybackReferenceAligner:
    """Use one shared delay for all reference channels. Positive delays describe playback leading capture."""

    def __init__(self, config, hop_samples, reference_channels):
        positive_integer(hop_samples, "hop_samples")
        positive_integer(reference_channels, "reference_channels")
        self.config = config
        self.hop = hop_samples
        self.channels = reference_channels
        self.reset()

    def reset(self):
        length = self.config.observation_samples + self.config.maximum_delay_samples + self.hop
        self.reference = np.zeros((length, self.channels))
        self.microphone = np.zeros(self.config.observation_samples)
        self.delay_samples = 0.
        self._blocks = 0

    def _measure(self):
        width = len(self.microphone)
        source = self.reference[-(width + self.config.maximum_delay_samples):]
        channel = np.argmax(np.sum(source * source, axis=0))
        ref = source[:, channel]
        mic = self.microphone - self.microphone.mean()
        energy = np.r_[0., np.cumsum(ref * ref)]
        sums = np.r_[0., np.cumsum(ref)]
        variance = np.maximum(0., energy[width:] - energy[:-width] - (sums[width:] - sums[:-width])**2 / width)
        denom = np.sqrt(variance * np.dot(mic, mic))
        scores = np.divide(abs(correlate(ref, mic, mode='valid', method='fft')), denom,
                           out=np.zeros_like(denom), where=denom > 1e-12)[::-1]
        best = int(np.argmax(scores))
        if scores[best] < self.config.minimum_correlation:
            return None
        offset = 0.
        if 0 < best < len(scores)-1:
            left, middle, right = scores[best-1:best+2]
            curvature = 2*middle-left-right
            if curvature > 0:
                offset = .5*(right-left)/curvature
        return best + offset

    def process_block(self, microphone, reference):
        mic, ref = audio_array(microphone), audio_array(reference)
        if len(mic) != self.hop or not mic.shape[1] or ref.shape != (self.hop, self.channels):
            raise ValueError('microphone and playback must share the configured hop')
        self._blocks += 1
        self.reference[:-self.hop] = self.reference[self.hop:]
        self.reference[-self.hop:] = ref
        width = min(self.hop, len(self.microphone))
        self.microphone[:-width] = self.microphone[width:]
        self.microphone[-width:] = mic[-width:, 0]
        if self._blocks*self.hop >= len(self.microphone) and self._blocks % self.config.update_interval_blocks == 0:
            measured = self._measure()
            if measured is not None:
                self.delay_samples = measured
        positions = len(self.reference)-self.hop + np.arange(self.hop) - self.delay_samples
        return np.column_stack([np.interp(positions, np.arange(len(self.reference)), ch) for ch in self.reference.T])
