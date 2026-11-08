"""Overlap-add synthesis, continuous de-emphasis and output scaling."""

import numpy as np
from scipy.signal import lfilter

from ..contracts import positive_integer, finite_scalar, spectral_frame


class EchoAudioSynthesizer:
    def __init__(self, hop_length_samples=512, channel_count=1, *, preemphasis=0.9,
                 enable_deemphasis=True, output_gain=1.0):
        positive_integer(hop_length_samples, 'hop_length_samples')
        positive_integer(channel_count, 'channel_count')
        finite_scalar(preemphasis, 'preemphasis', minimum=0, maximum=1)
        finite_scalar(output_gain, 'output_gain', minimum=0)
        self.hop_length_samples, self.channel_count = hop_length_samples, channel_count
        self.preemphasis, self.enable_deemphasis, self.output_gain = preemphasis, enable_deemphasis, output_gain
        self.window = np.sqrt(np.hanning(2*hop_length_samples))
        self.reset()

    def reset(self):
        self.overlap = np.zeros((self.hop_length_samples, self.channel_count))
        self.deemphasis_memory = np.zeros((1, self.channel_count))
        self.has_pending_output = False

    def _finish(self, values):
        if self.enable_deemphasis:
            values, self.deemphasis_memory = lfilter([1], [1, -self.preemphasis], values,
                                                      axis=0, zi=self.deemphasis_memory)
        return values*self.output_gain

    def process_frame(self, spectrum_frame):
        spectrum = spectral_frame(spectrum_frame, self.hop_length_samples+1, self.channel_count)
        time = np.fft.irfft(spectrum, n=2*self.hop_length_samples, axis=0)*self.window[:, None]
        output = self.overlap+time[:self.hop_length_samples]
        self.overlap = time[self.hop_length_samples:].copy()
        self.has_pending_output = True
        return self._finish(output)

    def flush(self):
        if not self.has_pending_output:
            return np.zeros((0, self.channel_count))
        output = self._finish(self.overlap)
        self.overlap.fill(0)
        self.has_pending_output = False
        return output
