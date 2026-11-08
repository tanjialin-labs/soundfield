"""Incremental Fourier analysis and overlap-add synthesis for arbitrary audio blocks."""

import numpy as np

from ..contracts import audio_array, spectral_array
from .short_time_fourier import FourierConfig, ShortTimeFourierTransform


class FourierAnalyzer:
    """Emit complete frames; flush pads the final incomplete frame once."""

    def __init__(self, config=FourierConfig()):
        self.transform = ShortTimeFourierTransform(config)
        self.config = config
        self.reset()

    def reset(self):
        self.sample_count = self.frame_count = 0
        self._pending = None
        self._closed = False

    def _empty(self):
        channels = self._pending.shape[1] if self._pending is not None else 1
        return np.empty((self.config.frequency_bin_count, 0, channels), complex)

    def process_block(self, audio):
        if self._closed:
            raise ValueError('reset the analyzer before starting another stream')
        audio = audio_array(audio)
        if self._pending is None:
            self._pending = np.empty((0, audio.shape[1]))
        if audio.shape[1] != self._pending.shape[1]:
            raise ValueError('channel count must remain constant within a stream')
        self.sample_count += len(audio)
        self._pending = np.concatenate((self._pending, audio))
        length, hop = self.config.frame_length_samples, self.config.hop_length_samples
        if len(self._pending) < length:
            return self._empty()
        count = 1+(len(self._pending)-length)//hop
        frames = self.transform.analyze(self._pending[:(count-1)*hop+length])
        self._pending = self._pending[count*hop:].copy()
        self.frame_count += count
        return frames

    def flush(self):
        if self._closed or not self.sample_count:
            self._closed = True
            return self._empty()
        self._closed = True
        length, hop = self.config.frame_length_samples, self.config.hop_length_samples
        required = 1+max(0, (self.sample_count-length+hop-1)//hop)
        if self.frame_count == required:
            return self._empty()
        frames = self.transform.analyze(self._pending)
        self.frame_count += frames.shape[1]
        return frames


class FourierSynthesizer:
    """Emit one hop per frame; flush returns the remaining padded output extent."""

    def __init__(self, config=FourierConfig()):
        self.transform = ShortTimeFourierTransform(config)
        self.config = config
        self._normalize, self._window, self._denominator_window = self.transform._synthesis_parameters()
        self.reset()

    def reset(self):
        self.frame_count = 0
        self._tail = None
        self._denominator = np.zeros(self.config.frame_length_samples)
        self._closed = False

    def _render(self, length):
        output = self._tail[:length].copy()
        if self._normalize:
            denominator = self._denominator[:length, None]
            np.divide(output, denominator, out=output, where=denominator > 0)
            output[self._denominator[:length] == 0] = 0
        return output

    def process_block(self, spectrum):
        if self._closed:
            raise ValueError('reset the synthesizer before starting another stream')
        spectrum = spectral_array(spectrum)
        length, hop = self.config.frame_length_samples, self.config.hop_length_samples
        if spectrum.shape[0] != self.config.frequency_bin_count:
            raise ValueError('frequency bins must match the configured transform')
        if self._tail is None:
            self._tail = np.zeros((length, spectrum.shape[2]))
        if spectrum.shape[2] != self._tail.shape[1]:
            raise ValueError('channel count must remain constant within a stream')
        frames = np.fft.irfft(spectrum, n=length, axis=0) if spectrum.shape[1] else np.empty((length, 0, spectrum.shape[2]))
        output = np.empty((spectrum.shape[1]*hop, spectrum.shape[2]))
        for index in range(spectrum.shape[1]):
            self._tail += frames[:, index]*self._window[:, None]
            if self._normalize:
                self._denominator += self._denominator_window
            output[index*hop:(index+1)*hop] = self._render(hop)
            self._tail[:-hop] = self._tail[hop:]
            self._tail[-hop:] = 0
            self._denominator[:-hop] = self._denominator[hop:]
            self._denominator[-hop:] = 0
        self.frame_count += spectrum.shape[1]
        return output

    def flush(self):
        channels = self._tail.shape[1] if self._tail is not None else 1
        if self._closed or not self.frame_count:
            self._closed = True
            return np.empty((0, channels))
        self._closed = True
        return self._render(self.config.frame_length_samples-self.config.hop_length_samples)
