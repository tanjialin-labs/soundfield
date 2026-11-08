"""One-sided short-time Fourier analysis with explicit overlap-add rules."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..contracts import audio_array, positive_integer, spectral_array

WINDOW_NAMES = (
    "sqrt_hann", "hann", "hamming", "rectangular",
    "hann_periodic", "sqrt_hann_periodic",
)


@dataclass(frozen=True)
class FourierConfig:
    frame_length_samples: int = 512
    hop_length_samples: int = 256
    window: str = "sqrt_hann"

    def __post_init__(self):
        positive_integer(self.frame_length_samples, "frame_length_samples")
        positive_integer(self.hop_length_samples, "hop_length_samples")
        if self.frame_length_samples < 4 or self.frame_length_samples % 2:
            raise ValueError("frame_length_samples must be even and at least 4")
        if self.hop_length_samples > self.frame_length_samples:
            raise ValueError("hop_length_samples cannot exceed frame_length_samples")
        if self.window not in WINDOW_NAMES:
            raise ValueError(f"window must be one of {WINDOW_NAMES}")

    @property
    def frequency_bin_count(self) -> int:
        return self.frame_length_samples // 2 + 1


class ShortTimeFourierTransform:
    """Transform real audio without centering or leading padding.

    Audio has shape (samples, channels); spectra have shape
    (bins, frames, channels). The final incomplete frame is zero-padded.
    """

    def __init__(self, config: FourierConfig | None = None):
        self._config = config if config is not None else FourierConfig()
        if not isinstance(self.config, FourierConfig):
            raise TypeError("config must be a FourierConfig")
        length = self.config.frame_length_samples
        name = self.config.window
        if name == "rectangular":
            window = np.ones(length)
        elif name == "hamming":
            window = np.hamming(length)
        else:
            window = np.hanning(length + 1)[:-1] if "periodic" in name else np.hanning(length)
            if name.startswith("sqrt_"):
                window = np.sqrt(np.maximum(window, 0.0))
        self._window = np.asarray(window, dtype=np.float64)
        self._window.flags.writeable = False

    @property
    def config(self) -> FourierConfig:
        return self._config

    @property
    def analysis_window(self):
        return self._window.copy()

    def analyze(self, audio):
        samples = audio_array(audio)
        length = self.config.frame_length_samples
        hop = self.config.hop_length_samples
        sample_count, channels = samples.shape
        if sample_count == 0:
            return np.empty((self.config.frequency_bin_count, 0, channels), dtype=np.complex128)
        frame_count = 1 + max(0, (sample_count - length + hop - 1) // hop)
        padded_length = (frame_count - 1) * hop + length
        padded = samples if padded_length == sample_count else np.pad(samples, ((0, padded_length - sample_count), (0, 0)))
        frames = np.lib.stride_tricks.sliding_window_view(padded, length, axis=0)[::hop]
        frames = frames.transpose(2, 0, 1)
        return np.fft.rfft(frames * self._window[:, None, None], axis=0)

    def _synthesis_parameters(self):
        length, hop, name = self.config.frame_length_samples, self.config.hop_length_samples, self.config.window
        normalize = not (length == 2 * hop and name in ("sqrt_hann", "hann"))
        synthesis_window = np.ones(length) if name in ("hann", "hann_periodic") else self._window
        if name == "hann" and normalize:
            synthesis_window = self._window
        denominator_window = np.full(length, 0.5) if "periodic" in name else self._window**2
        return normalize, synthesis_window, denominator_window

    def synthesize(self, spectrum, *, length_samples: int | None = None):
        frequencies = spectral_array(spectrum)
        if frequencies.shape[0] != self.config.frequency_bin_count:
            raise ValueError("spectrum frequency bins do not match the configuration")
        length = self.config.frame_length_samples
        hop = self.config.hop_length_samples
        frame_count, channels = frequencies.shape[1:]
        output_length = (frame_count - 1) * hop + length if frame_count else 0
        if length_samples is not None:
            if (isinstance(length_samples, bool) or not isinstance(length_samples, Integral)
                    or not 0 <= length_samples <= output_length):
                raise ValueError("length_samples must be between zero and the synthesized length")
        output = np.zeros((output_length, channels), dtype=np.float64)
        if frame_count == 0:
            return output
        frames = np.fft.irfft(frequencies, n=length, axis=0)
        normalize, synthesis_window, denominator_window = self._synthesis_parameters()
        denominator = np.zeros(output_length) if normalize else None
        for frame_index in range(frame_count):
            start = frame_index * hop
            output[start:start + length] += frames[:, frame_index, :] * synthesis_window[:, None]
            if normalize:
                denominator[start:start + length] += denominator_window
        if normalize:
            np.divide(output, denominator[:, None], out=output, where=denominator[:, None] > 0)
            output[denominator == 0] = 0.0
        return output if length_samples is None else output[:length_samples]
