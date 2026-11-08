"""WAV input and output with normalized floating-point samples."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ..contracts import audio_array, positive_integer


def read_audio(path):
    """Return (sample_rate_hz, audio) with a retained channel axis."""
    sample_rate_hz, samples = wavfile.read(Path(path))
    if samples.dtype.kind == "u":
        midpoint = float(2 ** (8 * samples.dtype.itemsize - 1))
        audio = (samples.astype(np.float64) - midpoint) / midpoint
    elif samples.dtype.kind == "i":
        audio = samples.astype(np.float64) / float(2 ** (8 * samples.dtype.itemsize - 1))
    else:
        audio = samples.astype(np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    return int(sample_rate_hz), audio_array(audio)


def write_audio(path, audio, sample_rate_hz: int, *, encoding: str = "pcm16"):
    """Write WAV audio; PCM16 output is rounded and limited to its range."""
    positive_integer(sample_rate_hz, "sample_rate_hz")
    samples = audio_array(audio)
    if encoding == "pcm16":
        data = np.rint(np.clip(samples, -1.0, 32767 / 32768) * 32768).astype(np.int16)
    elif encoding == "float64":
        data = samples
    else:
        raise ValueError("encoding must be 'pcm16' or 'float64'")
    wavfile.write(Path(path), sample_rate_hz, data)
