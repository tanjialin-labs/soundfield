"""Independent-channel background noise suppression for complete recordings."""

import numpy as np

from ..aec.noise_tracking import EchoNoiseConfig
from ..contracts import audio_array, positive_integer
from ..enhancement.statistical_suppression import StatisticalNoiseSuppressor
from ..transform import ShortTimeFourierTransform


def suppress_background_noise(audio, sample_rate_hz, *, minimum_gain=.16, transform_config=None):
    """Suppress additive noise without a model or a separate noise reference."""
    samples = audio_array(audio)
    positive_integer(sample_rate_hz, "sample_rate_hz")
    transform = ShortTimeFourierTransform(transform_config)
    bins = transform.config.frequency_bin_count
    bins_per_hz = transform.config.frame_length_samples / sample_rate_hz
    start = min(bins - 1, int(np.ceil(200 * bins_per_hz)))
    stop = min(bins, max(start + 1, int(4250 * bins_per_hz) + 1))
    config = EchoNoiseConfig(frequency_bin_count=bins, speech_bins=(start, stop))
    spectra = transform.analyze(samples)
    for channel in range(samples.shape[1]):
        processor = StatisticalNoiseSuppressor(config, minimum_gain=minimum_gain)
        for frame in range(spectra.shape[1]):
            current = spectra[:, frame, channel:channel+1]
            current[:] = processor.process_frame(current)
    return transform.synthesize(spectra, length_samples=len(samples))
