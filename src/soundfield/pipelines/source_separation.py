"""Whole-recording source separation with explicit analysis parameters."""

import numpy as np

from ..contracts import audio_array
from ..separation import DemixingConfig, OnlineSourceSeparator
from ..transform import FourierConfig, ShortTimeFourierTransform


def separate_sources(audio, config: DemixingConfig | None = None, *, transform_config=None):
    """Analyze, separate all frames, and return the original sample count."""
    samples = audio_array(audio)
    config = config if config is not None else DemixingConfig(channel_count=samples.shape[1])
    if samples.shape[1] != config.channel_count:
        raise ValueError("audio channels must agree with the separation configuration")
    if config.nearfield_control is not None:
        raise ValueError("nearfield evidence requires the frame-level separator interface")
    if transform_config is None:
        frame_length = 2 * (config.frequency_bin_count - 1)
        transform_config = FourierConfig(frame_length, frame_length // 2)
    transform = ShortTimeFourierTransform(transform_config)
    if transform.config.frequency_bin_count != config.frequency_bin_count:
        raise ValueError("transform and separator frequency bins must agree")
    spectrum = transform.analyze(samples)
    separator = OnlineSourceSeparator(config)
    output = np.empty_like(spectrum)
    for frame in range(spectrum.shape[1]):
        output[:, frame] = separator.process_frame(spectrum[:, frame])
    return transform.synthesize(output, length_samples=len(samples))
