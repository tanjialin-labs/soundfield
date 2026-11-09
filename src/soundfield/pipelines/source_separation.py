"""Whole-recording source separation with explicit analysis parameters."""

import numpy as np

from ..contracts import audio_array, nonnegative_integer, positive_integer
from ..separation import DemixingConfig, OnlineSourceSeparator
from ..separation.stationary_pair import render_stationary_pair
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


def separate_stationary_pair(audio, *, iterations=12, reference_channel_index=1, transform_config=None,
                             speech_enhancer=None, speaker_router=None, sample_rate_hz=16000):
    """Separate two microphones using one demixing matrix per bin fitted to the entire recording."""
    samples = audio_array(audio)
    positive_integer(iterations, "iterations")
    nonnegative_integer(reference_channel_index, "reference_channel_index")
    if samples.shape[1] != 2 or reference_channel_index >= 2:
        raise ValueError("stationary pair separation requires two microphones and a valid reference channel")
    transform = ShortTimeFourierTransform(transform_config or FourierConfig(4096, 2048))
    spectrum = transform.analyze(samples)
    render_stationary_pair(spectrum, iterations, reference_channel_index)
    images = transform.synthesize(spectrum, length_samples=len(samples))
    output = images
    if speech_enhancer is not None:
        output = np.column_stack([speech_enhancer.enhance_audio(images[:, i:i+1], sample_rate_hz,
                                  phase_preserving_floor=.65)[:, 0] for i in range(2)])
    return _route_images(images, output, speaker_router, sample_rate_hz)


def _route_images(images, output, speaker_router, sample_rate_hz):
    if speaker_router is not None:
        positive_integer(sample_rate_hz, 'sample_rate_hz')
        speaker_router.reset()
        output = output.copy()
        for start in range(0, len(images), 2*sample_rate_hz):
            span = slice(start, start+2*sample_rate_hz)
            speaker_router.process_block(images[span])
            output[span] = output[span][:, speaker_router.source_order]
    return output


def separate_playback_pair(audio, playback_audio, *, sample_rate_hz=16000, reference_channel_index=1,
                           speaker_router=None):
    """Separate two recorded microphones with a synchronized playback reference at 16 kHz."""
    from ..aec import EchoCancellationConfig, EchoFilterConfig, InputConditioningConfig, cancel_echo
    from ..separation.playback_transfer import subtract_recorded_playback

    microphones = audio_array(audio)
    playback = audio_array(playback_audio)
    nonnegative_integer(reference_channel_index, "reference_channel_index")
    if (sample_rate_hz != 16000 or microphones.shape[1] != 2 or reference_channel_index >= 2
            or len(playback) != len(microphones) or not 1 <= playback.shape[1] <= 3):
        raise ValueError("two microphones and one to three aligned 16 kHz playback channels are required")
    active = np.any(playback)
    if active:
        config = EchoCancellationConfig(input=InputConditioningConfig(microphone_count=2, reference_count=playback.shape[1]),
                                        linear_filter=EchoFilterConfig(reference_count=playback.shape[1]), communication_filter=None)
        microphones, _ = cancel_echo(microphones, playback, config)
    images = separate_stationary_pair(microphones, reference_channel_index=reference_channel_index)
    output = images
    if active:
        fitted = subtract_recorded_playback(images, playback)
        output = .5*(images + fitted)
    return _route_images(images, output, speaker_router, sample_rate_hz)
