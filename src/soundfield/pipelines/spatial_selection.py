"""Audio routing and spatial classification with explicit evidence."""

import numpy as np

from ..contracts import audio_array, unit_interval_array
from ..localization import NearfieldClassificationConfig, NearfieldSpatialClassifier
from ..selection import ChannelRoutingConfig, HysteresisChannelSelector
from ..transform import FourierConfig, ShortTimeFourierTransform


def route_audio_channels(audio, config: ChannelRoutingConfig | None = None, *, evidence_audio=None,
                         speech_probabilities=None, transform_config=None, compensate_delay=False):
    """Route one channel or preserve channel slots with the others set to zero."""
    samples = audio_array(audio)
    config = config if config is not None else ChannelRoutingConfig(channel_count=samples.shape[1])
    if not isinstance(compensate_delay, bool):
        raise ValueError("compensate_delay must be boolean")
    transform = ShortTimeFourierTransform(transform_config)
    if samples.shape[1] != config.channel_count or transform.config.frequency_bin_count != config.frequency_bin_count:
        raise ValueError("audio, transform, and channel selector dimensions must agree")
    evidence = samples if evidence_audio is None else audio_array(evidence_audio)
    if evidence.shape != samples.shape:
        raise ValueError("evidence_audio must match audio samples and channels")
    spectrum = transform.analyze(samples)
    evidence_spectrum = transform.analyze(evidence)
    probabilities = None if speech_probabilities is None else unit_interval_array(speech_probabilities, "speech_probabilities", spectrum.shape)
    engine = HysteresisChannelSelector(config)
    output = np.empty((*spectrum.shape[:2], config.output_channel_count), complex)
    for frame in range(spectrum.shape[1]):
        output[:, frame] = engine.select(spectrum[:, frame], evidence_spectrum[:, frame], None if probabilities is None else probabilities[:, frame])
    if compensate_delay and spectrum.shape[1]:
        extended = np.concatenate((output, engine.flush()), axis=1)
        output = extended[:, engine.latency_frames:engine.latency_frames + spectrum.shape[1]]
    return transform.synthesize(output, length_samples=len(samples))


def classify_nearfield_audio(audio, config: NearfieldClassificationConfig | None = None, *, transform_config=None):
    """Apply near-field channel gains, padding the last classification block."""
    samples = audio_array(audio)
    config = config if config is not None else NearfieldClassificationConfig()
    transform = ShortTimeFourierTransform(transform_config if transform_config is not None else FourierConfig(config.frame_length_samples, config.frame_length_samples // 2))
    if samples.shape[1] != len(config.microphone_positions_meters) or transform.config.frequency_bin_count != config.frequency_bin_count:
        raise ValueError("audio, transform, and spatial classifier dimensions must agree")
    spectrum = transform.analyze(samples)
    engine = NearfieldSpatialClassifier(config)
    frame_count = spectrum.shape[1]
    padded_frames = (frame_count + config.block_frames - 1) // config.block_frames * config.block_frames
    output = np.empty((config.frequency_bin_count, padded_frames, len(config.output_microphone_indices)), complex)
    silence = np.zeros((config.frequency_bin_count, samples.shape[1]), complex)
    for frame in range(padded_frames):
        result = engine.estimate(spectrum[:, frame] if frame < frame_count else silence)
        if result is not None:
            output[:, frame + 1 - config.block_frames:frame + 1] = result.spectrum
    return transform.synthesize(output[:, :frame_count], length_samples=len(samples))
