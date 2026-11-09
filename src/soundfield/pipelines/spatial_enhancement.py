"""Complete audio operations for spatial suppression and dereverberation."""

import numpy as np

from ..beamforming import MaskGuidedBeamformer, MaskGuidedConfig, MvdrConfig
from ..contracts import audio_array, unit_interval_array
from ..dereverb import WeightedPredictionDereverberator, WpeConfig
from ..enhancement import CorrelationNoiseSuppressor, CorrelationSuppressionConfig
from ..transform import FourierConfig, ShortTimeFourierTransform


def enhance_with_spatial_mask(audio, speech_gains, config: MaskGuidedConfig | None = None, *, transform_config=None):
    """Return mono audio from a (bins, frames, 1) speech-gain cube."""
    samples = audio_array(audio)
    config = config if config is not None else MaskGuidedConfig(MvdrConfig(microphone_count=samples.shape[1]))
    transform = ShortTimeFourierTransform(transform_config)
    dimensions = config.beamformer
    if samples.shape[1] != dimensions.microphone_count or transform.config.frequency_bin_count != dimensions.frequency_bin_count:
        raise ValueError("audio, transform, and beamformer dimensions must agree")
    spectrum = transform.analyze(samples)
    gains = unit_interval_array(speech_gains, "speech_gains", (*spectrum.shape[:2], 1))
    engine = MaskGuidedBeamformer(config)
    output = np.empty((*spectrum.shape[:2], 1), dtype=np.complex128)
    for frame in range(spectrum.shape[1]):
        output[:, frame] = engine.process_frame(spectrum[:, frame], gains[:, frame])
    return transform.synthesize(output, length_samples=len(samples))


def suppress_correlated_noise(audio, config: CorrelationSuppressionConfig | None = None, *, transform_config=None):
    """Enhance all channels with independent selected-channel histories."""
    samples = audio_array(audio)
    config = config if config is not None else CorrelationSuppressionConfig(channel_count=samples.shape[1])
    length = 2 * (config.frequency_bin_count - 1)
    transform = ShortTimeFourierTransform(transform_config if transform_config is not None else FourierConfig(length, length // 2))
    if samples.shape[1] != config.channel_count or transform.config.frequency_bin_count != config.frequency_bin_count:
        raise ValueError("audio, transform, and suppressor dimensions must agree")
    spectrum = transform.analyze(samples)
    output = np.empty_like(spectrum)
    engine = CorrelationNoiseSuppressor(config)
    for frame in range(spectrum.shape[1]):
        for channel in range(samples.shape[1]):
            output[:, frame, channel:channel + 1] = engine.process_frame(spectrum[:, frame], channel)
    return transform.synthesize(output, length_samples=len(samples))


def reduce_reverberation(audio, config: WpeConfig | None = None, *, transform_config=None, compensate_delay=False,
                         protect_short_decays=False, sample_rate_hz=16000):
    """Return the input sample count, optionally draining and removing frame delay.

    Without delay compensation, leading delayed frames are retained. With it,
    zero input frames drain the tail before the delayed prefix is removed.
    """
    samples = audio_array(audio)
    config = config if config is not None else WpeConfig(channel_count=samples.shape[1])
    if not isinstance(compensate_delay, bool):
        raise ValueError("compensate_delay must be boolean")
    transform = ShortTimeFourierTransform(transform_config if transform_config is not None else FourierConfig(config.frame_length_samples, config.frame_length_samples // 2))
    if samples.shape[1] != config.channel_count or transform.config.frequency_bin_count != config.frequency_bin_count:
        raise ValueError("audio, transform, and predictor dimensions must agree")
    spectrum = transform.analyze(samples)
    dry = 0.
    if protect_short_decays:
        if sample_rate_hz != 16000 or config.frame_length_samples != 512 or samples.shape[1] != 2 or not compensate_delay:
            raise ValueError("short-decay protection requires aligned 16 kHz two-microphone output on a 512-point transform")
        from ..dereverb.recording_decay import dry_fraction
        dry = dry_fraction(samples, spectrum)
        if np.isscalar(dry) and dry == 1.:
            return samples.copy()
    engine = WeightedPredictionDereverberator(config)
    frames = spectrum.shape[1]
    delay = engine.latency_frames if compensate_delay else 0
    for frame in range(frames):
        processed = engine.process_frame(spectrum[:, frame])
        if frame >= delay:
            spectrum[:, frame - delay] = processed
    pending = min(delay, frames)
    if pending:
        spectrum[:, -pending:] = engine.flush()[:, -pending:]
    output = transform.synthesize(spectrum, length_samples=len(samples))
    return output + dry*(samples-output) if protect_short_decays else output
