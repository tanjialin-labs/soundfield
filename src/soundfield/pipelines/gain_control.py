"""Speech-aware gain control of complete audio recordings."""

import numpy as np

from ..contracts import audio_array, real_array, finite_scalar
from ..metrics.programme_levels import interpolated_peak
from ..enhancement.automatic_gain import AutomaticGainControl, GainControlConfig
from ..transform.short_time_fourier import FourierConfig, ShortTimeFourierTransform


def apply_automatic_gain(audio, speech_gains, *, config: GainControlConfig | None = None,
                         reference_activity=None, peak_ceiling_dbtp=None, sample_rate_hz=16000):
    """Process all frames and return audio with the original sample count.

    speech_gains has shape (bins, frames, channels). Optional reference
    activity has shape (samples, 1) and is shared by the audio channels.
    """
    samples = audio_array(audio)
    config = config if config is not None else GainControlConfig()
    transform = ShortTimeFourierTransform(FourierConfig(
        config.frame_length_samples, config.hop_length_samples,
    ))
    spectra = transform.analyze(samples)
    gains = real_array(speech_gains, "speech_gains", 3)
    if gains.shape != spectra.shape or np.any((gains < 0) | (gains > 1)):
        raise ValueError("speech_gains must match the analyzed spectrum and lie between zero and one")
    reference = None
    if reference_activity is not None:
        reference = real_array(reference_activity, "reference_activity", 2)
        if reference.shape != (len(samples), 1) or np.any((reference < 0) | (reference > 1)):
            raise ValueError("reference_activity must have shape (samples, 1) and lie between zero and one")
    if not len(samples):
        return samples.copy()
    hop = config.hop_length_samples
    padded_length = max(len(samples), spectra.shape[1] * hop)
    padded = np.pad(samples, ((0, padded_length - len(samples)), (0, 0)))
    if reference is not None:
        reference = np.pad(reference, ((0, padded_length - len(reference)), (0, 0)))
    output_spectra = np.empty_like(spectra)
    for channel_index in range(samples.shape[1]):
        processor = AutomaticGainControl(config)
        for frame_index in range(spectra.shape[1]):
            start = frame_index * hop
            output_spectra[:, frame_index, channel_index:channel_index + 1] = processor.process_frame(
                spectra[:, frame_index, channel_index:channel_index + 1],
                audio_hop=padded[start:start + hop, channel_index:channel_index + 1],
                speech_gain=gains[:, frame_index, channel_index:channel_index + 1],
                reference_activity=0.0 if reference is None else reference[start:start + hop],
            )
    output = transform.synthesize(output_spectra, length_samples=len(samples))
    if peak_ceiling_dbtp is not None:
        finite_scalar(peak_ceiling_dbtp, "peak_ceiling_dbtp", minimum=-120, maximum=0)
        ceiling = 10**(peak_ceiling_dbtp/20.)
        peak = interpolated_peak(output,sample_rate_hz).max()
        if peak > ceiling:
            output *= ceiling/peak
    return output
