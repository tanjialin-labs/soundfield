"""Context-aligned neural masking, beamforming and automatic gain control."""

import numpy as np

from ..contracts import audio_array
from ..transform import FourierConfig, ShortTimeFourierTransform
from ..neural import SpectralMaskEstimator
from ..neural.band_features import round_away_from_zero
from ..enhancement.automatic_gain import AutomaticGainControl, GainControlConfig
from ..beamforming import MaskGuidedBeamformer, MaskGuidedConfig, MvdrConfig


def enhance_with_neural_masks(audio, mask_estimator: SpectralMaskEstimator, *, beamform=False, apply_gain_control=True):
    """Process normalized audio; right-context frames are padded and aligned explicitly."""
    samples = audio_array(audio)
    if samples.shape[1] != mask_estimator.channel_count:
        raise ValueError('audio channels must match the mask estimator')
    hop = mask_estimator.config.frequency_bin_count-1
    transform = ShortTimeFourierTransform(FourierConfig(2*hop, hop))
    spectrum = transform.analyze(samples)
    output_channels = 1 if beamform else samples.shape[1]
    output = np.zeros((hop+1, spectrum.shape[1], output_channels), complex)
    mask_estimator.reset()
    spatial = MaskGuidedBeamformer(MaskGuidedConfig(MvdrConfig(frequency_bin_count=hop+1, microphone_count=samples.shape[1]))) if beamform else None
    controllers = [AutomaticGainControl(GainControlConfig(frame_length_samples=2*hop, hop_length_samples=hop,
                    speech_band_start=min(15, hop), speech_band_stop=min(64, hop+1),
                    equalizer_start_bin=min(32, hop), equalizer_stop_bin=min(66, hop+1))) for _ in range(output_channels)] if apply_gain_control else []
    context = mask_estimator.config.right_context_frames
    for frame in range(spectrum.shape[1]+context):
        current = spectrum[:, frame] if frame < spectrum.shape[1] else np.zeros((hop+1, samples.shape[1]), complex)
        quantized = round_away_from_zero(current.real*2**21)+1j*round_away_from_zero(current.imag*2**21)
        result = mask_estimator.process_frame(quantized)
        target = result.target_frame_index
        if target < 0 or target >= spectrum.shape[1]:
            continue
        gain = result.gains
        enhanced = (spatial.process_frame(spectrum[:, target], gain[:, :1]) if spatial else spectrum[:, target]*gain)
        if controllers:
            hop_audio = np.fft.irfft(enhanced, n=2*hop, axis=0)[:hop]
            for channel, controller in enumerate(controllers):
                enhanced[:, channel:channel+1] = controller.process_frame(enhanced[:, channel:channel+1],
                    audio_hop=hop_audio[:, channel:channel+1], speech_gain=gain[:, channel:channel+1])
        output[:, target] = enhanced
    return transform.synthesize(output, length_samples=len(samples))
