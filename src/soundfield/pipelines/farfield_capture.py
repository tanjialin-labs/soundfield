"""Far-field speech capture from beam streams or microphone arrays."""

from dataclasses import dataclass
import numpy as np
from ..contracts import audio_array, spectral_array, unit_interval_array, finite_scalar
from ..transform import FourierConfig, ShortTimeFourierTransform
from ..neural.band_features import round_away_from_zero
from ..selection.speech_beam import SpeechBeamConfig, SpeechBeamSelector
from ..beamforming import MaskGuidedBeamformer, MaskGuidedConfig, MvdrConfig
from ..enhancement.automatic_gain import AutomaticGainControl, GainControlConfig
from ..numerics.spectral_functions import rescale_upper_range


@dataclass(frozen=True)
class FarfieldCaptureResult:
    audio: np.ndarray
    input_spectrum: np.ndarray
    feature_spectrum: np.ndarray
    speech_masks: np.ndarray
    enhanced_spectrum: np.ndarray
    gain_control_spectrum: np.ndarray
    gain_control_evidence: np.ndarray
    gain_db: np.ndarray
    selected_channels: np.ndarray
    noise_levels_db: np.ndarray
    context_delay_samples: int


def estimate_aligned_masks(spectrum, estimator):
    """Drain right context and place every mask on its input frame index."""
    values = spectral_array(spectrum)
    bins, frames, channels = values.shape
    if bins != estimator.config.frequency_bin_count or channels != estimator.channel_count:
        raise ValueError('feature spectra and mask estimator dimensions must agree')
    masks = np.empty(values.shape)
    probabilities = np.empty((frames, channels))
    estimator.reset()
    for frame in range(frames+estimator.config.right_context_frames):
        current = values[:, frame] if frame < frames else np.zeros((bins, channels), complex)
        quantized = round_away_from_zero(current.real*2**21)+1j*round_away_from_zero(current.imag*2**21)
        result = estimator.process_frame(quantized)
        if 0 <= result.target_frame_index < frames:
            masks[:, result.target_frame_index] = result.gains
            probabilities[result.target_frame_index] = result.speech_probability
    return masks, probabilities


def render_farfield_spectra(spectrum, speech_masks, *, mode='beam_selection', selection_config=None,
                            speech_probability=None, frame_probability=None, bin_probability=None,
                            statistical_gain=None, spatial_config=None, ambient_shared_mask=True,
                            enable_spatial_filter=True):
    """Render aligned 512-sample spectra; probabilities are explicit external evidence."""
    values = spectral_array(spectrum)
    bins, frames, channels = values.shape
    if bins != 257 or mode not in ('beam_selection', 'spatial', 'fixed_direction', 'ambient'):
        raise ValueError('mode must be beam_selection, spatial, fixed_direction or ambient on 257 bins')
    masks = unit_interval_array(speech_masks, 'speech_masks', np.shape(speech_masks))
    if masks.ndim != 3 or masks.shape[:2] != values.shape[:2] or not masks.shape[2]:
        raise ValueError('speech masks must share the signal time and frequency grid')
    output = np.zeros((bins, frames, 1), complex)
    evidence = np.zeros((bins, frames, 1))
    selected = np.zeros((frames, 1), int)
    noise_levels = np.zeros(frames)
    if mode == 'beam_selection':
        if masks.shape != values.shape:
            raise ValueError('beam selection requires one mask per beam')
        config = selection_config or SpeechBeamConfig(channel_count=channels)
        if config.channel_count != channels:
            raise ValueError('selection configuration must match beam channels')
        selector = SpeechBeamSelector(config)
        selected = np.empty((frames, config.selected_count), int)
        if speech_probability is not None:
            speech_probability = unit_interval_array(speech_probability, 'speech_probability', (frames, channels))
        for frame in range(frames):
            result = selector.process_frame(values[:, frame], masks[:, frame],
                      speech_probability=None if speech_probability is None else speech_probability[frame])
            output[:, frame], evidence[:, frame] = result.spectrum, result.speech_gain
            selected[frame], noise_levels[frame] = result.selected_channels, result.noise_level_db
    elif mode == 'spatial':
        spatial = MaskGuidedBeamformer(spatial_config or MaskGuidedConfig(MvdrConfig(microphone_count=channels))) if enable_spatial_filter else None
        floor = np.r_[np.full(50, .1), np.linspace(.1, .5, 51), np.full(156, .5)][:, None]
        for frame in range(frames):
            gain = masks[:, frame].max(axis=1, keepdims=True)
            post_gain = np.maximum(floor, rescale_upper_range(gain, .2, .7))
            gain[gain < .2] = 0
            evidence[:, frame] = gain
            beam = values[:, frame, :1] if spatial is None else spatial.process_frame(values[:, frame]*32768, gain**2)/32768
            output[:, frame] = beam*post_gain
        output[:6] = 0
    elif mode == 'fixed_direction':
        if channels != 1:
            raise ValueError('fixed_direction requires a single selected beam')
        frame_probability = unit_interval_array(frame_probability, 'frame_probability', (frames,))
        bin_probability = unit_interval_array(bin_probability, 'bin_probability', (bins, frames))
        statistical_gain = unit_interval_array(statistical_gain, 'statistical_gain', (bins, frames))
        gains = rescale_upper_range(masks.max(axis=2), .4, .8)
        floor = np.r_[np.linspace(.3, .5, 102), np.full(155, .5)]
        evidence[:, :, 0] = np.minimum(1, gains**frame_probability[None, :]*statistical_gain**(1-frame_probability[None, :])
                                      *.01**((1-bin_probability)*(1-frame_probability[None, :])))
        output[:, :, 0] = values[:, :, 0]*np.minimum(1, np.maximum(gains, floor[:, None]))
        output[:3] = 0
    else:
        if channels < 2 or masks.shape != values.shape:
            raise ValueError('ambient mixing requires at least two channels and one mask per channel')
        gains = rescale_upper_range(masks, .4, .8)
        floor = np.r_[np.linspace(1, .5, 34), np.full(223, .5)]
        evidence[:, :, 0] = gains.max(axis=2)
        applied_gains = gains.max(axis=2, keepdims=True) if ambient_shared_mask else gains
        magnitude = np.mean(np.abs(values)*np.maximum(applied_gains, floor[:, None, None]), axis=2)
        phase_source = values[:, :, 1]
        phase = np.divide(phase_source, np.abs(phase_source), out=np.zeros_like(phase_source), where=np.abs(phase_source)>0)
        output[:, :, 0] = magnitude*phase
        output[:3] = 0
    return output, evidence, selected, noise_levels


def capture_farfield(audio, mask_estimator, *, mode='beam_selection', feature_audio=None,
                     digital_gain=1.0, gain_config=GainControlConfig(), selection_config=None,
                     frame_probability=None, bin_probability=None, statistical_gain=None,
                     spatial_config=None, synthesis_scale=1.0, enable_spatial_filter=True,
                     ambient_shared_mask=True):
    """Return sample-aligned audio and all stages; context latency is reported separately.

    feature_audio may contain separated sources on the same sample timeline.
    Gain control measures reconstructed, clipped audio while processing the
    enhanced spectra. Set gain_config=None to retain the pre-gain output.
    """
    samples = audio_array(audio)
    finite_scalar(digital_gain, 'digital_gain', minimum=0)
    finite_scalar(synthesis_scale, 'synthesis_scale', minimum=np.finfo(float).tiny)
    transform = ShortTimeFourierTransform(FourierConfig(512, 256))
    spectrum = transform.analyze(samples*digital_gain)
    if feature_audio is None:
        features = spectrum
    else:
        feature_audio = audio_array(feature_audio)
        if len(feature_audio) != len(samples):
            raise ValueError('feature_audio must share the input sample timeline')
        features = transform.analyze(feature_audio)
    masks, probabilities = estimate_aligned_masks(features, mask_estimator)
    enhanced, evidence, selected, noise_levels = render_farfield_spectra(spectrum, masks, mode=mode,
        selection_config=selection_config, speech_probability=probabilities, frame_probability=frame_probability,
        bin_probability=bin_probability, statistical_gain=statistical_gain, spatial_config=spatial_config,
        enable_spatial_filter=enable_spatial_filter, ambient_shared_mask=ambient_shared_mask)
    enhanced *= synthesis_scale
    controlled = enhanced.copy()
    gains_db = np.zeros(enhanced.shape[1])
    if gain_config is not None:
        if gain_config.frame_length_samples != 512 or gain_config.hop_length_samples != 256:
            raise ValueError('gain control must use the 512/256 frame grid')
        reconstructed = transform.synthesize(enhanced)
        if mode in ('beam_selection', 'spatial'):
            reconstructed = np.clip(reconstructed, -1, 32767/32768)
        reconstructed = np.pad(reconstructed, ((0, max(0, enhanced.shape[1]*256-len(reconstructed))), (0, 0)))
        controller = AutomaticGainControl(gain_config)
        for frame in range(enhanced.shape[1]):
            controlled[:, frame] = controller.process_frame(enhanced[:, frame], audio_hop=reconstructed[frame*256:(frame+1)*256], speech_gain=evidence[:, frame])
            gains_db[frame] = controller.gain_db
    output = np.clip(transform.synthesize(controlled, length_samples=len(samples)), -1, 32767/32768)
    return FarfieldCaptureResult(output, spectrum, features, masks, enhanced, controlled, evidence, gains_db,
                                 selected, noise_levels, mask_estimator.config.right_context_frames*256)
