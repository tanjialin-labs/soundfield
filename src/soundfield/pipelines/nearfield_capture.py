"""Spatial classification as explicit evidence for adaptive source separation."""

from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from ..contracts import audio_array, finite_scalar
from ..localization import NearfieldSpatialClassifier, NearfieldClassificationConfig
from ..separation import OnlineSourceSeparator
from ..transform import FourierConfig, ShortTimeFourierTransform
from .array_enhancement import enhance_microphone_array


@dataclass(frozen=True)
class NearfieldCaptureResult:
    audio: np.ndarray
    classified_audio: np.ndarray
    enhanced_beam_audio: np.ndarray
    accepted_bin_counts: np.ndarray
    separation_evidence: np.ndarray
    nearfield_factors: np.ndarray
    separated_spectrum: np.ndarray


def capture_nearfield(audio, beam_bank, enhancement_config, separation_config, *,
                       classification_config=NearfieldClassificationConfig(), sigmoid_midpoint=45.,
                       sigmoid_slope=.5, channel_energy_weights=None):
    """Join short-frame classification and long-frame separation by sample position."""
    samples = audio_array(audio)
    if separation_config.nearfield_control is None:
        raise ValueError('separation requires nearfield_control')
    if samples.shape[1] != len(classification_config.microphone_positions_meters):
        raise ValueError('classification geometry must match audio channels')
    if len(classification_config.output_microphone_indices) != beam_bank.beam_count or separation_config.channel_count != beam_bank.beam_count:
        raise ValueError('classification, beam and source channel counts must agree')
    finite_scalar(sigmoid_midpoint, 'sigmoid_midpoint', minimum=0)
    finite_scalar(sigmoid_slope, 'sigmoid_slope', minimum=0)
    short_hop = classification_config.frame_length_samples//2
    long_hop = separation_config.frequency_bin_count-1
    if long_hop % short_hop:
        raise ValueError('separation hop must be an integer multiple of the classification hop')
    short_transform = ShortTimeFourierTransform(FourierConfig(2*short_hop, short_hop))
    long_transform = ShortTimeFourierTransform(FourierConfig(2*long_hop, long_hop))
    spectrum = short_transform.analyze(samples)
    frame_count = spectrum.shape[1]
    block = classification_config.block_frames
    padded_count = (frame_count+block-1)//block*block
    classified = np.zeros((short_hop+1, padded_count, beam_bank.beam_count), complex)
    counts = np.zeros(padded_count)
    classifier = NearfieldSpatialClassifier(classification_config)
    for frame in range(padded_count):
        result = classifier.estimate(spectrum[:, frame] if frame < frame_count else np.zeros((short_hop+1, samples.shape[1]), complex))
        if result is not None:
            classified[:, frame+1-block:frame+1] = result.spectrum
            counts[frame+1-block:frame+1] = result.accepted_bin_counts
    classified_audio = short_transform.synthesize(classified[:, :frame_count], length_samples=len(samples))
    beam_audio = enhance_microphone_array(samples, beam_bank, enhancement_config,
        transform_config=FourierConfig(2*(beam_bank.frequency_bin_count-1), beam_bank.frequency_bin_count-1))
    spatial = long_transform.analyze(classified_audio)
    beams = long_transform.analyze(beam_audio)
    separator = OnlineSourceSeparator(separation_config, channel_energy_weights=channel_energy_weights)
    weights = np.ones(beam_bank.beam_count) if channel_energy_weights is None else np.asarray(channel_energy_weights)
    output, evidence = np.zeros_like(beams), np.zeros(beams.shape[1])
    factors = np.zeros((beams.shape[1], beam_bank.beam_count))
    ratio = long_hop//short_hop
    counts = np.pad(counts[:frame_count], (0, max(0, beams.shape[1]*ratio-frame_count)))
    for frame in range(beams.shape[1]):
        current = float(np.mean(counts[frame*ratio:(frame+1)*ratio]))
        evidence[frame] = .5*(current+evidence[frame-1]) if frame and current < evidence[frame-1] else current
        probability = max(.01, expit(sigmoid_slope*(evidence[frame]-sigmoid_midpoint)))
        factors[frame] = 1-probability
        factors[frame, classification_config.focused_output_index] = probability
        power = np.mean(np.abs(spatial[separation_config.energy_start_bin:separation_config.energy_stop_bin, frame])**2, axis=0)*weights
        output[:, frame] = separator.process_frame(beams[:, frame], nearfield_power=power, nearfield_factors=factors[frame])
    return NearfieldCaptureResult(long_transform.synthesize(output, length_samples=len(samples)), classified_audio,
                                   beam_audio, counts[:frame_count], evidence, factors, output)
