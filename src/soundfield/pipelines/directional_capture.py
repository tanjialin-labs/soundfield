"""Combine spatial matching, speech persistence and externally directed gating."""

from dataclasses import dataclass
import numpy as np
from ..contracts import audio_array, spectral_array, unit_interval_array
from ..localization import DirectionalMatchingEstimator, SpeechGatedDirectionTracker
from ..localization.speech_direction import SpeechDirectionConfig
from ..enhancement.spatial_activity import SpatialActivityConfig, SpatialActivityGate
from ..transform import FourierConfig, ShortTimeFourierTransform
from .beam_separation import separate_beams


@dataclass(frozen=True)
class DirectionTrackingResult:
    angles_degrees: np.ndarray
    direction_indices: np.ndarray
    direction_scores: np.ndarray
    accepted_bin_counts: np.ndarray


def localize_speech(spectrum, speech_gains, selected_channels, matching_config, *,
                    tracking_config=SpeechDirectionConfig(), search_regions=None, speech_bins=(9, 200)):
    """Estimate a direction for each frame using sample-aligned speech gains."""
    values = spectral_array(spectrum)
    bins, frames, channels = values.shape
    gains = unit_interval_array(speech_gains, 'speech_gains', (bins, frames))
    selected = np.asarray(selected_channels)
    directions = len(matching_config.scan_angles_degrees)
    if (matching_config.block_frames != 1 or matching_config.event_delay_frames != 0
            or matching_config.frequency_bin_count != bins or matching_config.geometry.microphone_count != channels):
        raise ValueError('matching configuration must describe a single aligned frame of the input array')
    regions = tuple(search_regions) if search_regions is not None else ((0, directions),)*len(matching_config.beam_intervals_degrees)
    if selected.shape != (frames,) or selected.dtype.kind not in 'iu' or np.any(selected<0) or np.any(selected>=len(regions)):
        raise ValueError('selected_channels must contain one valid region index per frame')
    if not 0 <= speech_bins[0] < speech_bins[1] <= bins:
        raise ValueError('speech_bins must fit the spectrum')
    matcher = DirectionalMatchingEstimator(matching_config)
    tracker = SpeechGatedDirectionTracker(tracking_config, search_regions=regions, direction_count=directions,
                                          circular_search=matching_config.circular_search)
    indices, angles = np.full(frames, -1, int), np.full(frames, np.nan)
    scores, counts = np.zeros((directions, frames)), np.zeros(frames, int)
    for frame in range(frames):
        gain = gains[:, frame].copy()
        gain[gain < .2] = 0
        match = matcher.estimate(values[:, frame]*gain[:, None], beam_activity=np.ones(len(matching_config.beam_intervals_degrees)))
        scores[:, frame] = match.direction_scores
        counts[frame] = np.count_nonzero(match.accepted_bins)
        index = tracker.process_frame(scores[:, frame], float(gains[slice(*speech_bins), frame].sum()), int(selected[frame]))
        if index is not None:
            indices[frame], angles[frame] = index, matching_config.scan_angles_degrees[index]
    return DirectionTrackingResult(angles, indices, scores, counts)


@dataclass(frozen=True)
class DirectedCaptureResult:
    audio: np.ndarray
    beam_spectrum: np.ndarray
    separated_spectrum: np.ndarray
    localization_spectrum: np.ndarray
    direction_scores: np.ndarray
    spatial_evidence: np.ndarray
    activity_gains: np.ndarray
    output_spectrum: np.ndarray
    gate_delay_samples: int


def capture_directed_region(audio, beam_bank, enhancement_config, separation_config, matching_config, *,
                             target_source_index=1, target_direction_indices=None, half_width_steps=2,
                             activity_config=SpatialActivityConfig()):
    """Use a target direction timeline to gate a separated source with spatial evidence."""
    samples = audio_array(audio)
    if matching_config.block_frames != 1 or matching_config.event_delay_frames != 0:
        raise ValueError('direction matching must use one frame and no event delay')
    if not isinstance(half_width_steps, int) or half_width_steps < 0 or not 0 <= target_source_index < beam_bank.beam_count:
        raise ValueError('target source and angular half-width must be valid')
    stages = separate_beams(samples, beam_bank, enhancement_config, separation_config)
    if matching_config.frequency_bin_count != beam_bank.frequency_bin_count or matching_config.geometry.microphone_count != samples.shape[1]:
        raise ValueError('spatial matching must share the beam analysis grid and microphones')
    bins, frames = stages.input_spectrum.shape[:2]
    targets = np.asarray(target_direction_indices)
    directions = len(matching_config.scan_angles_degrees)
    if targets.shape != (frames,) or targets.dtype.kind not in 'iu' or np.any(targets<0) or np.any(targets>=directions):
        raise ValueError('target_direction_indices must contain one valid scan index per frame')
    localization = stages.input_spectrum*stages.separation_gains[:, :, target_source_index, None]
    transform = ShortTimeFourierTransform(FourierConfig(2*(bins-1), bins-1))
    localization = transform.analyze(transform.synthesize(localization, length_samples=len(samples)))
    target_audio = transform.synthesize(stages.separated_spectrum[:, :, target_source_index:target_source_index+1], length_samples=len(samples))
    target_spectrum = transform.analyze(target_audio)
    matcher, gate = DirectionalMatchingEstimator(matching_config), SpatialActivityGate(activity_config)
    evidence, gains = np.zeros(frames), np.zeros(frames)
    scores = np.zeros((directions, frames))
    delay = activity_config.vote_history_frames-1
    for frame in range(frames+delay):
        if frame < frames:
            match = matcher.estimate(localization[:, frame], beam_activity=np.ones(len(matching_config.beam_intervals_degrees)))
            scores[:, frame] = match.direction_scores
            target_index = int(targets[frame])
            offsets = np.arange(directions) - target_index
            distances = np.abs(offsets)
            if matching_config.circular_search:
                distances = np.minimum(distances, directions - distances)
            region = distances <= half_width_steps
            evidence[frame] = np.count_nonzero(np.sum(np.abs(match.bin_coefficients[:, :, region]), axis=(1, 2)))
        gain = gate.process_frame(float(evidence[frame]) if frame < frames else 0.)
        target = max(frame-delay, 0)
        if target < frames:
            gains[target] = gain
    output = target_spectrum*gains[None, :, None]
    return DirectedCaptureResult(transform.synthesize(output, length_samples=len(samples)), stages.enhanced_beam_spectrum,
        stages.separated_spectrum, localization, scores, evidence, gains, output, delay*(bins-1))
