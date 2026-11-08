"""Channelwise fidelity measures on an explicitly aligned sample grid."""

from dataclasses import dataclass

import numpy as np

from ..contracts import audio_array


@dataclass(frozen=True)
class WaveformComparison:
    """One value per channel; level difference is processed minus reference."""

    root_mean_square_error: np.ndarray
    signal_to_error_db: np.ndarray
    scale_invariant_ratio_db: np.ndarray
    level_difference_db: np.ndarray


def compare_waveforms(reference, processed, *, remove_mean=False):
    """Compare equal (samples, channels) arrays without alignment or cropping.

    Optional mean removal applies to every measure. SI-SDR projects processed
    audio onto the reference; signal-to-error ratio retains gain errors.
    No energy floor is applied: a positive energy divided by zero gives +inf,
    zero divided by a positive energy gives -inf, and undefined ratios give
    NaN. A silent reference has undefined reference-based dB measures.
    """
    reference, processed = audio_array(reference), audio_array(processed)
    if reference.shape != processed.shape or not len(reference):
        raise ValueError('signals must have equal nonempty sample and channel axes')
    if not isinstance(remove_mean, bool):
        raise ValueError('remove_mean must be boolean')
    scale = np.maximum(np.max(np.abs(reference), axis=0), np.max(np.abs(processed), axis=0))
    scale = np.where(scale == 0, 1.0, scale)
    clean, altered = reference/scale, processed/scale
    if remove_mean:
        clean = clean-clean.mean(axis=0)
        altered = altered-altered.mean(axis=0)
    clean_energy = np.sum(clean*clean, axis=0)
    altered_energy = np.sum(altered*altered, axis=0)
    error_energy = np.sum((clean-altered)**2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        projection = clean*(np.sum(clean*altered, axis=0)/clean_energy)
        projected_energy = np.sum(projection*projection, axis=0)
        residual_energy = np.sum((altered-projection)**2, axis=0)
        signal_ratio = 10*(np.log10(clean_energy)-np.log10(error_energy))
        invariant_ratio = 10*(np.log10(projected_energy)-np.log10(residual_energy))
        level_difference = 10*(np.log10(altered_energy)-np.log10(clean_energy))
    signal_ratio[clean_energy == 0] = np.nan
    level_difference[clean_energy == 0] = np.nan
    return WaveformComparison(np.sqrt(error_energy/len(clean))*scale,
                              signal_ratio, invariant_ratio, level_difference)
