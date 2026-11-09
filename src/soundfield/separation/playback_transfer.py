"""Estimate a shared playback transfer model for a whole microphone recording."""
import numpy as np
from ..transform import FourierConfig, ShortTimeFourierTransform


def subtract_recorded_playback(microphones, playback):
    transform = ShortTimeFourierTransform(FourierConfig(512, 256))
    microphones_spectrum = transform.analyze(microphones)
    reference = transform.analyze(playback)
    bins, frames, channels = reference.shape
    taps = min(16, max(1, frames // (4*channels)))
    history = np.lib.stride_tricks.sliding_window_view(
        np.pad(reference, ((0, 0), (taps-1, 0), (0, 0))), taps, axis=1)
    echo = np.empty_like(microphones_spectrum)
    # Limit working storage while fitting independent frequency bins.
    width = max(1, 65536 // max(1, frames*channels*taps))
    identity = np.eye(channels*taps)
    for start in range(0, bins, width):
        band = slice(start, start+width)
        design = history[band].reshape(-1, frames, channels*taps)
        target = microphones_spectrum[band]
        weights = np.ones(target.shape[:2])
        adjoint = design.conj().swapaxes(1, 2)
        for iteration in range(2):
            covariance = adjoint @ (design*weights[:,:,None])
            loading = np.maximum(.002*covariance.diagonal(axis1=1, axis2=2).real.mean(axis=1), 1e-9)
            cross = adjoint @ (target*weights[:,:,None])
            coefficients = np.linalg.solve(covariance + loading[:,None,None]*identity, cross)
            estimate = design @ coefficients
            if iteration == 0:
                power = np.mean(abs(target-estimate)**2,axis=2)
                weights = 1 / np.maximum(power, .05*power.mean(axis=1,keepdims=True) + 1e-9)
        echo[band] = estimate
    return microphones - transform.synthesize(echo, length_samples=len(microphones))
