"""Whole-recording two-source demixing using weighted spatial covariances."""
import numpy as np


def _blocks(bins, frames):
    width = max(1, min(bins, 16384 // frames))
    for start in range(0, bins, width):
        yield slice(start, min(start + width, bins))


def _update(covariance):
    first, second = covariance[..., 0], covariance[..., 1]
    whitening = np.linalg.inv(np.linalg.cholesky(second))
    adjoint = whitening.conj().swapaxes(1, 2)
    values, vectors = np.linalg.eigh(whitening @ first @ adjoint)
    coefficients = adjoint @ vectors
    coefficients[:, :, 0] /= np.sqrt(np.maximum(values[:, :1], np.finfo(float).eps))
    return coefficients


def render_stationary_pair(spectrum, iterations, reference_channel):
    """Fit a stationary demixer, then replace the owned input cube with source images."""
    bins, frames, channels = spectrum.shape
    if not frames:
        return
    blocks = list(_blocks(bins, frames))
    coefficients = np.broadcast_to(np.eye(channels), (bins, channels, channels)).astype(complex)
    for _ in range(iterations):
        power = np.zeros((frames, channels))
        for band in blocks:
            estimates = np.einsum('fms,ftm->fts', coefficients[band].conj(), spectrum[band])
            power += np.sum(np.abs(estimates)**2, axis=0)
        power /= bins
        precision = 1 / np.maximum(power, 1e-7*power.max(axis=0) + 1e-16)
        for band in blocks:
            block = spectrum[band]
            outer = block[:, :, :, None] * block[:, :, None, :].conj()
            covariance = np.einsum('ftmn,ts->fmns', outer, precision) / frames
            loading = 1e-7*(1 + covariance[:, 0, 0].real + covariance[:, 1, 1].real)
            covariance[:, 0, 0] += loading
            covariance[:, 1, 1] += loading
            coefficients[band] = _update(covariance)
    for band in blocks:
        demixer = coefficients[band].conj().swapaxes(1, 2)
        estimates = np.einsum('fsm,ftm->fts', demixer, spectrum[band])
        mixing = np.linalg.inv(demixer)[:, reference_channel]
        images = estimates * mixing[:, None, :]
        power = np.abs(images)**2
        total = power.sum(axis=2, keepdims=True)
        mask = np.divide(power, total, out=np.full_like(power, .5), where=total > 0)
        reference = spectrum[band, :, reference_channel, None]
        spectrum[band] = .5*(images + reference*mask)
