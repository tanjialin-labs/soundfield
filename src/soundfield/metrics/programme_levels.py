"""Integrated programme loudness and oversampled peak levels for mono or stereo recordings."""

import numpy as np
from scipy.signal import lfilter, resample_poly

from ..contracts import audio_array, positive_integer


def interpolated_peak(audio, sample_rate_hz):
    """Return per-channel linear peaks with bandlimited interpolation to at least 192 kHz."""
    samples = audio_array(audio)
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    if not len(samples):
        return np.zeros(samples.shape[1])
    factor = max(1, int(np.ceil(192000/sample_rate_hz)))
    if factor == 1:
        return np.max(abs(samples), axis=0)
    peaks = np.max(abs(samples), axis=0)
    for start in range(0, len(samples), 32768):
        end = min(start+32768, len(samples))
        left, right = max(0, start-16), min(len(samples), end+16)
        high = resample_poly(samples[left:right], factor, 1, axis=0, window=('kaiser', 8.6))
        interior = high[(start-left)*factor:(end-left)*factor]
        peaks = np.maximum(peaks, np.max(abs(interior), axis=0))
    return peaks


def integrated_loudness(audio, sample_rate_hz):
    """Return gated LUFS or None when no complete block exceeds the absolute gate."""
    x = audio_array(audio)
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    if sample_rate_hz < 8000 or x.shape[1] not in (1, 2):
        raise ValueError('programme loudness requires mono or stereo at 8 kHz or above')
    width, hop = round(.4*sample_rate_hz), round(.1*sample_rate_hz)
    if len(x) < width:
        return None
    for f,q,shelf in ((1681.9744509555,.70717523696,True),(38.1354708761,.5003270373,False)):
        k = np.tan(np.pi*f/sample_rate_hz)
        denom = 1+k/q+k*k
        a = (1, 2*(k*k-1)/denom, (1-k/q+k*k)/denom)
        if shelf:
            vh = 10**(3.999843854/20)
            vb = vh**.499666774155
            b = np.array([vh+vb*k/q+k*k, 2*(k*k-vh), vh-vb*k/q+k*k])/denom
        else:
            b = [1., -2., 1.]
        x = lfilter(b, a, x, axis=0)
    cumulative = np.r_[0.,np.cumsum(np.sum(x*x, axis=1))]
    start = np.arange(0,len(x)-width+1,hop)
    energies = (cumulative[start+width]-cumulative[start])/width
    active = energies[energies > 10**((-70+.691)/10)]
    if not len(active):
        return None
    selected = active[active > .1*active.mean()]
    return float(-.691+10*np.log10(selected.mean()))
