"""Convolution and sample alignment for recorded room responses."""

import numpy as np
from scipy.signal import fftconvolve
from ..contracts import audio_array, finite_scalar, nonnegative_integer


def convolve_room_response(audio, impulse_response, *, response_gain=1., keep_tail=False):
    """Convolve corresponding channels, broadcasting a mono source or response."""
    samples, response = audio_array(audio), audio_array(impulse_response)
    finite_scalar(response_gain, 'response_gain')
    if not len(response) or not len(samples):
        raise ValueError('source and response must contain samples')
    count = max(samples.shape[1], response.shape[1])
    if samples.shape[1] not in (1, count) or response.shape[1] not in (1, count):
        raise ValueError('source and response channels must agree or be mono')
    output = np.stack([fftconvolve(samples[:, min(channel, samples.shape[1]-1)], response[:, min(channel, response.shape[1]-1)]*response_gain)
                       for channel in range(count)], axis=1)
    return output if keep_tail else output[:len(samples)]


def align_delayed_recordings(reference, reverberant, enhanced, delay_samples):
    """Trim a declared processing delay without optimizing waveform correlation."""
    nonnegative_integer(delay_samples, 'delay_samples')
    reference, reverberant, enhanced = map(audio_array, (reference, reverberant, enhanced))
    count = min(len(reference), len(reverberant), len(enhanced)-delay_samples)
    if count <= 0:
        raise ValueError('declared delay leaves no overlapping samples')
    return reference[:count].copy(), reverberant[:count].copy(), enhanced[delay_samples:delay_samples+count].copy()
