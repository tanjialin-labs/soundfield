"""Piecewise numerical approximations used in spectral gain estimation."""

import numpy as np

from ..contracts import finite_scalar, real_array


def _values(value):
    return real_array(value, "values", np.ndim(value))


def suppress_below_threshold(values, threshold):
    """Set values strictly below threshold to zero, preserving the input shape."""
    finite_scalar(threshold, "threshold")
    array = _values(values)
    return np.where(array < threshold, 0.0, array)


def rescale_upper_range(values, lower_bound, upper_bound):
    """Preserve values below the lower bound; map the remaining interval to one."""
    finite_scalar(lower_bound, "lower_bound")
    finite_scalar(upper_bound, "upper_bound")
    if upper_bound <= lower_bound:
        raise ValueError("upper_bound must exceed lower_bound")
    array = _values(values)
    return np.where(array < lower_bound, array, np.where(array > upper_bound, 1.0,
                    (1 - lower_bound) / (upper_bound - lower_bound) * (array - lower_bound) + lower_bound))


def spectral_amplitude_gain(values):
    """Tabulated nonnegative spectral amplitude estimator with a rational tail."""
    array = _values(values)
    if np.any(array < 0):
        raise ValueError("gain arguments must be nonnegative")
    table = np.array([.82157, 1.02017, 1.20461, 1.37534, 1.53363, 1.68092,
                      1.81865, 1.94811, 2.07038, 2.18638, 2.29688, 2.40255,
                      2.50391, 2.60144, 2.69551, 2.78647, 2.87458, 2.96015,
                      3.04333, 3.12431, 3.20326])
    flat = array.reshape(-1)
    result = np.empty_like(flat)
    tabulated = flat < 9.5
    indices = np.floor(2 * flat[tabulated]).astype(int)
    fraction = 2 * flat[tabulated] - indices
    result[tabulated] = ((1 - fraction) * table[indices] + fraction * table[indices + 1]) / np.sqrt(flat[tabulated] + .0001)
    result[~tabulated] = 1 + .1296 / flat[~tabulated]
    return result.reshape(array.shape)


def exponential_integral_approximation(values):
    """Approximate E1 for nonnegative real arguments; E1(0) is positive infinity."""
    array = _values(values)
    if np.any(array < 0):
        raise ValueError("exponential integral arguments must be nonnegative")
    flat = array.reshape(-1)
    result = np.empty_like(flat)
    small = flat < 1
    x = flat[small]
    with np.errstate(divide="ignore"):
        result[small] = -np.log(x) - .57721566 + x * (.99999193 + x * (-.24991055 + x * (.05519968 + x * (-.00976004 + x * .00107857))))
    x = flat[~small]
    inverse = 1 / x
    numerator = inverse * (1 + inverse * (8.5733287401 + inverse * (18.0590169730 + inverse * (8.6347608925 + inverse * .2677737343))))
    denominator = 1 + inverse * (9.5733223454 + inverse * (25.6329561486 + inverse * (21.0996530827 + inverse * 3.9584969228)))
    result[~small] = numerator / denominator * np.exp(-x)
    return result.reshape(array.shape)


def channel_phase(spectrum, preferred_channel):
    """Return the preferred channel's unit phase, using a live channel at spectral zeros."""
    reference = spectrum[..., preferred_channel]
    missing = reference == 0
    if np.any(missing):
        reference = reference.copy()
        available = spectrum[missing]
        strongest = np.argmax(np.abs(available), axis=-1)
        reference[missing] = available[np.arange(len(strongest)), strongest]
    magnitude = np.abs(reference)
    return np.divide(reference, magnitude, out=np.zeros_like(reference), where=magnitude > 0)
