"""Array shapes and numeric validation for audio processing."""

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def real_array(value, name: str, dimensions: int) -> RealArray:
    array = np.asarray(value)
    if np.iscomplexobj(array) or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    if array.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def audio_array(value) -> RealArray:
    """Validate audio with shape (samples, channels)."""
    array = real_array(value, "audio", 2)
    if array.shape[1] < 1:
        raise ValueError("audio must have at least one channel")
    return array


def spectral_array(value, dimensions: int = 3) -> ComplexArray:
    """Validate a spectrum with channels on the last axis."""
    array = np.asarray(value)
    if array.dtype.kind not in "fciu" or array.ndim != dimensions:
        raise ValueError(f"spectrum must be a {dimensions}-dimensional numeric array")
    array = np.asarray(array, dtype=np.complex128)
    if array.shape[-1] < 1 or not np.all(np.isfinite(array)):
        raise ValueError("spectrum must have channels and only finite values")
    return array


def finite_scalar(value, name: str, *, minimum=None, maximum=None) -> None:
    if (isinstance(value, (bool, str)) or not np.isscalar(value)
            or np.iscomplexobj(value) or not np.isfinite(value)):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def spectral_frame(value, bins: int, channels: int):
    array = spectral_array(value, 2)
    if array.shape != (bins, channels):
        raise ValueError(f"spectrum must have shape ({bins}, {channels})")
    return array


def unit_interval_array(value, name: str, shape: tuple):
    array = real_array(value, name, len(shape))
    if array.shape != shape or np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must have shape {shape} and values in [0, 1]")
    return array
