"""Quantized triangular-band features and temporal normalization."""

from dataclasses import dataclass

import numpy as np

from ..contracts import positive_integer, nonnegative_integer, real_array


def round_away_from_zero(values):
    values = np.asarray(values, dtype=np.float64)
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


@dataclass(frozen=True)
class BandFeatureConfig:
    frequency_bin_count: int = 257
    band_count: int = 64
    left_context_frames: int = 2
    right_context_frames: int = 2
    mean_window_frames: int = 39
    enable_mean_removal: bool = True

    def __post_init__(self):
        for name in ('frequency_bin_count', 'band_count', 'mean_window_frames'):
            positive_integer(getattr(self, name), name)
        for name in ('left_context_frames', 'right_context_frames'):
            nonnegative_integer(getattr(self, name), name)
        if self.band_count < 2 or self.band_edges[-1] > self.frequency_bin_count:
            raise ValueError('band edges must fit inside the spectrum')

    @property
    def band_edges(self):
        return round_away_from_zero(650 * np.sinh(np.arange(self.band_count) / 20)
                                    * self.frequency_bin_count / 8000).astype(int)

    @property
    def feature_count(self):
        return self.band_count * (1 + self.left_context_frames + self.right_context_frames)


def triangular_band_energy(spectrum, band_edges, *, other_spectrum=None, quantized=True):
    """Accumulate cross energy; quantized inputs use the Q21 spectrum scale."""
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    other = spectrum if other_spectrum is None else np.asarray(other_spectrum, dtype=np.complex128)
    edges = np.asarray(band_edges)
    if (spectrum.ndim != 1 or other.shape != spectrum.shape or edges.ndim != 1
            or len(edges) < 2 or np.any(edges != np.floor(edges))
            or edges[0] < 0 or edges[-1] > len(spectrum) or np.any(np.diff(edges) < 0)
            or not np.isfinite(spectrum).all() or not np.isfinite(other).all()):
        raise ValueError('spectra and band edges have incompatible values or shapes')
    energy = np.zeros(len(edges))
    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        start, stop = int(start), int(stop)
        width = stop - start
        if not width:
            continue
        fraction = np.arange(width) / width
        cross_real = other[start:stop].real * spectrum[start:stop].real
        cross_imag = other[start:stop].imag * spectrum[start:stop].imag
        if quantized:
            fraction = round_away_from_zero(fraction * 32768)
            cross = round_away_from_zero(cross_real / 32768) + round_away_from_zero(cross_imag / 32768)
            energy[index] += np.sum(round_away_from_zero((32768 - fraction) * cross / 32768))
            energy[index + 1] += np.sum(round_away_from_zero(fraction * cross / 32768))
        else:
            cross = cross_real + cross_imag
            energy[index] += np.sum((1 - fraction) * cross)
            energy[index + 1] += np.sum(fraction * cross)
    return round_away_from_zero(energy / 1024) if quantized else energy


def interpolate_band_gains(band_gains, band_edges, frequency_bin_count):
    gains = real_array(band_gains, 'band_gains', 1)
    edges = np.asarray(band_edges)
    if (edges.shape != gains.shape or len(edges) < 2 or np.any(edges != np.floor(edges))
            or edges[0] < 0 or edges[-1] > frequency_bin_count or np.any(np.diff(edges) < 0)):
        raise ValueError('band gains and edges must agree and fit inside the spectrum')
    output = np.zeros(frequency_bin_count)
    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        start, stop = int(start), int(stop)
        if stop > start:
            fraction = np.arange(stop - start) / (stop - start)
            output[start:stop] = (1 - fraction) * gains[index] + fraction * gains[index + 1]
    return output


class TemporalFeatureNormalizer:
    def __init__(self, config: BandFeatureConfig, offset, scale):
        self.config = config
        self.offset = round_away_from_zero(real_array(offset, 'offset', 1) * 2**24)
        self.scale = round_away_from_zero(real_array(scale, 'scale', 1) * 2**24)
        if self.offset.shape != (config.feature_count,) or self.scale.shape != self.offset.shape:
            raise ValueError('normalization arrays must match the context feature count')
        self.reset()

    def reset(self):
        self.context = np.zeros(self.config.feature_count)
        self.mean_history = np.repeat(-self.offset[:self.config.band_count, None],
                                      self.config.mean_window_frames, axis=1)

    def process_frame(self, quantized_features):
        features = real_array(quantized_features, 'quantized_features', 1).copy()
        if features.shape != (self.config.band_count,):
            raise ValueError('features must contain one value per band')
        if self.config.enable_mean_removal:
            self.mean_history[:, :-1] = self.mean_history[:, 1:]
            self.mean_history[:, -1] = features
            features -= np.mean(self.mean_history, axis=1)
        count = self.config.band_count
        self.context[:-count] = self.context[count:]
        self.context[-count:] = features
        return (self.context + self.offset) * self.scale / 2**24
