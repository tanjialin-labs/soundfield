"""Energy-gated pairwise correlation for selecting a microphone region."""

from dataclasses import dataclass

import numpy as np

from ..contracts import positive_integer, nonnegative_integer, finite_scalar, spectral_array


@dataclass(frozen=True)
class RegionalCorrelationConfig:
    channel_count: int = 4
    frequency_bin_count: int = 513
    sample_rate_hz: int = 16000
    energy_bins: tuple = (10, 200)
    rear_energy_stop_bin: int = 200
    steering_bins: tuple = (20, 120)
    correlation_bins: tuple = (10, 200)
    excluded_lag_samples: int = 2
    scan_extent_meters: float = 0.2
    scan_step_meters: float = 0.005
    dominant_distance_meters: float = 0.2
    energy_ratios: tuple = (2.0, 2.0)
    front_energy_ratios: tuple = (2.0, 2.0)
    rear_energy_ratios: tuple = (2.0, 2.0)
    lag_limits_samples: tuple = (20, 20)

    def __post_init__(self):
        positive_integer(self.channel_count, 'channel_count')
        positive_integer(self.frequency_bin_count, 'frequency_bin_count')
        positive_integer(self.sample_rate_hz, 'sample_rate_hz')
        for band in (self.energy_bins, self.steering_bins, self.correlation_bins):
            if len(band) != 2:
                raise ValueError("frequency intervals must contain two indices")
            nonnegative_integer(band[0], 'start bin')
            positive_integer(band[1], 'stop bin')
            if not 0 <= band[0] < band[1] <= self.frequency_bin_count:
                raise ValueError('frequency intervals must fit inside the spectrum')
        positive_integer(self.rear_energy_stop_bin, 'rear_energy_stop_bin')
        if not self.energy_bins[0] < self.rear_energy_stop_bin <= self.frequency_bin_count:
            raise ValueError('rear energy interval is invalid')
        for name in ('scan_extent_meters', 'scan_step_meters', 'dominant_distance_meters'):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).eps)
        nonnegative_integer(self.excluded_lag_samples, 'excluded_lag_samples')
        if not 0 <= self.excluded_lag_samples < self.frequency_bin_count-1:
            raise ValueError('excluded lag interval is invalid')
        for name in ('energy_ratios', 'front_energy_ratios', 'rear_energy_ratios', 'lag_limits_samples'):
            pair = tuple(getattr(self, name))
            if len(pair) != 2:
                raise ValueError(f"{name} must contain two values")
            for value in pair:
                finite_scalar(value, name, minimum=np.finfo(float).tiny)
            object.__setattr__(self, name, pair)



class RegionalCorrelationEstimator:
    def __init__(self, config=RegionalCorrelationConfig()):
        self.config = config
        distances = np.arange(-config.scan_extent_meters,
                              config.scan_extent_meters+config.scan_step_meters/2,
                              config.scan_step_meters)
        start, stop = config.steering_bins
        angular_frequency = 2*np.pi*np.arange(start, stop)/(config.frequency_bin_count-1)*(config.sample_rate_hz/2)
        self.steering_weights = np.exp(-1j*angular_frequency[:, None]*distances[None, :]/340)
        length = 2 * (config.frequency_bin_count - 1)
        lags = np.arange(length)
        lags[lags > length // 2] -= length
        valid = ((np.abs(lags) >= config.excluded_lag_samples) &
                 (lags >= -config.lag_limits_samples[0]) & (lags <= config.lag_limits_samples[1]))
        self._lag_indices = np.flatnonzero(valid)
        self._lags = lags[valid]
        self.reset()

    def reset(self):
        self.pair_scores = np.zeros((self.config.channel_count, self.config.channel_count))

    def estimate(self, spectrum, enhanced_spectrum):
        raw, enhanced = spectral_array(spectrum), spectral_array(enhanced_spectrum)
        config = self.config
        if (raw.shape != enhanced.shape or raw.shape[0] != config.frequency_bin_count
                or raw.shape[2] != config.channel_count or raw.shape[1] == 0):
            raise ValueError('spectra must have matching bins, nonempty frames and configured channels')
        scores = np.zeros_like(self.pair_scores)
        for first in range(config.channel_count-1):
            for second in range(first+1, config.channel_count):
                if not np.any(raw[:, :, first]) and not np.any(raw[:, :, second]):
                    continue
                denominator = np.mean(np.abs(raw[:, :, first])*np.abs(raw[:, :, second]), axis=1)
                correlation = np.divide(np.mean(raw[:, :, first]*raw[:, :, second].conj(), axis=1),
                                        denominator, out=np.zeros_like(denominator, dtype=complex), where=denominator>0)
                start, stop = config.energy_bins
                if first == 2:
                    stop = config.rear_energy_stop_bin
                energy = np.sum(np.abs(enhanced[start:stop])**2, axis=(0, 1))
                ratios = (config.front_energy_ratios if (first, second) == (0, 1) else
                          config.rear_energy_ratios if (first, second) == (2, 3) else config.energy_ratios)
                floor = config.dominant_distance_meters/config.scan_step_meters
                if energy[first] > ratios[0]*energy[second]:
                    score = floor
                elif ratios[1]*energy[first] < energy[second]:
                    score = -floor
                elif (first, second) == (0, 1):
                    start, stop = config.steering_bins
                    response = np.mean((correlation[start:stop, None]*self.steering_weights).real, axis=0)
                    score = int(np.argmax(response))-(len(response)-1)/2 if np.ptp(response) > 0 else 0.
                else:
                    start, stop = config.correlation_bins
                    selected = np.zeros(config.frequency_bin_count, complex)
                    cross = correlation[start:stop]
                    magnitude = np.abs(cross)
                    np.divide(cross, magnitude, out=selected[start:stop], where=magnitude > 0)
                    response = np.fft.irfft(selected)[self._lag_indices]
                    score = 0.0
                    if response.size and response.max() > 0:
                        peaks = self._lags[np.isclose(response, response.max(), rtol=1e-12, atol=0)]
                        if len(peaks) == 1:
                            score = -peaks[0] * 340 / config.sample_rate_hz / config.scan_step_meters
                scores[first, second], scores[second, first] = score, -score
        self.pair_scores = scores
        totals = scores.sum(axis=1)
        region = int(np.argmax(totals)) if np.ptp(totals) > 0 else None
        return scores.copy(), region
