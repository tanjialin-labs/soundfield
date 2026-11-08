"""Impulse-response and normalized correlation delay estimators."""

from dataclasses import dataclass

import numpy as np

from ..contracts import positive_integer, nonnegative_integer, real_array, finite_scalar


def estimate_filter_delay(adaptive_weights, hop_length_samples, reference_count=1, *, constrained=False):
    """Return the peak position in samples, using a one-sample origin."""
    positive_integer(hop_length_samples, 'hop_length_samples')
    positive_integer(reference_count, 'reference_count')
    weights = np.asarray(adaptive_weights, dtype=complex)
    if (weights.ndim != 2 or weights.shape[0] != hop_length_samples+1 or not weights.shape[1]
            or weights.shape[1] % reference_count or not np.isfinite(weights).all()):
        raise ValueError('adaptive weights must contain complete reference partitions')
    impulse = np.fft.irfft(weights[:, ::reference_count], n=2*hop_length_samples, axis=0)
    if constrained:
        response = impulse[:hop_length_samples].ravel(order='F')
    elif impulse.shape[1] == 1:
        response = impulse[:, 0]
    else:
        response = (impulse[:hop_length_samples, :-1]+impulse[hop_length_samples:, 1:]).ravel(order='F')
    return int(np.argmax(response))+1


@dataclass(frozen=True)
class CorrelationDelayConfig:
    hop_length_samples: int = 256
    correlation_hops: int = 8
    initial_delay_samples: int = 0
    peak_threshold: float = 0.1
    relative_mean_threshold: float = 0.2

    def __post_init__(self):
        positive_integer(self.hop_length_samples, 'hop_length_samples')
        positive_integer(self.correlation_hops, 'correlation_hops')
        nonnegative_integer(self.initial_delay_samples, 'initial_delay_samples')
        finite_scalar(self.peak_threshold, 'peak_threshold', minimum=0, maximum=1)
        finite_scalar(self.relative_mean_threshold, 'relative_mean_threshold', minimum=0, maximum=1)


class CorrelationDelayEstimator:
    def __init__(self, config=CorrelationDelayConfig()):
        self.config = config
        self.reset()

    def reset(self):
        count = self.config.hop_length_samples*self.config.correlation_hops
        self.microphone_history = np.zeros(count)
        self.reference_history = np.zeros(count+self.config.initial_delay_samples)
        self.delay_samples = 0
        self.peak_correlation = 0.0

    def process_block(self, microphone_audio, reference_audio):
        microphone = real_array(microphone_audio, 'microphone_audio', 1)
        reference = real_array(reference_audio, 'reference_audio', 1)
        hop = self.config.hop_length_samples
        if microphone.shape != (hop,) or reference.shape != (hop,):
            raise ValueError('delay estimation requires one hop from a microphone and reference')
        for history, values in ((self.microphone_history, microphone), (self.reference_history, reference)):
            history[:-hop], history[-hop:] = history[hop:].copy(), values
        count = len(self.microphone_history)
        selected_reference = self.reference_history[:count]
        scale = np.linalg.norm(self.microphone_history)*np.linalg.norm(self.reference_history)
        if scale == 0:
            self.peak_correlation = 0.0
            return self.delay_samples
        cross = np.fft.rfft(self.microphone_history, n=2*count)*np.fft.rfft(selected_reference, n=2*count).conj()
        correlation = np.abs(np.fft.fftshift(np.fft.irfft(cross, n=2*count)))/scale
        peak = int(np.argmax(correlation))
        self.peak_correlation = float(correlation[peak])
        relative_mean = correlation.mean()/self.peak_correlation if self.peak_correlation > 0 else 1
        if relative_mean < self.config.relative_mean_threshold and self.peak_correlation > self.config.peak_threshold:
            self.delay_samples = peak-count+1
        return self.delay_samples
