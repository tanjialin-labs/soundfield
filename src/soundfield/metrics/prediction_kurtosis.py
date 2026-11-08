"""Speech-weighted kurtosis of linear prediction residuals."""

import numpy as np
from scipy.signal import lfilter

from ..contracts import audio_array, positive_integer, unit_interval_array


def prediction_coefficients(autocorrelation, order):
    """Solve a real autocorrelation system by Levinson recursion."""
    positive_integer(order, 'order')
    correlation = np.asarray(autocorrelation, dtype=float)
    if correlation.ndim != 1 or len(correlation) <= order or not np.isfinite(correlation).all():
        raise ValueError('autocorrelation must contain finite lags zero through order')
    coefficients = np.zeros(order)
    if correlation[0] <= 0:
        return np.r_[1.0, coefficients]
    coefficients[0] = correlation[1]/correlation[0]
    for degree in range(2, order+1):
        preceding = coefficients[:degree-1].copy()
        reverse_lags = correlation[1:degree][::-1]
        denominator = correlation[0]-reverse_lags @ preceding[::-1]
        if abs(denominator) <= np.finfo(float).eps*correlation[0]:
            break
        reflection = (correlation[degree]-reverse_lags @ preceding)/denominator
        coefficients[:degree-1] = preceding-reflection*preceding[::-1]
        coefficients[degree-1] = reflection
    return np.r_[1.0, -coefficients]


class PredictionResidualKurtosis:
    """Consume 160-sample hops and return a residual frame and weighted channel scores."""

    def __init__(self, channel_count=1, *, order=10, history_frames=200):
        positive_integer(channel_count, 'channel_count')
        positive_integer(order, 'order')
        positive_integer(history_frames, 'history_frames')
        if order >= 320:
            raise ValueError('order must be smaller than the analysis frame')
        self.channel_count, self.order, self.history_frames = channel_count, order, history_frames
        self.window = np.hanning(321)[:-1]
        self.reset()

    def reset(self):
        self.audio_history = np.zeros((320, self.channel_count))
        self.kurtosis_history = np.zeros((self.history_frames, self.channel_count))
        self.speech_history = np.zeros_like(self.kurtosis_history)
        self.frame_count = 0

    def process_block(self, audio_block, speech_probability):
        audio = audio_array(audio_block)
        probabilities = unit_interval_array(speech_probability, 'speech_probability', (self.channel_count,))
        if audio.shape != (160, self.channel_count):
            raise ValueError('audio_block must contain 160 samples per configured channel')
        self.audio_history[:160] = self.audio_history[160:]
        self.audio_history[160:] = audio
        windowed = self.audio_history*self.window[:, None]
        residual = np.zeros_like(windowed)
        scores = np.full(self.channel_count, 3.0)
        for channel in range(self.channel_count):
            values = windowed[:, channel]
            if np.mean(np.abs(values)) >= 1e-6:
                correlation = np.correlate(values, values, mode='full')[319:]
                coefficients = prediction_coefficients(correlation, self.order)
                residual[:, channel] = lfilter(coefficients, [1], values)
                centered = residual[:, channel]-residual[:, channel].mean()
                power = np.sum(centered**2)
                if power > 0:
                    scores[channel] = len(values)*np.sum(centered**4)/power**2
        self.kurtosis_history[:-1] = self.kurtosis_history[1:]
        self.speech_history[:-1] = self.speech_history[1:]
        self.kurtosis_history[-1], self.speech_history[-1] = scores, probabilities
        if self.frame_count == 0:
            self.kurtosis_history[:] = scores
            self.speech_history[:] = probabilities
        weights = self.speech_history.sum(axis=0)
        smoothed = np.divide(np.sum(self.kurtosis_history*self.speech_history, axis=0), weights,
                             out=self.kurtosis_history.mean(axis=0), where=weights>0)
        self.frame_count += 1
        return smoothed, residual
