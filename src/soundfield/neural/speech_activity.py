"""Mel-energy speech activity estimation with independent channel histories."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit

from ..contracts import positive_integer, real_array, audio_array


@dataclass(frozen=True)
class SpeechActivityModel:
    hidden_weights: tuple
    hidden_biases: tuple
    output_weights: np.ndarray
    output_bias: np.ndarray
    feature_offset: np.ndarray
    feature_scale: np.ndarray

    def __post_init__(self):
        if len(self.hidden_weights) != len(self.hidden_biases):
            raise ValueError('hidden weights and biases must have equal lengths')
        width = 112
        for weights, bias in zip((*self.hidden_weights, self.output_weights),
                                 (*self.hidden_biases, self.output_bias)):
            weights = real_array(weights, 'weights', 2)
            bias = real_array(bias, 'bias', 1)
            if weights.shape[1] != width or bias.shape != (weights.shape[0],):
                raise ValueError('speech model layer dimensions do not agree')
            width = weights.shape[0]
        if width != 2:
            raise ValueError('speech model must output two class probabilities')
        for value in (self.feature_offset, self.feature_scale):
            if real_array(value, 'normalization', 1).shape != (112,):
                raise ValueError('speech model normalization must contain 112 features')

    @classmethod
    def load(cls, path):
        """Load a numeric .npz resource without executing serialized objects."""
        with np.load(Path(path), allow_pickle=False) as archive:
            count = int(archive['hidden_layer_count'])
            return cls(tuple(archive[f'hidden_weights_{i}'].copy() for i in range(count)),
                       tuple(archive[f'hidden_bias_{i}'].copy() for i in range(count)),
                       archive['output_weights'].copy(), archive['output_bias'].copy(),
                       archive['feature_offset'].copy(), archive['feature_scale'].copy())

    def save(self, path):
        values = dict(hidden_layer_count=len(self.hidden_weights), output_weights=self.output_weights,
                      output_bias=self.output_bias, feature_offset=self.feature_offset,
                      feature_scale=self.feature_scale)
        for index, (weights, bias) in enumerate(zip(self.hidden_weights, self.hidden_biases)):
            values[f'hidden_weights_{index}'], values[f'hidden_bias_{index}'] = weights, bias
        np.savez_compressed(path, **values)


class SpeechActivityDetector:
    """Consume 160 normalized samples per channel at 16 kHz; context delay is 480 samples."""

    sample_rate_hz = 16000
    hop_length_samples = 160
    latency_samples = 480

    def __init__(self, model: SpeechActivityModel, channel_count=1):
        positive_integer(channel_count, 'channel_count')
        self.model, self.channel_count = model, channel_count
        self.window = np.hamming(320)
        frequencies = np.linspace(0, 8000, 257)
        centers = 700 * np.expm1(np.linspace(0, np.log1p(8000 / 700), 18))
        self.filterbank = np.maximum(0, np.minimum(
            (frequencies[None, :] - centers[:-2, None]) / np.diff(centers)[:-1, None],
            (centers[2:, None] - frequencies[None, :]) / np.diff(centers)[1:, None]))
        self.reset()

    def reset(self):
        self.audio_history = np.zeros((320, self.channel_count))
        self.feature_history = np.zeros((112, self.channel_count))
        self.probabilities = np.zeros(self.channel_count)
        self.layer_outputs = ()

    def process_block(self, audio_block):
        audio = audio_array(audio_block)
        if audio.shape != (160, self.channel_count):
            raise ValueError(f'audio_block must have shape (160, {self.channel_count})')
        self.audio_history[:160] = self.audio_history[160:]
        self.audio_history[160:] = audio
        centered = self.audio_history * 32768
        centered -= centered.mean(axis=0)
        emphasized = centered.copy()
        emphasized[0] *= 0.03
        emphasized[1:] -= 0.97 * centered[:-1]
        spectrum = np.fft.rfft(emphasized * self.window[:, None], n=512, axis=0)
        features = np.log(self.filterbank @ np.abs(spectrum)**2 + 1e-10)
        self.feature_history[:-16] = self.feature_history[16:]
        self.feature_history[-16:] = features
        values = ((self.feature_history + self.model.feature_offset[:, None])
                  * self.model.feature_scale[:, None])
        outputs = []
        for weights, bias in zip(self.model.hidden_weights, self.model.hidden_biases):
            values = expit(weights @ values + bias[:, None])
            outputs.append(values.copy())
        logits = self.model.output_weights @ values + self.model.output_bias[:, None]
        probabilities = np.exp(logits - np.max(logits, axis=0))
        probabilities /= probabilities.sum(axis=0)
        self.probabilities = probabilities[1]
        self.layer_outputs = (*outputs, probabilities.copy())
        return self.probabilities.copy()


class SpeechActivityGate:
    """Per-channel hysteresis with bounded hangover, measured in probability frames."""

    def __init__(self, channel_count=1, *, onset_probability=.55, release_probability=.35, hold_frames=3):
        from ..contracts import finite_scalar, nonnegative_integer
        positive_integer(channel_count, 'channel_count')
        finite_scalar(onset_probability, 'onset_probability', minimum=0, maximum=1)
        finite_scalar(release_probability, 'release_probability', minimum=0, maximum=onset_probability)
        nonnegative_integer(hold_frames, 'hold_frames')
        self.onset = onset_probability
        self.release = release_probability
        self.hold = hold_frames
        self._remaining = np.zeros(channel_count, int)
        self._active = np.zeros(channel_count, bool)

    def reset(self):
        self._active.fill(False)
        self._remaining.fill(0)

    def process_frame(self, probability):
        from ..contracts import unit_interval_array
        values = unit_interval_array(probability, 'probability', self._active.shape)
        present = values >= np.where(self._active, self.release, self.onset)
        self._active = present | (self._remaining > 0)
        self._remaining = np.where(present, self.hold, np.maximum(self._remaining-1, 0))
        return self._active.copy()
