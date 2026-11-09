"""Linked-channel soft-knee compression with sample-continuous gain smoothing."""

from dataclasses import dataclass

import numpy as np

from ..contracts import audio_array, finite_scalar, positive_integer


@dataclass(frozen=True)
class DynamicRangeConfig:
    sample_rate_hz: int = 16000
    threshold_db: float = -20
    compression_ratio: float = 4
    knee_width_db: float = 6
    makeup_gain_db: float = 0
    attack_seconds: float = .01
    release_seconds: float = .1
    peak_ceiling: float = 1

    def __post_init__(self):
        positive_integer(self.sample_rate_hz, 'sample_rate_hz')
        for name in ('threshold_db', 'makeup_gain_db'):
            finite_scalar(getattr(self, name), name)
        finite_scalar(self.compression_ratio, 'compression_ratio', minimum=1)
        finite_scalar(self.knee_width_db, 'knee_width_db', minimum=0)
        for name in ('attack_seconds', 'release_seconds', 'peak_ceiling'):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).eps)


class DynamicRangeController:
    """Input levels and peak ceiling use full-scale linear amplitude."""

    def __init__(self, config=DynamicRangeConfig()):
        self.config = config
        self.attack = np.exp(-1/(config.sample_rate_hz*config.attack_seconds))
        self.release = np.exp(-1/(config.sample_rate_hz*config.release_seconds))
        self.reset()

    def reset(self):
        self.gain_db = 0.0

    def process_block(self, audio):
        samples = audio_array(audio)
        if not len(samples):
            return samples.copy()
        config = self.config
        peaks = np.max(np.abs(samples), axis=1)
        levels = 20*np.log10(np.maximum(peaks, np.finfo(float).tiny))
        delta = levels-config.threshold_db
        knee = config.knee_width_db
        reduction = (1/config.compression_ratio-1)*np.maximum(delta, 0)
        if knee:
            selected = np.abs(delta) <= knee/2
            reduction[selected] = (1/config.compression_ratio-1)*(delta[selected]+knee/2)**2/(2*knee)
        targets = reduction+config.makeup_gain_db
        ceiling_db = 20*np.log10(config.peak_ceiling)
        limits = ceiling_db - levels
        smoothed = self.gain_db
        gains_db = np.empty(len(samples))
        for index, target in enumerate(targets):
            memory = self.attack if target < smoothed else self.release
            smoothed = min(memory*smoothed+(1-memory)*target, limits[index])
            gains_db[index] = smoothed
        self.gain_db = smoothed
        return samples * np.power(10., gains_db / 20)[:, None]
