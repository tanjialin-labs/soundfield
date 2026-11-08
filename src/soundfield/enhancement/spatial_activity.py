"""Temporal voting over external spatial activity evidence."""

from dataclasses import dataclass

import numpy as np

from ..contracts import finite_scalar, positive_integer


@dataclass(frozen=True)
class SpatialActivityConfig:
    accumulator_limit: float = 100
    decay_per_frame: float = 10
    activation_threshold: float = 50
    initial_accumulator: float = 0
    vote_history_frames: int = 5
    vote_threshold: float = 0.4

    def __post_init__(self):
        positive_integer(self.vote_history_frames, 'vote_history_frames')
        for name in ('accumulator_limit', 'decay_per_frame', 'activation_threshold',
                     'initial_accumulator', 'vote_threshold'):
            finite_scalar(getattr(self, name), name, minimum=0)


class SpatialActivityGate:
    def __init__(self, config=SpatialActivityConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.accumulator = self.config.initial_accumulator
        self.vote_history = np.zeros(self.config.vote_history_frames)

    def process_frame(self, spatial_evidence):
        finite_scalar(spatial_evidence, 'spatial_evidence', minimum=0)
        self.accumulator = min(max(self.accumulator + spatial_evidence
                                   - self.config.decay_per_frame, 0), self.config.accumulator_limit)
        self.vote_history[:-1] = self.vote_history[1:]
        self.vote_history[-1] = self.accumulator > self.config.activation_threshold
        return float(self.vote_history.sum() > self.config.vote_threshold)
