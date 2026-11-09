"""Windowed minima and speech-dependent noise power tracking."""

from dataclasses import dataclass

import numpy as np

from ..contracts import finite_scalar, nonnegative_integer, positive_integer


@dataclass(frozen=True)
class NoiseTrackingConfig:
    power_memory: float = .6
    presence_memory: float = .1
    noise_memory: float = .8
    detection_ratio: float = 5.0
    history_windows: int = 8
    window_frames: int = 15
    initial_frame_offset: int = 2
    initial_window_index: int = 0
    initialize_from_observation: bool = False

    def __post_init__(self):
        if not isinstance(self.initialize_from_observation, bool):
            raise ValueError("initialize_from_observation must be boolean")
        for name in ("power_memory", "presence_memory", "noise_memory"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        finite_scalar(self.detection_ratio, "detection_ratio", minimum=0)
        positive_integer(self.history_windows, "history_windows")
        positive_integer(self.window_frames, "window_frames")
        nonnegative_integer(self.initial_frame_offset, "initial_frame_offset")
        nonnegative_integer(self.initial_window_index, "initial_window_index")
        if self.initial_frame_offset >= self.window_frames or self.initial_window_index >= self.history_windows:
            raise ValueError("initial offsets must fit their tracking windows")


class _MinimumStatistics:
    def __init__(self, shape, config):
        self.config = config
        self.power = np.zeros(shape)
        self.minimum = np.zeros(shape)
        self.temporary = np.zeros(shape)
        self.presence = np.zeros(shape)
        self.noise = np.zeros(shape)
        self.history = np.zeros((*shape, config.history_windows))
        self.frame_offset = config.initial_frame_offset
        self.window_index = config.initial_window_index

    def update(self, instantaneous_power):
        config = self.config
        if config.initialize_from_observation:
            starting = (self.power == 0) & (instantaneous_power > 0)
            self.power[starting] = instantaneous_power[starting]
            self.minimum[starting] = instantaneous_power[starting]
            self.temporary[starting] = instantaneous_power[starting]
            self.history[starting] = instantaneous_power[starting, None]
            self.noise[starting] = instantaneous_power[starting]
        self.power = config.power_memory * self.power + (1 - config.power_memory) * instantaneous_power
        self.minimum = np.minimum(self.minimum, self.power)
        self.temporary = np.minimum(self.temporary, self.power)
        # Comparing cross-products also defines the zero-power case without 0/0.
        detected = self.power > config.detection_ratio * self.minimum
        self.frame_offset += 1
        if self.frame_offset == config.window_frames:
            self.history[..., self.window_index] = self.temporary
            self.window_index = (self.window_index + 1) % config.history_windows
            self.minimum = np.min(self.history, axis=-1)
            self.temporary = self.power.copy()
            self.frame_offset = 0
        self.presence = config.presence_memory * self.presence + (1 - config.presence_memory) * detected
        memory = config.noise_memory + (1 - config.noise_memory) * self.presence
        self.noise = memory * self.noise + (1 - memory) * instantaneous_power


def _posterior_ratio(power, noise):
    ratio = np.divide(power, noise, out=np.full_like(power, 40.0), where=noise > 0)
    return np.minimum(ratio, 40)


def _speech_probability(absence, prior, exponent):
    present = 1 - absence
    denominator = present + absence * (1 + prior) * np.exp(-exponent)
    return np.divide(present, denominator, out=present.copy(), where=denominator > 0)
