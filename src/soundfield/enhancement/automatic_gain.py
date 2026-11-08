"""Speech-aware gain control for a single audio channel."""

from dataclasses import dataclass, field
import math
from numbers import Integral, Real

import numpy as np

from ..contracts import positive_integer, real_array, spectral_array


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _probability(value, name):
    _finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")


def _mean(values):
    return math.fsum(float(value) for value in values.flat) / values.size


@dataclass(frozen=True)
class GainCurveConfig:
    lower_level_db: float = -80.0
    upper_level_db: float = -6.0
    quiet_output_db: float = -80.0
    expansion_ratio: float = 0.5
    compression_ratio: float = 0.1
    target_offset_db: float = 0.0
    enable_speech_gate: bool = True
    speech_threshold: float = 0.35

    def __post_init__(self):
        for name in ("lower_level_db", "upper_level_db", "quiet_output_db", "expansion_ratio",
                     "compression_ratio", "target_offset_db"):
            _finite(getattr(self, name), name)
        if self.lower_level_db >= self.upper_level_db:
            raise ValueError("lower_level_db must be below upper_level_db")
        if self.expansion_ratio < 0 or self.compression_ratio < 0:
            raise ValueError("gain ratios must be nonnegative")
        if not isinstance(self.enable_speech_gate, bool):
            raise ValueError("enable_speech_gate must be boolean")
        _probability(self.speech_threshold, "speech_threshold")


@dataclass(frozen=True)
class GainSmoothingConfig:
    short_activity_frames: int = 3
    long_activity_frames: int = 5
    gain_history_frames: int = 50
    short_activity_threshold: float = 0.5
    long_activity_threshold: float = 0.2
    attack_memory: float = 0.1
    speech_memory: float = 0.9
    silence_memory: float = 1.0
    release_memory: float = 1.0
    reference_memory: float = 0.96
    initial_gain_db: float = 0.0

    def __post_init__(self):
        for name in ("short_activity_frames", "long_activity_frames", "gain_history_frames"):
            positive_integer(getattr(self, name), name)
        if self.short_activity_frames > self.long_activity_frames:
            raise ValueError("short activity history cannot exceed long activity history")
        for name in ("short_activity_threshold", "long_activity_threshold", "attack_memory",
                     "speech_memory", "silence_memory", "release_memory", "reference_memory"):
            _probability(getattr(self, name), name)
        _finite(self.initial_gain_db, "initial_gain_db")


@dataclass(frozen=True)
class GainControlConfig:
    frame_length_samples: int = 512
    hop_length_samples: int = 256
    speech_band_start: int = 15
    speech_band_stop: int = 64
    activity_mode: str = "mean"
    enable_reference_activity: bool = False
    reference_threshold: float = 0.1
    noise_gain_floor: float = 0.5
    enable_equalizer: bool = True
    equalizer_start_bin: int = 32
    equalizer_stop_bin: int = 66
    equalizer_low_gain: float = 0.5
    equalizer_high_gain: float = 1.0
    curve: GainCurveConfig = field(default_factory=GainCurveConfig)
    smoothing: GainSmoothingConfig = field(default_factory=GainSmoothingConfig)

    def __post_init__(self):
        positive_integer(self.frame_length_samples, "frame_length_samples")
        positive_integer(self.hop_length_samples, "hop_length_samples")
        if self.frame_length_samples != 2 * self.hop_length_samples:
            raise ValueError("frame_length_samples must equal twice hop_length_samples")
        if not isinstance(self.curve, GainCurveConfig) or not isinstance(self.smoothing, GainSmoothingConfig):
            raise TypeError("curve and smoothing must use their configuration types")
        if self.activity_mode not in ("mean", "mean_square", "weighted_mean", "disabled"):
            raise ValueError("unknown activity_mode")
        for name in ("enable_reference_activity", "enable_equalizer"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for start, stop in ((self.speech_band_start, self.speech_band_stop),
                            (self.equalizer_start_bin, self.equalizer_stop_bin)):
            if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (start, stop)):
                raise ValueError("frequency bin boundaries must be integers")
            if not 0 <= start < stop <= self.frequency_bin_count:
                raise ValueError("bin ranges must be nonempty and inside the spectrum")
        if self.equalizer_stop_bin - self.equalizer_start_bin < 2:
            raise ValueError("equalizer transition needs at least two bins")
        for name in ("reference_threshold", "noise_gain_floor"):
            _probability(getattr(self, name), name)
        for name in ("equalizer_low_gain", "equalizer_high_gain"):
            _finite(getattr(self, name), name)
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def frequency_bin_count(self):
        return self.frame_length_samples // 2 + 1


@dataclass
class _GainState:
    audio_history: np.ndarray
    activity_history: np.ndarray
    gain_history_db: np.ndarray
    smoothed_gain_db: float
    output_gain_db: float = 0.0
    gain_linear: float = 1.0
    activity: float = 0.0
    short_activity: float = 0.0
    long_activity: float = 0.0


class AutomaticGainControl:
    """Maintain independent gain history for one channel of normalized audio.

    Frequency ranges use zero-based start and exclusive stop indices.
    Each call accepts (bins, 1) spectra and an (hop_length_samples, 1) audio block.
    """

    def __init__(self, config: GainControlConfig | None = None):
        self._config = config if config is not None else GainControlConfig()
        if not isinstance(self.config, GainControlConfig):
            raise TypeError("config must be a GainControlConfig")
        self._window = np.sqrt(np.maximum(np.hanning(self.config.frame_length_samples), 0.0))
        start, stop = self.config.equalizer_start_bin, self.config.equalizer_stop_bin
        self._equalizer = np.concatenate((
            np.full(start, self.config.equalizer_low_gain),
            np.linspace(self.config.equalizer_low_gain, self.config.equalizer_high_gain, stop - start),
            np.full(self.config.frequency_bin_count - stop, self.config.equalizer_high_gain),
        ))
        self.reset()

    @property
    def config(self) -> GainControlConfig:
        return self._config

    def reset(self):
        config = self.config
        self._state = _GainState(
            np.zeros(config.frame_length_samples),
            np.zeros(config.smoothing.long_activity_frames),
            np.zeros(config.smoothing.gain_history_frames),
            config.smoothing.initial_gain_db,
        )

    @property
    def gain_linear(self) -> float:
        return self._state.gain_linear

    @property
    def gain_db(self) -> float:
        return self._state.output_gain_db

    def process_frame(self, spectrum, *, audio_hop, speech_gain, reference_activity=0.0):
        config = self.config
        frequencies = spectral_array(spectrum, dimensions=2)
        samples = real_array(audio_hop, "audio_hop", 2)
        gains = real_array(speech_gain, "speech_gain", 2)
        if frequencies.shape != (config.frequency_bin_count, 1) or gains.shape != frequencies.shape:
            raise ValueError("spectrum and speech_gain must have shape (bins, 1)")
        if samples.shape != (config.hop_length_samples, 1):
            raise ValueError("audio_hop must have shape (hop_length_samples, 1)")
        if np.any((gains < 0) | (gains > 1)):
            raise ValueError("speech_gain must be between zero and one")
        reference = np.asarray(reference_activity)
        if reference.ndim not in (0, 2) or (reference.ndim == 2 and reference.shape != samples.shape):
            raise ValueError("reference_activity must be a scalar or match audio_hop")
        reference = real_array(reference, "reference_activity", reference.ndim)
        if np.any((reference < 0) | (reference > 1)):
            raise ValueError("reference_activity must be between zero and one")

        state = self._state
        band = gains[config.speech_band_start:config.speech_band_stop, 0]
        if config.activity_mode == "disabled":
            activity = 1.0
        elif config.activity_mode == "mean_square":
            activity = _mean(band**2)
        else:
            activity = _mean(band)
            if config.activity_mode == "weighted_mean":
                activity *= float(gains[-1, 0])
        reference_active = config.enable_reference_activity and _mean(reference) > config.reference_threshold
        state.activity = activity
        state.audio_history[:-config.hop_length_samples] = state.audio_history[config.hop_length_samples:]
        state.audio_history[-config.hop_length_samples:] = samples[:, 0]
        level_db = float(20 * np.log10(np.max(np.abs(state.audio_history * self._window)) + 1e-10))
        state.activity_history[:-1] = state.activity_history[1:]
        state.activity_history[-1] = 0.0 if reference_active else activity
        state.short_activity = _mean(state.activity_history[-config.smoothing.short_activity_frames:])
        state.long_activity = _mean(state.activity_history)
        self._update_gain(level_db, reference_active)
        output = frequencies * (config.noise_gain_floor + (1 - config.noise_gain_floor) * gains) * state.gain_linear
        if config.enable_equalizer:
            output *= self._equalizer[:, None]
        return output

    def _update_gain(self, level_db, reference_active):
        state, curve, smoothing = self._state, self.config.curve, self.config.smoothing
        if reference_active or level_db < curve.lower_level_db:
            target_db = curve.quiet_output_db
        elif curve.enable_speech_gate and state.short_activity <= curve.speech_threshold:
            target_db = level_db
        elif level_db < curve.upper_level_db:
            target_db = curve.upper_level_db - (curve.upper_level_db - level_db) * curve.expansion_ratio + curve.target_offset_db
        else:
            target_db = curve.upper_level_db + (level_db - curve.upper_level_db) * curve.compression_ratio + curve.target_offset_db
        requested_gain_db = target_db - level_db
        smoothed = state.smoothed_gain_db
        if reference_active:
            if requested_gain_db < smoothed:
                smoothed = smoothing.reference_memory * smoothed + (1 - smoothing.reference_memory) * requested_gain_db
        elif state.long_activity >= smoothing.long_activity_threshold:
            memory = smoothing.speech_memory if state.short_activity >= smoothing.short_activity_threshold else smoothing.release_memory
            smoothed = memory * smoothed + (1 - memory) * requested_gain_db
        elif state.short_activity >= smoothing.short_activity_threshold:
            smoothed = smoothing.attack_memory * smoothed + (1 - smoothing.attack_memory) * requested_gain_db
        else:
            smoothed = smoothing.silence_memory * smoothed + (1 - smoothing.silence_memory) * float(np.min(state.gain_history_db))
        state.gain_history_db[:-1] = state.gain_history_db[1:]
        if smoothed + level_db > curve.upper_level_db:
            state.output_gain_db = curve.upper_level_db - level_db
            state.gain_history_db[-1] = 0.0
        else:
            state.output_gain_db = smoothed
            state.gain_history_db[-1] = smoothed * smoothing.silence_memory
        state.smoothed_gain_db = smoothed
        state.gain_linear = float(10 ** (state.output_gain_db / 20))
