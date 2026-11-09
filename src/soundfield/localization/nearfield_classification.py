"""Fresnel-subspace classification for symmetric linear microphone arrays."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, real_array, spectral_frame


def _distance_grid():
    return tuple(np.unique(np.round(np.concatenate((np.arange(.4, .701, .05), np.arange(.7, 1.001, .05), np.arange(1, 2.001, .2), [2., 5.])), 12)))


@dataclass(frozen=True)
class NearfieldClassificationConfig:
    microphone_positions_meters: tuple[float, ...] = (-.16, -.13, -.08, -.01, .01, .08, .13, .16)
    output_microphone_indices: tuple[int, ...] = (0, 2, 4, 5, 7)
    focused_output_index: int = 2
    frame_length_samples: int = 512
    sample_rate_hz: int = 16000
    block_frames: int = 8
    start_bin: int = 29
    stop_bin: int = 150
    sound_speed_meters_per_second: float = 340.0
    scan_angles_degrees: tuple[float, ...] = (-90, -60, -50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90)
    scan_distances_meters: tuple[float, ...] = _distance_grid()
    minimum_angle_degrees: float = -30
    maximum_angle_degrees: float = 30
    maximum_distance_meters: float = .86
    focus_gain: float = .98

    def __post_init__(self):
        for name in ("frame_length_samples", "sample_rate_hz", "block_frames", "stop_bin"):
            positive_integer(getattr(self, name), name)
        nonnegative_integer(self.start_bin, "start_bin")
        if self.frame_length_samples % 2 or not self.start_bin < self.stop_bin <= self.frequency_bin_count:
            raise ValueError("frame length must be even and frequency interval valid")
        positions = real_array(self.microphone_positions_meters, "microphone_positions_meters", 1)
        if len(positions) < 2 or len(positions) % 2 or not np.allclose(positions, -positions[::-1], atol=1e-12):
            raise ValueError("microphones must form symmetric pairs about the origin")
        if np.any(np.diff(positions) <= 0):
            raise ValueError("microphone positions must be strictly increasing")
        object.__setattr__(self, "microphone_positions_meters", tuple(positions))
        indices = tuple(self.output_microphone_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("output_microphone_indices must be nonempty and distinct")
        for index in indices:
            nonnegative_integer(index, "output_microphone_indices")
            if index >= len(positions):
                raise ValueError("output microphone is outside the array")
        object.__setattr__(self, "output_microphone_indices", indices)
        nonnegative_integer(self.focused_output_index, "focused_output_index")
        if self.focused_output_index >= len(indices):
            raise ValueError("focused_output_index is outside the output channels")
        for name in ("scan_angles_degrees", "scan_distances_meters"):
            grid = real_array(getattr(self, name), name, 1)
            if not len(grid) or np.any(np.diff(grid) <= 0):
                raise ValueError(f"{name} must be nonempty and strictly increasing")
            object.__setattr__(self, name, tuple(grid))
        if min(self.scan_distances_meters) <= 0:
            raise ValueError("scan distances must be positive")
        for name in ("sound_speed_meters_per_second", "maximum_distance_meters"):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).tiny)
        for name in ("minimum_angle_degrees", "maximum_angle_degrees"):
            finite_scalar(getattr(self, name), name)
        if self.minimum_angle_degrees > self.maximum_angle_degrees:
            raise ValueError("classification angle interval is reversed")
        finite_scalar(self.focus_gain, "focus_gain", minimum=0, maximum=1)

    @property
    def frequency_bin_count(self):
        return self.frame_length_samples // 2 + 1


@dataclass(frozen=True)
class NearfieldClassification:
    spectrum: np.ndarray
    accepted_bin_counts: np.ndarray
    angles_degrees: np.ndarray
    distances_meters: np.ndarray


class NearfieldSpatialClassifier:
    def __init__(self, config: NearfieldClassificationConfig | None = None, *, steering_frequencies_hz=None):
        self.config = config if config is not None else NearfieldClassificationConfig()
        config = self.config
        frequency = (np.arange(config.start_bin, config.stop_bin) * config.sample_rate_hz / config.frame_length_samples
                     if steering_frequencies_hz is None else real_array(steering_frequencies_hz, "steering_frequencies_hz", 1))
        if frequency.shape != (config.stop_bin - config.start_bin,) or np.any(frequency < 0):
            raise ValueError("steering_frequencies_hz must match the analysis interval")
        positions = np.asarray(config.microphone_positions_meters)
        angles = np.deg2rad(config.scan_angles_degrees)
        distances = np.asarray(config.scan_distances_meters)
        first_half = positions[:len(positions) // 2] ** 2
        squared_pairs = np.concatenate((first_half, first_half[::-1]))
        linear = -2 * np.pi * positions[:, None] * np.sin(angles)[None]
        curvature = np.pi * squared_pairs[:, None, None] * np.cos(angles)[None, :, None] ** 2 / distances
        phase = (linear[:, :, None] + curvature)[None] * frequency[:, None, None, None] / config.sound_speed_meters_per_second
        self._responses = np.exp(-1j * phase)
        self.reset()

    @property
    def steering_responses(self):
        return self._responses.copy()

    @property
    def frame_index(self):
        return self._frame_index

    def reset(self):
        config = self.config
        self._history = np.zeros((config.frequency_bin_count, config.block_frames, len(config.microphone_positions_meters)), complex)
        self._frame_index = 0

    def estimate(self, spectrum_frame):
        config = self.config
        spectrum = spectral_frame(spectrum_frame, config.frequency_bin_count, len(config.microphone_positions_meters))
        self._history[:, :-1] = self._history[:, 1:]
        self._history[:, -1] = spectrum
        self._frame_index += 1
        if self._frame_index % config.block_frames:
            return None
        output = self._history[:, :, config.output_microphone_indices].copy()
        angle_results = np.full(config.stop_bin - config.start_bin, np.nan)
        distance_results = np.full_like(angle_results, np.nan)
        accepted = 0
        channels = len(config.microphone_positions_meters)
        for offset, index in enumerate(range(config.start_bin, config.stop_bin)):
            samples = self._history[index].T
            if not np.any(samples):
                continue
            covariance = samples @ samples.conj().T / config.block_frames
            diagonal = np.abs(np.diag(covariance))
            covariance[np.diag_indices(channels)] = diagonal + 1e-5 * np.max(diagonal)
            vectors = svd(covariance, lapack_driver="gesvd")[0][:, 1:]
            projector = vectors @ vectors.conj().T
            responses = self._responses[offset].reshape(channels, -1, order="F")
            denominator = np.sum(responses.conj() * (projector @ responses), axis=0) + 1e-8
            score = 1 / denominator
            maximum = int(np.argmax(np.abs(score)))
            angle_index, distance_index = np.unravel_index(maximum, self._responses.shape[2:], order="F")
            angle, distance = config.scan_angles_degrees[angle_index], config.scan_distances_meters[distance_index]
            angle_results[offset], distance_results[offset] = angle, distance
            inside = config.minimum_angle_degrees <= angle <= config.maximum_angle_degrees and distance <= config.maximum_distance_meters
            gains = np.full(output.shape[-1], 1 - config.focus_gain if inside else config.focus_gain)
            gains[config.focused_output_index] = config.focus_gain if inside else 1 - config.focus_gain
            output[index] *= gains
            accepted += int(inside)
        return NearfieldClassification(output, np.full(config.block_frames, accepted, dtype=np.int64), angle_results, distance_results)
