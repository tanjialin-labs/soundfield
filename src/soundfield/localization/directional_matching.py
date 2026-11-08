"""Directional dictionary matching with bin screening and event constraints."""

from dataclasses import dataclass, field

import numpy as np

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, real_array, spectral_frame
from .geometry import ArrayGeometry, linear_array, plane_wave_responses


@dataclass(frozen=True)
class DirectionalMatchingConfig:
    geometry: ArrayGeometry = field(default_factory=linear_array)
    frame_length_samples: int = 512
    sample_rate_hz: int = 16000
    block_frames: int = 40
    event_delay_frames: int = 10
    analysis_band: tuple[int, int] = (48, 151)
    scan_angles_degrees: tuple[float, ...] = tuple(range(0, 181, 5))
    beam_intervals_degrees: tuple[tuple[float, float], ...] = ((0, 70), (60, 120), (110, 180))
    polar_angle_radians: float = np.pi / 2
    propagation_sign: int = -1
    source_count: int = 1
    periodic_estimation: bool = False
    circular_search: bool = False
    search_all_directions: bool = False
    microphone_imbalance_db: float = 5.0
    amplitude_ratio_bounds: tuple[float, float] | None = (.3, 3.0)
    phase_screening: bool = True
    use_frame_energy: bool = True
    coherence_weight_floor: float | None = .2
    smoothing_half_width_steps: int = 1
    center_beam_constraint: bool = True
    exclude_endfire_peaks: bool = False

    def __post_init__(self):
        if not isinstance(self.geometry, ArrayGeometry):
            raise TypeError("geometry must be an ArrayGeometry")
        for name in ("frame_length_samples", "sample_rate_hz", "block_frames", "source_count"):
            positive_integer(getattr(self, name), name)
        for name in ("event_delay_frames", "smoothing_half_width_steps"):
            nonnegative_integer(getattr(self, name), name)
        if self.frame_length_samples % 2:
            raise ValueError("frame_length_samples must be even")
        if self.source_count >= self.geometry.microphone_count:
            raise ValueError("source_count must be smaller than microphone count")
        if len(self.analysis_band) != 2:
            raise ValueError("analysis_band must contain start and stop")
        nonnegative_integer(self.analysis_band[0], "analysis_band")
        positive_integer(self.analysis_band[1], "analysis_band")
        if not self.analysis_band[0] < self.analysis_band[1] <= self.frequency_bin_count:
            raise ValueError("invalid analysis_band")
        object.__setattr__(self, "analysis_band", tuple(self.analysis_band))
        angles = real_array(self.scan_angles_degrees, "scan_angles_degrees", 1)
        if len(angles) < 3 or np.any(np.diff(angles) <= 0) or self.smoothing_half_width_steps >= len(angles):
            raise ValueError("scan angles must increase and contain the smoothing window")
        object.__setattr__(self, "scan_angles_degrees", tuple(angles))
        intervals = real_array(self.beam_intervals_degrees, "beam_intervals_degrees", 2)
        if intervals.shape[1] != 2 or not len(intervals) or not np.all(np.isin(intervals, angles)):
            raise ValueError("each beam interval endpoint must appear in scan_angles_degrees")
        object.__setattr__(self, "beam_intervals_degrees", tuple(map(tuple, intervals)))
        if self.center_beam_constraint and not (90 in angles and np.any((intervals[:, 0] < 90) & (intervals[:, 1] > 90))):
            raise ValueError("center_beam_constraint requires an interval spanning 90 degrees")
        if self.propagation_sign not in (-1, 1):
            raise ValueError("propagation_sign must be -1 or 1")
        finite_scalar(self.polar_angle_radians, "polar_angle_radians")
        finite_scalar(self.microphone_imbalance_db, "microphone_imbalance_db", minimum=0)
        if self.amplitude_ratio_bounds is not None:
            bounds = tuple(self.amplitude_ratio_bounds)
            if len(bounds) != 2:
                raise ValueError("amplitude_ratio_bounds must contain lower and upper bounds")
            for value in bounds:
                finite_scalar(value, "amplitude_ratio_bounds", minimum=np.finfo(float).tiny)
            if bounds[0] > bounds[1]:
                raise ValueError("amplitude_ratio_bounds are reversed")
            object.__setattr__(self, "amplitude_ratio_bounds", bounds)
        if self.coherence_weight_floor is not None:
            finite_scalar(self.coherence_weight_floor, "coherence_weight_floor", minimum=np.finfo(float).tiny, maximum=1)

    @property
    def frequency_bin_count(self):
        return self.frame_length_samples // 2 + 1


@dataclass(frozen=True)
class DirectionalMatch:
    azimuth_degrees: float
    direction_scores: np.ndarray
    bin_coefficients: np.ndarray
    accepted_bins: np.ndarray


class DirectionalMatchingEstimator:
    def __init__(self, config: DirectionalMatchingConfig | None = None, *, steering_frequencies_hz=None):
        self.config = config if config is not None else DirectionalMatchingConfig()
        config = self.config
        frequency = (np.arange(*config.analysis_band) * config.sample_rate_hz / config.frame_length_samples
                     if steering_frequencies_hz is None else real_array(steering_frequencies_hz, "steering_frequencies_hz", 1))
        if frequency.shape != (config.analysis_band[1] - config.analysis_band[0],):
            raise ValueError("steering_frequencies_hz must match analysis_band")
        self._dictionary = plane_wave_responses(config.geometry, frequency, config.scan_angles_degrees,
                                                polar_angle_radians=config.polar_angle_radians,
                                                propagation_sign=config.propagation_sign) / np.sqrt(config.geometry.microphone_count)
        self.reset()

    @property
    def steering_dictionary(self):
        return self._dictionary.copy()

    @property
    def frame_index(self):
        return self._frame_index

    @property
    def estimation_count(self):
        return self._estimation_count

    def reset(self):
        config = self.config
        self._history = np.zeros((config.frequency_bin_count, config.block_frames + config.event_delay_frames,
                                  config.geometry.microphone_count), complex)
        self._frame_index = 0
        self._estimation_count = 0

    def estimate(self, spectrum_frame, *, beam_activity=None):
        config = self.config
        spectrum = spectral_frame(spectrum_frame, config.frequency_bin_count, config.geometry.microphone_count)
        activity = None
        if beam_activity is not None:
            activity = real_array(beam_activity, "beam_activity", 1)
            if activity.shape != (len(config.beam_intervals_degrees),) or np.any((activity != 0) & (activity != 1)):
                raise ValueError("beam_activity must contain one zero or one per beam")
        self._history[:, :-1] = self._history[:, 1:]
        self._history[:, -1] = spectrum
        self._frame_index += 1
        if config.periodic_estimation:
            if self._frame_index % config.block_frames:
                return None
            activity = None
        elif activity is None:
            return None
        self._estimation_count += 1
        block = self._history[:, :config.block_frames]
        energy = 10 * np.log10(np.maximum(np.mean(np.abs(block) ** 2, axis=(0, 1)), .0001))
        if activity is None or np.sum(activity) == 0 or config.search_all_directions or np.ptp(energy) > config.microphone_imbalance_db:
            activity = np.ones(len(config.beam_intervals_degrees))
        return self._estimate_block(block, activity)

    def _estimate_block(self, block, activity):
        config = self.config
        values = block[slice(*config.analysis_band)]
        mask = np.ones(values.shape[:2], bool)
        amplitude = np.abs(values)
        if config.amplitude_ratio_bounds is not None:
            low, high = config.amplitude_ratio_bounds
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = amplitude[:, :, :, None] / amplitude[:, :, None, :]
            mask &= (np.min(ratios, axis=(2, 3)) >= low) & (np.max(ratios, axis=(2, 3)) <= high)
        if config.phase_screening:
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized = values / amplitude
                phase = np.real(normalized[:, :, 1:] / normalized[:, :, :1])
            bounds = np.real(self._dictionary[:, 1:]) * np.sqrt(config.geometry.microphone_count)
            mask &= np.all((phase >= np.min(bounds, axis=2)[:, None]) & (phase <= np.max(bounds, axis=2)[:, None]), axis=2)
        coefficients = np.zeros((len(values), config.block_frames, len(config.scan_angles_degrees)), complex)
        for frequency in range(len(values)):
            valid = np.flatnonzero(mask[frequency])
            if not len(valid):
                continue
            dictionary = self._dictionary[frequency]
            observation = values[frequency, valid]
            residual = observation.copy()
            projection = observation @ dictionary.conj()
            weights = np.zeros_like(projection)
            rows = np.arange(len(valid))
            for _ in range(config.source_count):
                correlation = residual @ dictionary.conj()
                chosen = np.argmax(np.abs(correlation), axis=1)
                weights[rows, chosen] = projection[rows, chosen]
                residual = observation - weights @ dictionary.T
            if config.coherence_weight_floor is not None:
                denominator = np.abs(observation[:, 0])*np.abs(observation[:, 1])
                coherence = np.divide(np.abs(np.real(observation[:, 0]*observation[:, 1].conj())),
                                      denominator, out=np.zeros_like(denominator), where=denominator > 0)
                weights /= np.maximum(coherence, config.coherence_weight_floor)[:, None]
            coefficients[frequency, valid] = weights
        frames = np.sum(np.abs(coefficients), axis=0)
        if config.use_frame_energy:
            frames *= np.mean(np.abs(block[:, :, 0]) ** 2, axis=0)[:, None]
        scores = np.sum(frames, axis=0)
        half = config.smoothing_half_width_steps
        if half:
            kernel = .5 ** np.abs(np.arange(-half, half + 1))
            extended = np.concatenate((scores[-half:], scores)) if config.circular_search else scores.copy()
            smoothed = extended.copy()
            for index in range(half, len(extended) - half - 1):
                smoothed[index] = kernel @ extended[index - half:index + half + 1] / np.sum(kernel)
            scores = smoothed[half:] if config.circular_search else smoothed
        allowed = np.zeros(len(scores), bool)
        angles = np.asarray(config.scan_angles_degrees)
        for enabled, (start, stop) in zip(activity, config.beam_intervals_degrees):
            if enabled:
                allowed |= ((angles >= start) & (angles <= stop)) if start <= stop else ((angles >= start) | (angles <= stop))
        if config.center_beam_constraint:
            middle = [index for index, (start, stop) in enumerate(config.beam_intervals_degrees) if start < 90 < stop][-1]
            if activity[middle]:
                start, stop = config.beam_intervals_degrees[middle]
                left, right = np.count_nonzero(allowed[angles < start]), np.count_nonzero(allowed[angles > stop])
                if right == 0 and left != 1:
                    allowed[angles > 90] = False
                elif left == 0 and right != 1:
                    allowed[angles < 90] = False
        scores *= allowed
        index = int(np.argmax(scores))
        if config.exclude_endfire_peaks:
            index = int(np.clip(index, 1, len(scores) - 2))
        return DirectionalMatch(float(angles[index]), scores, coefficients, mask)
