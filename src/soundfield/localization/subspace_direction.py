"""Event-conditioned broadband noise-subspace direction estimation."""

from dataclasses import dataclass, field

import numpy as np

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, real_array, spectral_frame, spectral_array
from .geometry import ArrayGeometry, linear_array, plane_wave_responses


@dataclass(frozen=True)
class NoiseSubspaceConfig:
    microphone_history_frames: int = 61
    event_frames: int = 60
    noise_history_frames: int = 60
    warmup_frames: int = 80
    noise_sample_index: int = 10
    level_band: tuple[int, int] = (16, 145)
    speech_to_noise_threshold: float = 3.0
    noise_vector_index: int = 0
    noise_vector_gain: float = 1.0

    def __post_init__(self):
        for name in ("microphone_history_frames", "event_frames", "noise_history_frames"):
            positive_integer(getattr(self, name), name)
        for name in ("warmup_frames", "noise_sample_index", "noise_vector_index"):
            nonnegative_integer(getattr(self, name), name)
        if self.event_frames > self.microphone_history_frames or self.noise_sample_index >= self.microphone_history_frames:
            raise ValueError("noise sampling and event history must fit microphone history")
        for name in ("speech_to_noise_threshold", "noise_vector_gain"):
            finite_scalar(getattr(self, name), name, minimum=0)
        object.__setattr__(self, "level_band", tuple(self.level_band))


@dataclass(frozen=True)
class ReflectionSelectionConfig:
    minimum_separation_steps: int = 24
    frequency_band: tuple[int, int] = (32, 129)
    array_radius_meters: float = .035
    radius_multiplier: float = 1.5
    reverse_peak_ratio: float = .8

    def __post_init__(self):
        positive_integer(self.minimum_separation_steps, "minimum_separation_steps")
        for name in ("array_radius_meters", "radius_multiplier", "reverse_peak_ratio"):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).tiny)
        object.__setattr__(self, "frequency_band", tuple(self.frequency_band))


@dataclass(frozen=True)
class SubspaceDirectionConfig:
    geometry: ArrayGeometry = field(default_factory=linear_array)
    frame_length_samples: int = 512
    sample_rate_hz: int = 16000
    block_frames: int = 40
    source_count: int = 1
    analysis_band: tuple[int, int] = (48, 151)
    energy_band: tuple[int, int] = (15, 144)
    beam_count: int = 3
    directions_per_beam: int = 12
    beam_width_degrees: float = 60.0
    grid_step_degrees: float = 5.0
    azimuth_offset_degrees: float = 30.0
    polar_angle_radians: float = np.pi / 2
    search_half_width_steps: int = 6
    smoothing_half_width_steps: int = 0
    circular_search: bool = False
    search_all_directions: bool = False
    strongest_active_beam_only: bool = True
    microphone_imbalance_db: float = 5.0
    active_confidence_ratio: float = 1.0
    inactive_confidence_ratio: float = 1.0
    interference_ratio: float = 2.0
    secondary_peak_fraction: float = .4
    secondary_peak_separation_steps: int = 6
    noise_subspace: NoiseSubspaceConfig | None = None
    reflection_selection: ReflectionSelectionConfig | None = None
    frequency_voting: str = "raw"

    def __post_init__(self):
        if self.frequency_voting not in ("raw", "normalized", "coherent"):
            raise ValueError("frequency_voting must be raw, normalized or coherent")
        if not isinstance(self.geometry, ArrayGeometry):
            raise TypeError("geometry must be an ArrayGeometry")
        for name in ("frame_length_samples", "sample_rate_hz", "block_frames", "source_count", "beam_count", "directions_per_beam"):
            positive_integer(getattr(self, name), name)
        if self.frame_length_samples % 2 or self.source_count >= self.geometry.microphone_count:
            raise ValueError("frame length must be even and source_count smaller than microphone count")
        for name in ("search_half_width_steps", "smoothing_half_width_steps", "secondary_peak_separation_steps"):
            nonnegative_integer(getattr(self, name), name)
        if self.smoothing_half_width_steps >= self.direction_count:
            raise ValueError("smoothing window is larger than the direction grid")
        for name in ("beam_width_degrees", "grid_step_degrees"):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).tiny)
        if not np.isclose(self.beam_width_degrees / self.grid_step_degrees, round(self.beam_width_degrees / self.grid_step_degrees)):
            raise ValueError("beam_width_degrees must be a multiple of grid_step_degrees")
        for name in ("azimuth_offset_degrees", "polar_angle_radians"):
            finite_scalar(getattr(self, name), name)
        for name in ("microphone_imbalance_db", "active_confidence_ratio", "inactive_confidence_ratio", "interference_ratio", "secondary_peak_fraction"):
            finite_scalar(getattr(self, name), name, minimum=0)
        for name in ("analysis_band", "energy_band"):
            self._validate_band(getattr(self, name), name)
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.noise_subspace is not None:
            if not isinstance(self.noise_subspace, NoiseSubspaceConfig):
                raise TypeError("noise_subspace must be a NoiseSubspaceConfig or None")
            self._validate_band(self.noise_subspace.level_band, "level_band")
            if self.noise_subspace.noise_vector_index >= self.geometry.microphone_count:
                raise ValueError("noise_vector_index is outside the microphone subspace")
        if self.reflection_selection is not None:
            if not isinstance(self.reflection_selection, ReflectionSelectionConfig):
                raise TypeError("reflection_selection must be a ReflectionSelectionConfig or None")
            self._validate_band(self.reflection_selection.frequency_band, "frequency_band")
            if not self.circular_search or self.reflection_selection.minimum_separation_steps > self.direction_count // 2:
                raise ValueError("reflection selection requires a circular grid and attainable separation")

    def _validate_band(self, band, name):
        if len(band) != 2:
            raise ValueError(f"{name} must contain start and stop")
        nonnegative_integer(band[0], name)
        positive_integer(band[1], name)
        if not band[0] < band[1] <= self.frequency_bin_count:
            raise ValueError(f"invalid {name}")

    @property
    def frequency_bin_count(self):
        return self.frame_length_samples // 2 + 1

    @property
    def direction_count(self):
        return self.beam_count * self.directions_per_beam


@dataclass(frozen=True)
class DirectionEstimate:
    azimuth_degrees: float | None
    spatial_spectrum: np.ndarray
    searched_directions: np.ndarray


class SubspaceDirectionEstimator:
    def __init__(self, config: SubspaceDirectionConfig | None = None, *, fixed_beam_weights=None):
        self.config = config if config is not None else SubspaceDirectionConfig()
        config = self.config
        self._angles = np.mod(config.azimuth_offset_degrees + np.arange(config.direction_count) * config.grid_step_degrees,
                              config.direction_count * config.grid_step_degrees)
        frequencies = np.arange(*config.analysis_band) * config.sample_rate_hz / config.frame_length_samples
        self._responses = plane_wave_responses(config.geometry, frequencies, self._angles, polar_angle_radians=config.polar_angle_radians)
        self._fixed_beam_weights = None
        if config.reflection_selection is not None:
            weights = spectral_array(fixed_beam_weights)
            if weights.shape != (config.frequency_bin_count, config.geometry.microphone_count, config.direction_count):
                raise ValueError("fixed_beam_weights must have shape (bins, microphones, directions)")
            self._fixed_beam_weights = weights.copy()
            reflection = config.reflection_selection
            speed = config.geometry.sound_speed_meters_per_second
            if np.ceil(2 * reflection.radius_multiplier * reflection.array_radius_meters / speed * config.sample_rate_hz) > np.ceil(2 / speed * config.sample_rate_hz):
                raise ValueError("reflection exclusion radius leaves no correlation delays")
        self.reset()

    @property
    def frame_index(self):
        return self._frame_index

    @property
    def scan_angles_degrees(self):
        return self._angles.copy()

    def reset(self):
        config = self.config
        history = config.noise_subspace.microphone_history_frames if config.noise_subspace else config.block_frames
        self._microphone_history = np.zeros((config.frequency_bin_count, history, config.geometry.microphone_count), complex)
        self._beam_history = np.zeros((config.frequency_bin_count, config.block_frames, config.beam_count), complex)
        self._noise_history = np.zeros((config.frequency_bin_count, config.noise_subspace.noise_history_frames if config.noise_subspace else 0,
                                        config.geometry.microphone_count), complex)
        self._noise_eligible = np.ones(history, bool)
        self._frame_index = 0
        self._last_azimuth = None

    def estimate(self, spectrum_frame, beam_frame=None, *, wake_confidence=None, confidence_threshold=1.0, reference_is_quiet=True):
        config = self.config
        spectrum = spectral_frame(spectrum_frame, config.frequency_bin_count, config.geometry.microphone_count)
        beams = None if beam_frame is None else spectral_frame(beam_frame, config.frequency_bin_count, config.beam_count)
        confidence = None
        if wake_confidence is not None:
            confidence = real_array(wake_confidence, "wake_confidence", 1)
            if confidence.shape != (config.beam_count,) or np.any(confidence < 0):
                raise ValueError("wake_confidence must contain one nonnegative value per beam")
            finite_scalar(confidence_threshold, "confidence_threshold", minimum=0)
        if not isinstance(reference_is_quiet, (bool, np.bool_)):
            raise ValueError("reference_is_quiet must be boolean")
        self._microphone_history[:, :-1] = self._microphone_history[:, 1:]
        self._microphone_history[:, -1] = spectrum
        self._frame_index += 1
        noise_config = config.noise_subspace
        if noise_config is None:
            self._beam_history[:, :-1] = self._beam_history[:, 1:]
            self._beam_history[:, -1] = 0 if beams is None else beams
        else:
            self._noise_eligible[:-1] = self._noise_eligible[1:]
            self._noise_eligible[-1] = reference_is_quiet
            if confidence is not None:
                self._noise_eligible[-noise_config.event_frames:] = False
            sample = noise_config.noise_sample_index
            if (self._frame_index > noise_config.warmup_frames and reference_is_quiet
                    and self._frame_index >= len(self._noise_eligible) - sample and self._noise_eligible[sample]):
                self._noise_history[:, :-1] = self._noise_history[:, 1:]
                self._noise_history[:, -1] = self._microphone_history[:, sample]
        if confidence is None:
            return None
        microphone_block = self._microphone_history
        if noise_config is not None:
            microphone_block = microphone_block[:, -noise_config.event_frames:]
        weights = np.ones(microphone_block.shape[:2])
        energy_db = 10 * np.log10(np.maximum(np.mean(np.abs(microphone_block) ** 2, axis=(0, 1)), .0001))
        search_all = config.search_all_directions or np.ptp(energy_db) > config.microphone_imbalance_db
        active = np.arange(config.beam_count)
        if not search_all and noise_config is None:
            if np.sum(confidence) == 0:
                inactive = np.empty(0, int)
            else:
                active = np.flatnonzero(confidence >= config.active_confidence_ratio * confidence_threshold)
                inactive = np.flatnonzero(confidence < config.inactive_confidence_ratio * confidence_threshold)
            if not len(active):
                self._last_azimuth = None
                scores = np.zeros((config.direction_count, config.analysis_band[1] - config.analysis_band[0]))
                return DirectionEstimate(None, scores, np.zeros(config.direction_count, dtype=bool))
            partial = len(active) > 0 and len(inactive) > 0
            if config.strongest_active_beam_only or partial:
                energies = np.mean(np.abs(self._beam_history[slice(*config.energy_band), :, active]) ** 2, axis=(0, 1))
                strongest = int(active[np.argmax(energies)])
                if partial and config.beam_count > 1:
                    power = np.abs(self._beam_history) ** 2
                    ratios = np.empty_like(power)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        for beam in range(config.beam_count):
                            ratios[:, :, beam] = power[:, :, beam] / np.max(np.delete(power, beam, axis=2), axis=2)
                    weights[ratios[:, :, strongest] <= np.max(ratios[:, :, inactive], axis=2) * config.interference_ratio] = 0
                if config.strongest_active_beam_only and energies.max() > 0:
                    active = np.array([strongest])
        scores, scan = self._scan(microphone_block, weights, active, full_grid=search_all)
        if not np.any(scores[scan]):
            self._last_azimuth = None
            return DirectionEstimate(None, np.abs(scores), scan.copy())
        if config.circular_search and noise_config is None:
            candidates = self._peaks(scores, scan)
            chosen = candidates[0]
            if config.reflection_selection is not None and len(candidates) > 1:
                chosen = self._choose_direct_arrival(candidates, microphone_block)
            azimuth = self._angles[chosen]
        else:
            azimuth = self._angles[np.argmax(np.abs(np.sum(scores, axis=1) * scan))]
        self._last_azimuth = float(azimuth)
        return DirectionEstimate(float(azimuth), np.abs(scores), scan.copy())

    def _scan(self, samples, weights, active, *, full_grid=False):
        config = self.config
        scan = np.zeros(config.direction_count, bool)
        extended = np.zeros_like(scan)
        for beam in active:
            center = int(round(beam * config.beam_width_degrees / config.grid_step_degrees))
            for mask, half in ((scan, config.search_half_width_steps), (extended, config.search_half_width_steps + config.smoothing_half_width_steps)):
                mask[(center + np.arange(-half, half + 1)) % config.direction_count] = True
        if full_grid:
            scan[:] = extended[:] = True
        band = slice(*config.analysis_band)
        snapshots = (samples[band] * weights[band, :, None]).swapaxes(1, 2)
        used = np.any(snapshots != 0, axis=(1, 2))
        scores = np.zeros((config.direction_count, len(used)))
        if not np.any(used):
            return scores, scan
        data = snapshots[used]
        covariance = data @ data.conj().swapaxes(1, 2) / samples.shape[1]
        eigenvalues, vectors = np.linalg.eigh(covariance)
        vectors = vectors[:, :, ::-1]
        basis = vectors[:, :, config.source_count:]
        gap = eigenvalues[:, -config.source_count] - eigenvalues[:, -config.source_count-1]
        resolved = gap > 32 * np.finfo(float).eps * eigenvalues[:, -1]
        noise_config = config.noise_subspace
        if noise_config is not None:
            level_band = slice(*noise_config.level_band)
            speech_level = np.mean(np.abs(samples[level_band]))
            noise_level = np.mean(np.abs(self._noise_history[level_band]))
            if noise_level > 0 and speech_level <= noise_config.speech_to_noise_threshold * noise_level:
                history = self._noise_history[band][used].swapaxes(1, 2)
                noise_covariance = history @ history.conj().swapaxes(1, 2) / history.shape[2]
                _, noise_vectors = np.linalg.eigh(noise_covariance)
                index = noise_config.noise_vector_index
                gain = min(noise_level / max(speech_level, np.finfo(float).tiny) * noise_config.noise_vector_gain, 1)
                noise_vector = gain * noise_vectors[:, :, -(index+1)]
                basis = np.concatenate((noise_vector[:, :, None], vectors[:, :, index+2:]), axis=2)
        responses = self._responses[used][:, :, extended]
        projection = basis.conj().swapaxes(1, 2) @ responses
        power = np.sum(np.abs(projection)**2, axis=1)
        microphones = config.geometry.microphone_count
        power = np.maximum(power, microphones * np.finfo(float).eps)
        spectra = resolved[:, None] * microphones / power
        if config.frequency_voting != "raw":
            peak = spectra.max(axis=1, keepdims=True)
            np.divide(spectra, peak, out=spectra, where=peak > 0)
            if config.frequency_voting == "coherent":
                # Favor energy separated from the adjacent noise eigenvalue.
                confidence = np.divide(gap, eigenvalues[:, -1], out=np.zeros_like(gap), where=resolved)
                spectra *= (gap * confidence)[:, None]
        scores[np.ix_(extended, used)] = spectra.T
        return scores, scan

    def _peaks(self, scores, scan):
        config = self.config
        half = config.smoothing_half_width_steps
        magnitude = np.abs(scores)
        smoothed = sum(np.roll(magnitude, shift, axis=0) for shift in range(-half, half + 1)) / (2 * half + 1)
        summed = np.sum(smoothed, axis=1) * scan
        target = int(np.argmax(summed))
        falling = (np.concatenate(([summed[-1]], summed)) - np.concatenate((summed, [summed[0]]))) > 0
        candidates = np.flatnonzero(falling[:-1].astype(int) - falling[1:].astype(int) == -1)
        selected = [target]
        if len(candidates) > 1:
            ranking = np.argsort(-summed[candidates], kind="stable")
            baseline, peak = np.min(summed[scan]), summed[candidates[ranking[0]]]
            for index in range(1, len(candidates)):
                first, second = candidates[ranking[0]], candidates[ranking[index]]
                separation = min((second - first) % len(scan), (first - second) % len(scan))
                if (summed[candidates[ranking[index]]] - baseline > config.secondary_peak_fraction * (peak - baseline)
                        and separation >= config.secondary_peak_separation_steps):
                    selected.append(int(candidates[ranking[index]]))
                    break
        return selected

    def _choose_direct_arrival(self, candidates, samples):
        config = self.config
        reflection = config.reflection_selection
        first, second = candidates
        for _ in range(config.direction_count):
            distances = ((second - first) % config.direction_count, (first - second) % config.direction_count)
            if min(distances) >= reflection.minimum_separation_steps:
                break
            direction = 1 if distances[0] <= distances[1] else -1
            first = (first - direction) % config.direction_count
            second = (second + direction) % config.direction_count
        direct = np.sum(samples * self._fixed_beam_weights[:, :, first][:, None], axis=2)
        reverse = np.sum(samples * self._fixed_beam_weights[:, :, second][:, None], axis=2)
        band = slice(*reflection.frequency_band)
        cross = direct[band] * reverse[band].conj()
        phase = np.divide(cross, np.abs(cross), out=np.zeros_like(cross), where=np.abs(cross) > 0)
        speed = config.geometry.sound_speed_meters_per_second
        maximum = int(np.ceil(2 / speed * config.sample_rate_hz))
        minimum = int(np.ceil(2 * reflection.radius_multiplier * reflection.array_radius_meters / speed * config.sample_rate_hz))
        delays = np.arange(-maximum, maximum + 1) / config.sample_rate_hz
        frequency = np.arange(*reflection.frequency_band) * config.sample_rate_hz / config.frame_length_samples
        correlation = np.real(np.sum(phase, axis=1) @ np.exp(-2j * np.pi * frequency[:, None] * delays))
        negative = np.max(correlation[:maximum - minimum + 1])
        positive = np.max(correlation[maximum + minimum:])
        return candidates[1] if positive < reflection.reverse_peak_ratio * negative else candidates[0]
