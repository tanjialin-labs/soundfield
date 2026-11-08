"""Binaural coherence models for bandwise direct-to-reverberant ratios."""

from dataclasses import dataclass, field

import numpy as np

from ..contracts import audio_array, finite_scalar, nonnegative_integer, positive_integer


@dataclass(frozen=True)
class ReferenceCancellationConfig:
    minimum_reference_amplitude: float = 1e-3
    single_reference_regularization: float = 1e-4
    dual_reference_regularization: float = 1e-10
    diagonal_loading: float = .01
    power_memory: float = .98
    estimation_segments: int = 8

    def __post_init__(self):
        positive_integer(self.estimation_segments, "estimation_segments")
        for name in ("minimum_reference_amplitude", "single_reference_regularization", "dual_reference_regularization", "diagonal_loading"):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).tiny)
        finite_scalar(self.power_memory, "power_memory", minimum=0, maximum=1)


@dataclass(frozen=True)
class BinauralReverberationConfig:
    sample_rate_hz: int = 16000
    microphone_spacing_meters: float = .03
    microphone_indices: tuple[int, int] = (0, 1)
    fft_length_samples: int = 512
    azimuth_degrees: float = 90.0
    scan_half_width_degrees: float = 0.0
    scan_step_degrees: float = 5.0
    power_memory: float = .95
    coherence_memory: float = .5
    activity_coherence_threshold: float = .98
    inactive_memory: float = .999
    active_memory: float = .95
    low_band: tuple[int, int] = (83, 164)
    high_band: tuple[int, int] = (163, 236)
    averaging_start_frame: int = 184
    averaging_stop_frame: int = 200
    reference_cancellation: ReferenceCancellationConfig | None = field(default_factory=ReferenceCancellationConfig)

    def __post_init__(self):
        for name in ("sample_rate_hz", "fft_length_samples", "averaging_stop_frame"):
            positive_integer(getattr(self, name), name)
        nonnegative_integer(self.averaging_start_frame, "averaging_start_frame")
        if self.fft_length_samples < 320 or self.fft_length_samples % 2:
            raise ValueError("fft_length_samples must be even and at least 320")
        if self.averaging_start_frame >= self.averaging_stop_frame:
            raise ValueError("averaging frame interval must be nonempty")
        indices = tuple(self.microphone_indices)
        if len(indices) != 2 or indices[0] == indices[1]:
            raise ValueError("microphone_indices must select two different channels")
        for index in indices:
            nonnegative_integer(index, "microphone_indices")
        object.__setattr__(self, "microphone_indices", indices)
        for name in ("microphone_spacing_meters", "scan_step_degrees"):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).tiny)
        finite_scalar(self.azimuth_degrees, "azimuth_degrees", minimum=0, maximum=180)
        finite_scalar(self.scan_half_width_degrees, "scan_half_width_degrees", minimum=0, maximum=180)
        steps = 2 * self.scan_half_width_degrees / self.scan_step_degrees
        if not np.isclose(steps, round(steps)):
            raise ValueError("scan width must contain an integer number of steps")
        for name in ("power_memory", "coherence_memory", "activity_coherence_threshold", "inactive_memory", "active_memory"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        for name in ("low_band", "high_band"):
            interval = tuple(getattr(self, name))
            if len(interval) != 2:
                raise ValueError(f"{name} must contain start and stop")
            nonnegative_integer(interval[0], name)
            positive_integer(interval[1], name)
            if not interval[0] < interval[1] <= self.fft_length_samples // 2 + 1:
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, interval)
        if self.reference_cancellation is not None and not isinstance(self.reference_cancellation, ReferenceCancellationConfig):
            raise TypeError("reference_cancellation must be a ReferenceCancellationConfig or None")


@dataclass(frozen=True)
class ReverberationRatio:
    low_band_db: float
    high_band_db: float
    low_band_history_db: np.ndarray
    high_band_history_db: np.ndarray
    scan_angles_degrees: np.ndarray


def _cancel_reference(microphones, references, config):
    if references is None or config is None:
        return microphones
    references = audio_array(references)
    count = references.shape[1]
    if len(references) != len(microphones) or count not in (1, 2):
        raise ValueError("reference_audio must have matching samples and one or two channels")
    if np.mean(np.abs(references[:, 0] - np.mean(references[:, 0]))) < config.minimum_reference_amplitude:
        return microphones
    required = (config.estimation_segments - 1) * 512 + 8192
    if len(microphones) < required:
        raise ValueError(f"reference cancellation requires at least {required} samples")
    output = np.zeros_like(microphones)
    for channel in range(2):
        cross = np.zeros((4097, count), complex)
        auto = np.zeros((4097, count), complex)
        reference_cross = np.zeros(4097, complex)
        for index in range(config.estimation_segments):
            start = index * 512
            target = np.fft.rfft(microphones[start:start + 8192, channel])
            reference = np.fft.rfft(references[start:start + 8192], axis=0)
            cross = config.power_memory * cross + (1 - config.power_memory) * target[:, None] * reference.conj()
            auto = config.power_memory * auto + (1 - config.power_memory) * reference * reference.conj()
            if count == 2:
                reference_cross = (config.power_memory * reference_cross
                                   + (1 - config.power_memory) * reference[:, 0] * reference[:, 1].conj())
        if count == 1:
            response = cross / (auto + config.single_reference_regularization)
        else:
            diagonal = auto + config.diagonal_loading * np.mean(auto, axis=0)
            coupling = .99 * reference_cross
            denominator = diagonal[:, 0] * diagonal[:, 1] - np.abs(coupling) ** 2 + config.dual_reference_regularization
            response = np.column_stack((cross[:, 0] * diagonal[:, 1] - cross[:, 1] * coupling.conj(),
                                         cross[:, 1] * diagonal[:, 0] - cross[:, 0] * coupling)) / denominator[:, None]
        impulse = np.fft.irfft(response, n=8192, axis=0)[:4096]
        partitions = impulse.reshape(256, 16, count, order="F")
        coefficients = np.fft.rfft(partitions, n=512, axis=0).reshape(257, 16 * count)
        history = np.zeros_like(coefficients)
        frame = np.zeros((512, count))
        for start in range(0, len(microphones), 256):
            length = min(256, len(microphones) - start)
            frame[:256] = frame[256:]
            frame[256:] = 0
            frame[256:256 + length] = references[start:start + length]
            history[:, count:] = history[:, :-count]
            history[:, :count] = np.fft.rfft(frame, axis=0)
            prediction = np.fft.irfft(np.sum(history * coefficients, axis=1), n=512)[256:]
            output[start:start + length, channel] = microphones[start:start + length, channel] - prediction[:length]
    return output


class BinauralReverberationEstimator:
    def __init__(self, config: BinauralReverberationConfig | None = None):
        self.config = config if config is not None else BinauralReverberationConfig()

    def evaluate(self, audio, *, reference_audio=None):
        config = self.config
        samples = audio_array(audio)
        if max(config.microphone_indices) >= samples.shape[1]:
            raise ValueError("microphone_indices exceed the available channels")
        frame_count = len(samples) // 160
        if frame_count < config.averaging_stop_frame:
            raise ValueError("audio does not cover the averaging frame interval")
        microphones = _cancel_reference(samples[:, config.microphone_indices], reference_audio, config.reference_cancellation)
        scan_count = int(round(2 * config.scan_half_width_degrees / config.scan_step_degrees)) + 1
        angles = np.clip(config.azimuth_degrees - config.scan_half_width_degrees
                         + np.arange(scan_count) * config.scan_step_degrees, 0, 180)
        frequency = np.linspace(0, config.sample_rate_hz / 2, config.fft_length_samples // 2 + 1)
        spacing = config.microphone_spacing_meters
        diffuse = np.sinc(8.8 * frequency * spacing / 680) / np.sqrt(1 + (2 * np.pi * frequency * spacing / 680) ** 4)
        radians = np.deg2rad(angles - 90)
        delay = spacing / 680 * (radians + np.sin(radians))
        left = 1.05 + .95 * np.cos(1.2 * (radians + np.pi / 2))
        right = 1.05 + .95 * np.cos(1.2 * (radians - np.pi / 2))
        level = np.sqrt((4 * (680 / spacing) ** 2 + (2 * np.pi * frequency[:, None] * right) ** 2)
                        / (4 * (680 / spacing) ** 2 + (2 * np.pi * frequency[:, None] * left) ** 2))
        epsilon = np.finfo(float).eps
        window = np.sqrt(.5 - .5 * np.cos(2 * np.pi * np.arange(320) / 320))
        frame = np.zeros((320, 2))
        auto = np.zeros((len(frequency), 2))
        cross = np.zeros(len(frequency), complex)
        coherence = np.zeros_like(cross)
        smooth_ratio = np.zeros((len(frequency), scan_count))
        low_history = np.empty((frame_count, scan_count))
        high_history = np.empty_like(low_history)
        for index in range(frame_count):
            frame[:160] = frame[160:]
            frame[160:] = microphones[index * 160:(index + 1) * 160]
            spectrum = np.fft.rfft(frame * window[:, None], n=config.fft_length_samples, axis=0)
            power = np.abs(spectrum) ** 2
            auto = config.power_memory * auto + (1 - config.power_memory) * power
            cross = config.power_memory * cross + (1 - config.power_memory) * spectrum[:, 0] * spectrum[:, 1].conj()
            instant = cross / np.maximum(np.sqrt(auto[:, 0] * auto[:, 1]), epsilon)
            coherence = config.coherence_memory * coherence + (1 - config.coherence_memory) * instant
            magnitude = np.abs(coherence[:, None]) ** 2
            quadratic = level ** 2 * (1 - magnitude)
            linear = (2 * level * np.cos(2 * np.pi * frequency[:, None] * delay) * diffuse[:, None]
                      - (1 + level ** 2) * magnitude)
            constant = diffuse[:, None] ** 2 - magnitude
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = ((1 + level ** 2) / 2 * (-linear + np.sqrt(np.maximum(linear ** 2 - 4 * quadratic * constant, epsilon)))
                         / (2 * quadratic))
            ratio = np.clip(np.nan_to_num(ratio, nan=.01, posinf=100, neginf=.01), .01, 100)
            instantaneous_coherence = spectrum[:, 0].conj() * spectrum[:, 1] / np.sqrt(power[:, 0] * power[:, 1] + epsilon)
            memory = (config.inactive_memory if np.mean(np.abs(instantaneous_coherence) ** 2) < config.activity_coherence_threshold
                      else config.active_memory)
            smooth_ratio = memory * smooth_ratio + (1 - memory) * ratio
            low_history[index] = 10 * np.log10(np.maximum(np.mean(smooth_ratio[slice(*config.low_band)], axis=0), np.finfo(float).tiny))
            high_history[index] = 10 * np.log10(np.maximum(np.mean(smooth_ratio[slice(*config.high_band)], axis=0), np.finfo(float).tiny))
        interval = slice(config.averaging_start_frame, config.averaging_stop_frame)
        return ReverberationRatio(float(np.max(np.mean(low_history[interval], axis=0))),
                                  float(np.max(np.mean(high_history[interval], axis=0))),
                                  low_history, high_history, angles)
