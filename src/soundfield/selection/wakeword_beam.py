"""Select a wake-word beam from speech and preceding noise intervals."""

from dataclasses import dataclass

import numpy as np

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, spectral_array, unit_interval_array


@dataclass(frozen=True)
class WakeWordSelectionConfig:
    frequency_bin_count: int = 257
    excluded_tail_frames: int = 8
    speech_frames: int = 30
    separation_frames: int = 30
    noise_frames: int = 30
    full_band: tuple[int, int] = (2, 256)
    high_band: tuple[int, int] = (128, 256)
    low_band: tuple[int, int] = (2, 64)
    noise_band: tuple[int, int] = (31, 256)
    low_band_fraction: float = .4
    noise_imbalance_ratio: float = .1

    def __post_init__(self):
        for name in ("frequency_bin_count", "speech_frames", "noise_frames"):
            positive_integer(getattr(self, name), name)
        for name in ("excluded_tail_frames", "separation_frames"):
            nonnegative_integer(getattr(self, name), name)
        for name in ("full_band", "high_band", "low_band", "noise_band"):
            interval = tuple(getattr(self, name))
            if len(interval) != 2:
                raise ValueError(f"{name} must contain start and stop")
            nonnegative_integer(interval[0], name)
            positive_integer(interval[1], name)
            if not interval[0] < interval[1] <= self.frequency_bin_count:
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, interval)
        for name in ("low_band_fraction", "noise_imbalance_ratio"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)

    @property
    def required_frames(self):
        return self.excluded_tail_frames + self.speech_frames + self.separation_frames + self.noise_frames


class WakeWordBeamSelector:
    def __init__(self, config: WakeWordSelectionConfig | None = None):
        self.config = config if config is not None else WakeWordSelectionConfig()

    def select(self, beam_activity, spectrum):
        config = self.config
        values = spectral_array(spectrum)
        if values.shape[0] != config.frequency_bin_count:
            raise ValueError("frequency bin count does not match configuration")
        activity = unit_interval_array(beam_activity, "beam_activity", (values.shape[2],))
        if np.any((activity != 0) & (activity != 1)):
            raise ValueError("beam_activity must contain zero or one")
        active = np.flatnonzero(activity)
        if len(active) == 1:
            return int(active[0])
        if not len(active):
            activity = np.ones_like(activity)
        if values.shape[1] < config.required_frames:
            raise ValueError(f"at least {config.required_frames} frames are required")
        end = values.shape[1] - config.excluded_tail_frames
        speech_start = end - config.speech_frames
        noise_end = speech_start - config.separation_frames
        def energy(band, start, stop):
            return activity * np.sum(np.abs(values[band[0]:band[1], start:stop]) ** 2, axis=(0, 1))
        low = energy(config.low_band, speech_start, end)
        full = energy(config.full_band, speech_start, end)
        high = energy(config.high_band, speech_start, end)
        noise = energy(config.noise_band, noise_end - config.noise_frames, noise_end)
        ordered = np.argsort(low, kind="stable")
        if low[ordered[-1]] == 0:
            return int(np.flatnonzero(activity)[0])
        candidates = ordered[low[ordered] >= config.low_band_fraction * low[ordered[-1]]]
        candidates = candidates[activity[candidates] == 1]
        if len(candidates) == 1:
            return int(candidates[0])
        noise_max = np.max(noise[candidates])
        use_full = noise_max > 0 and np.min(noise[candidates]) / noise_max < config.noise_imbalance_ratio
        numerator = full[candidates] if use_full else high[candidates]
        ratio = np.divide(numerator, noise[candidates], out=np.full_like(numerator, np.inf), where=noise[candidates] > 0)
        ratio[(noise[candidates] == 0) & (numerator == 0)] = 0
        return int(candidates[np.argmax(ratio)])
