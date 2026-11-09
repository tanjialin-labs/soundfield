"""Speech RMS tracking for half-overlapped square-root Hann frames."""

from dataclasses import dataclass

import numpy as np

from ..contracts import finite_scalar, positive_integer


@dataclass(frozen=True)
class SpeechLevelConfig:
    """Digital levels use dBFS. The peak ceiling applies to reconstructed samples."""

    sample_rate_hz: int = 16000
    target_rms_dbfs: float = -20.0
    maximum_boost_db: float = 24.0
    noise_ceiling_dbfs: float = -40.0
    peak_ceiling_dbfs: float = -1.0

    def __post_init__(self):
        positive_integer(self.sample_rate_hz, "sample_rate_hz")
        finite_scalar(self.peak_ceiling_dbfs, "peak_ceiling_dbfs", maximum=0)
        finite_scalar(self.target_rms_dbfs, "target_rms_dbfs", maximum=self.peak_ceiling_dbfs)
        finite_scalar(self.maximum_boost_db, "maximum_boost_db", minimum=0)
        finite_scalar(self.noise_ceiling_dbfs, "noise_ceiling_dbfs", maximum=self.target_rms_dbfs)


class SpeechLevelTracker:
    def __init__(self, config, hop, window, initial_gain_db):
        self.config = config
        self.hop_seconds = hop / config.sample_rate_hz
        self.activity_weight = -np.expm1(-self.hop_seconds / .22)
        self.level_weight = -np.expm1(-self.hop_seconds / .35)
        self.window_energy = float(window @ window)
        self.overlap_sum = window + np.roll(window, hop)
        self.speech_level_db = None
        self.noise_level_db = None
        self.applied_gain_db = min(initial_gain_db, config.maximum_boost_db)
        self.inactive_seconds = 0.0
        self.activity_mean = 0.0
        self.nominal_gain_db = self.applied_gain_db

    def gain_for_frame(self, spectrum, *, activity, curve, reference_active):
        frame = np.fft.irfft(spectrum, n=len(self.overlap_sum))
        power = float(frame @ frame) / self.window_energy
        level = 10 * np.log10(max(power, 1e-24))
        self.activity_mean += self.activity_weight * (activity - self.activity_mean)
        speech = (activity >= curve.speech_threshold and level > curve.lower_level_db
                  and power > 0 and not reference_active)
        if self.noise_level_db is not None:
            speech = speech and level >= self.noise_level_db + 4

        if speech:
            self.inactive_seconds = 0.0
            if self.speech_level_db is None:
                self.speech_level_db = level
            self.speech_level_db += self.level_weight * (level - self.speech_level_db)
            requested = self.config.target_rms_dbfs - self.speech_level_db
            if self.noise_level_db is not None:
                noise_headroom = self.config.noise_ceiling_dbfs - self.noise_level_db
                requested = min(requested, max(0.0, noise_headroom))
            self.nominal_gain_db = min(requested, self.config.maximum_boost_db)
        else:
            self.inactive_seconds += self.hop_seconds
            if not reference_active and activity < curve.speech_threshold and self.inactive_seconds >= .20:
                if self.noise_level_db is None:
                    self.noise_level_db = level
                # Quieter observations reach the estimate sooner than rising backgrounds.
                self.noise_level_db += np.clip(level - self.noise_level_db,
                                              -20 * self.hop_seconds, 3 * self.hop_seconds)

        requested = self.nominal_gain_db
        if requested > 0:
            confidence = np.clip((self.activity_mean - .18) / .24, 0.0, 1.0)
            requested *= confidence
        if self.inactive_seconds > .30:
            requested = min(0.0, requested)
        self.applied_gain_db += np.clip(requested - self.applied_gain_db,
                                       -60 * self.hop_seconds, 48 * self.hop_seconds)

        # Each overlapping frame contributes at most its window's share of the ceiling.
        peak_bound = np.max(np.abs(frame) * self.overlap_sum)
        peak_gain_db = self.config.peak_ceiling_dbfs - 20 * np.log10(max(peak_bound, 1e-12))
        gain = min(self.applied_gain_db, peak_gain_db)
        if reference_active:
            gain = min(gain, 0.0, curve.quiet_output_db - level)
        return float(gain)
