"""Far-end single-talk decisions from residual level and echo reduction."""

from dataclasses import dataclass

import numpy as np

from ..contracts import audio_array, finite_scalar, positive_integer, real_array, spectral_frame


@dataclass(frozen=True)
class FarendSpeechConfig:
    hop_length_samples: int = 512
    frequency_band: tuple = (31, 192)
    echo_power_threshold: float = .005
    active_residual_threshold: float = .004
    quiet_residual_threshold: float = .0005
    active_reduction_threshold_db: float = 20
    quiet_reduction_threshold_db: float = 20
    activation_frames: int = 1
    release_frames: int = 30

    def __post_init__(self):
        for name in ('hop_length_samples', 'activation_frames', 'release_frames'):
            positive_integer(getattr(self, name), name)
        start, stop = self.frequency_band
        if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop <= self.hop_length_samples+1:
            raise ValueError('frequency_band must be a nonempty interval inside the spectrum')
        for name in ('echo_power_threshold', 'active_residual_threshold', 'quiet_residual_threshold',
                     'active_reduction_threshold_db', 'quiet_reduction_threshold_db'):
            finite_scalar(getattr(self, name), name, minimum=0)


@dataclass(frozen=True)
class FarendSpeechResult:
    is_farend_speech: bool
    is_double_talk: bool
    residual_power: float
    echo_power: float
    microphone_power: float
    echo_reduction_db: float


class FarendSpeechDetector:
    def __init__(self, config=FarendSpeechConfig()):
        self.config = config
        self.window = np.sqrt(np.hanning(2*config.hop_length_samples))
        self.reset()

    def reset(self):
        self.microphone_history = np.zeros(2*self.config.hop_length_samples)
        self.is_farend_speech = False
        self.active_frames = self.quiet_frames = 0

    def process_block(self, microphone_audio, residual_spectrum, echo_power):
        config, hop = self.config, self.config.hop_length_samples
        microphone = audio_array(microphone_audio)
        residual = spectral_frame(residual_spectrum, hop+1, 1)
        echo = real_array(echo_power, 'echo_power', 1)
        if microphone.shape != (hop, 1) or echo.shape != (hop+1,) or np.any(echo < 0):
            raise ValueError('microphone and echo power must match the configured frame')
        self.microphone_history[:hop] = self.microphone_history[hop:]
        self.microphone_history[hop:] = microphone[:, 0]
        spectrum = np.fft.rfft(self.microphone_history*self.window)
        band = slice(*config.frequency_band)
        residual_level = float(np.mean(np.abs(residual[band, 0])**2))
        echo_level = float(np.mean(echo[band]))
        microphone_level = float(np.mean(np.abs(spectrum[band])**2))
        tiny = np.finfo(float).tiny
        reduction = float(10*(np.log10(max(microphone_level, tiny))-np.log10(max(residual_level, tiny))))
        reference_active = echo_level > config.echo_power_threshold
        level_threshold = config.active_residual_threshold if reference_active else config.quiet_residual_threshold
        reduction_threshold = config.active_reduction_threshold_db if reference_active else config.quiet_reduction_threshold_db
        double_talk = residual_level > level_threshold and reduction < reduction_threshold
        single_talk = reference_active and not double_talk
        if not self.is_farend_speech:
            self.active_frames = self.active_frames+1 if single_talk else 0
            if self.active_frames >= config.activation_frames:
                self.is_farend_speech, self.active_frames = True, 0
        elif double_talk:
            self.is_farend_speech, self.quiet_frames = False, 0
        else:
            self.quiet_frames = 0 if single_talk else self.quiet_frames+1
            if self.quiet_frames >= config.release_frames:
                self.is_farend_speech, self.quiet_frames = False, 0
        return FarendSpeechResult(self.is_farend_speech, bool(double_talk), residual_level,
                                 echo_level, microphone_level, reduction)
