"""Speech-driven beam selection with separate speech and noise persistence."""

from dataclasses import dataclass
import numpy as np
from ..contracts import positive_integer, finite_scalar, spectral_frame, unit_interval_array
from ..numerics.spectral_functions import rescale_upper_range


@dataclass(frozen=True)
class SpeechBeamConfig:
    channel_count: int = 6
    selected_count: int = 1
    high_band_count: int = 1
    evidence_band: tuple[int, int] = (9, 161)
    speech_threshold: float = .45
    switch_frames: int = 6
    energy_weighted: bool = True
    use_network_probability: bool = False
    noise_threshold: float = .3
    noise_switch_frames: int = 3
    magnitude_reduction: str = 'maximum'
    gain_bounds: tuple[float, float] | None = None
    output_mode: str = 'blended'

    def __post_init__(self):
        for name in ('channel_count', 'selected_count', 'high_band_count', 'switch_frames', 'noise_switch_frames'):
            positive_integer(getattr(self, name), name)
        if not self.high_band_count <= self.selected_count <= self.channel_count:
            raise ValueError('beam counts must satisfy high_band_count <= selected_count <= channel_count')
        if not 0 <= self.evidence_band[0] < self.evidence_band[1] <= 257:
            raise ValueError('evidence_band must fit a 257-bin spectrum')
        for name in ('speech_threshold', 'noise_threshold'):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        if self.magnitude_reduction not in ('maximum', 'mean') or self.output_mode not in ('blended', 'single'):
            raise ValueError('unsupported beam reduction or output mode')
        if self.output_mode == 'single' and self.selected_count != 1:
            raise ValueError('single output mode requires one selected beam')
        if self.gain_bounds is not None and not 0 <= self.gain_bounds[0] < self.gain_bounds[1] <= 1:
            raise ValueError('gain_bounds must increase within [0, 1]')


@dataclass(frozen=True)
class SpeechBeamResult:
    spectrum: np.ndarray
    speech_gain: np.ndarray
    selected_channels: np.ndarray
    speech_evidence: float
    noise_channel: int
    noise_level_db: float
    dominant_channel: int


class SpeechBeamSelector:
    """Accept normalized spectra; noise levels use 16-bit amplitude spectral units."""

    def __init__(self, config=SpeechBeamConfig()):
        self.config = config
        self.floor = np.r_[np.full(32, .11), np.linspace(.11, .1, 33), np.full(192, .1)]
        self.exponents = np.r_[np.full(100, 2.), np.linspace(2., 1., 72), np.ones(85)]
        self.reset()

    def reset(self):
        self.selected = np.arange(self.config.selected_count)
        self.pending_channel = 0
        self.pending_frames = 0
        self.noise_channel = 0
        self.pending_noise_channel = 0
        self.pending_noise_frames = 0
        self.noise_level_db = 30.
        self.direction_history = np.zeros(64, int)

    def process_frame(self, spectrum_frame, speech_gains, *, speech_probability=None):
        config = self.config
        spectrum = spectral_frame(spectrum_frame, 257, config.channel_count)
        gains = unit_interval_array(speech_gains, 'speech_gains', spectrum.shape).copy()
        evidence_gains = gains.copy()
        evidence_gains[:, self.selected[0]] = 0
        band = slice(*config.evidence_band)
        if config.use_network_probability:
            probabilities = unit_interval_array(speech_probability, 'speech_probability', (config.channel_count,))
            evidence = float(np.max(probabilities))
        else:
            evidence = float(np.max(np.mean(evidence_gains[band], axis=0)))
        if evidence > config.speech_threshold:
            energy = np.sum(np.abs(spectrum[band])*gains[band] if config.energy_weighted
                            else gains[band]**self.exponents[band, None], axis=0)
            candidates = np.argsort(-energy, kind='stable')[:config.selected_count]
            if candidates[0] != self.selected[0]:
                if self.pending_frames == 0:
                    self.pending_channel = int(candidates[0])
                self.pending_frames = self.pending_frames+1 if candidates[0] == self.pending_channel else 0
                if self.pending_frames >= config.switch_frames:
                    self.selected = candidates
                    self.pending_frames = 0
            else:
                self.pending_frames = 0
        if evidence > .35:
            self.direction_history[:-1] = self.direction_history[1:]
            self.direction_history[-1] = self.selected[0]
        if evidence < config.noise_threshold:
            noise_energy = np.sum(np.abs(spectrum[49:])*(1-gains[49:]), axis=0)
            noise_energy[self.selected] = 0
            candidate = int(np.argmax(noise_energy))
            if candidate != self.noise_channel:
                if self.pending_noise_frames == 0:
                    self.pending_noise_channel = candidate
                self.pending_noise_frames = self.pending_noise_frames+1 if candidate == self.pending_noise_channel else 0
                if self.pending_noise_frames >= config.noise_switch_frames:
                    self.noise_channel, self.pending_noise_frames = candidate, 0
            else:
                self.pending_noise_frames = 0
        noise_gain = gains[3:, self.noise_channel].copy()
        noise_gain[noise_gain > .4] = 1
        noise_energy = np.sum(np.abs(spectrum[3:, self.noise_channel])*32768*(1-noise_gain))
        self.noise_level_db = .5*self.noise_level_db+.5*10*np.log10(max(2, noise_energy))
        if config.gain_bounds is not None:
            lower, upper = config.gain_bounds
            gains = rescale_upper_range(gains, lower, upper)
        speech_gain = np.max(gains[:, self.selected], axis=1, keepdims=True)
        output = np.zeros((257, 1), complex)
        if config.output_mode == 'single':
            exponent = self.exponents if evidence > .1 else np.full(257, 2.)
            floor = self.floor*3 if evidence > .1 else self.floor
            output[:, 0] = spectrum[:, self.selected[0]]*np.maximum(floor, gains[:, self.selected[0]]**exponent)
        else:
            for interval, selected in ((slice(1, 50), self.selected), (slice(50, None), self.selected[:config.high_band_count])):
                gain = gains[interval][:, selected].max(axis=1)
                attenuation = np.maximum(self.floor[interval], gain**np.maximum(1, 2-gain))
                values = spectrum[interval][:, selected]
                magnitude = (np.max if config.magnitude_reduction == 'maximum' else np.mean)(np.abs(values)*attenuation[:, None], axis=1)
                phase = np.divide(values[:, 0], np.abs(values[:, 0]), out=np.zeros(len(values), complex), where=np.abs(values[:, 0])>0)
                output[interval, 0] = magnitude*phase
        dominant = int(np.argmax(np.bincount(self.direction_history, minlength=config.channel_count)))
        return SpeechBeamResult(output, speech_gain, self.selected.copy(), evidence, self.noise_channel, self.noise_level_db, dominant)
