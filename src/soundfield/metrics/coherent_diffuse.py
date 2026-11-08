"""Two-microphone coherent-to-diffuse energy and impulse-response ratios."""

from dataclasses import dataclass
import numpy as np
from ..contracts import audio_array, real_array, positive_integer, finite_scalar


@dataclass(frozen=True)
class CoherentDiffuseConfig:
    sample_rate_hz: int = 16000
    microphone_spacing_meters: float = .02
    sound_speed_meters_per_second: float = 340.
    block_samples: int = 320
    hop_samples: int = 320
    fft_length_samples: int = 512
    power_memory: float = .85
    coherence_memory: float = .5
    active_memory: float = .98
    inactive_memory: float = 1.
    activity_threshold: float = .95
    ratio_limits_db: tuple[float, float] = (-20., 20.)

    def __post_init__(self):
        for name in ('sample_rate_hz', 'block_samples', 'hop_samples', 'fft_length_samples'):
            positive_integer(getattr(self, name), name)
        if self.fft_length_samples < self.block_samples or self.fft_length_samples % 2:
            raise ValueError('FFT length must be even and cover the analysis block')
        for name in ('microphone_spacing_meters', 'sound_speed_meters_per_second'):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).tiny)
        for name in ('power_memory', 'coherence_memory', 'active_memory', 'inactive_memory', 'activity_threshold'):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        if len(self.ratio_limits_db) != 2 or not np.isfinite(self.ratio_limits_db).all() or self.ratio_limits_db[0] >= self.ratio_limits_db[1]:
            raise ValueError('ratio limits must be finite and increasing')


def coherent_diffuse_ratio(audio, config=CoherentDiffuseConfig()):
    """Return one dB value per complete frame; zero smoothed energy gives -inf."""
    samples = audio_array(audio)
    if samples.shape[1] != 2 or len(samples) < config.block_samples:
        raise ValueError('input must have two channels and at least one full block')
    frequency = np.linspace(1, config.sample_rate_hz/2, config.fft_length_samples//2+1)
    diffuse = np.sinc(2*frequency*config.microphone_spacing_meters/config.sound_speed_meters_per_second)
    window = np.sqrt(.5-.5*np.cos(2*np.pi*np.arange(config.block_samples)/config.block_samples))
    auto = np.zeros((len(frequency), 2))
    cross, coherence = np.zeros(len(frequency), complex), np.zeros(len(frequency), complex)
    ratio = np.zeros(len(frequency))
    history = []
    epsilon = np.finfo(float).eps
    for start in range(0, len(samples)-config.block_samples+1, config.hop_samples):
        spectrum = np.fft.rfft(samples[start:start+config.block_samples]*window[:, None], n=config.fft_length_samples, axis=0)
        power = np.abs(spectrum)**2
        auto = config.power_memory*auto+(1-config.power_memory)*power
        cross = config.power_memory*cross+(1-config.power_memory)*spectrum[:, 0].conj()*spectrum[:, 1]
        coherence = config.coherence_memory*coherence+(1-config.coherence_memory)*cross/np.maximum(np.sqrt(auto[:, 0]*auto[:, 1]), epsilon)
        with np.errstate(divide='ignore', invalid='ignore'):
            estimate = (diffuse-coherence.real)/(coherence.real-1)
        estimate = np.nan_to_num(estimate, nan=0., posinf=np.inf, neginf=0.)
        estimate = np.clip(estimate, *10**(np.asarray(config.ratio_limits_db)/10))
        activity = np.mean(np.abs(spectrum[:, 0].conj()*spectrum[:, 1]/np.sqrt(power[:, 0]*power[:, 1]+epsilon))**2)
        memory = config.inactive_memory if activity < config.activity_threshold else config.active_memory
        ratio = memory*ratio+(1-memory)*estimate
        history.append(10*np.log10(np.mean(ratio)) if np.any(ratio) else -np.inf)
    return np.asarray(history)


def impulse_response_onset(impulse_response, sample_rate_hz, *, method='window_peak'):
    """Locate the peak directly or use overlapping 1 ms maximum windows."""
    response = real_array(impulse_response, 'impulse_response', 1)
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    if not len(response) or not np.any(response):
        raise ValueError('impulse response must contain nonzero samples')
    if method == 'peak':
        return int(np.argmax(np.abs(response)))
    if method != 'window_peak' or sample_rate_hz % 2000:
        raise ValueError('window_peak requires a sample rate divisible by 2000')
    length, hop = sample_rate_hz//1000, sample_rate_hz//2000
    if len(response) < length:
        raise ValueError('response is shorter than the onset window')
    maxima = np.max(np.abs(np.lib.stride_tricks.sliding_window_view(response, length)[::hop]), axis=1)
    return int(np.floor(np.linspace(1, len(response), len(maxima))[np.argmax(maxima)]))-1


def impulse_direct_reverberant_ratio(impulse_response, sample_rate_hz, *, direct_seconds=.005, onset_method='window_peak'):
    """Include pre-onset energy and the inclusive direct interval in the numerator."""
    response = real_array(impulse_response, 'impulse_response', 1)
    finite_scalar(direct_seconds, 'direct_seconds', minimum=0)
    onset = impulse_response_onset(response, sample_rate_hz, method=onset_method)
    stop = min(len(response), onset+int(np.floor(direct_seconds*sample_rate_hz))+1)
    direct, reverberant = np.sum(response[:stop]**2), np.sum(response[stop:]**2)
    return float(10*np.log10(direct/reverberant)) if reverberant else float('inf')
