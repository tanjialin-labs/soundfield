"""Coherence-based residual echo estimation and critical-band suppression."""

from dataclasses import dataclass

import numpy as np

from ..contracts import positive_integer, finite_scalar, spectral_frame, unit_interval_array
from ..numerics import spectral_amplitude_gain


@dataclass(frozen=True)
class ResidualEchoConfig:
    frequency_bin_count: int = 513
    sample_rate_hz: int = 16000
    band_count: int = 24
    method: str = 'critical_band'
    coherence_smoothing: float = .984
    coherence_scale: float = 1.7
    noise_floor: float = 1e-7
    posterior_weight: float = .02
    minimum_prior_ratio: float = .0032
    enable_energy_gate: bool = True
    energy_smoothing: float = .9
    residual_energy_ratio: float = .3
    double_talk_energy_threshold: float = .08
    energy_intervals: tuple = ((0, 2), (99, 512))
    mean_gain_scale: float = .1
    enable_level_detection: bool = True
    high_level_threshold: float = .0005
    high_level_attack_frames: int = 10
    high_level_release_frames: int = 200
    maximum_snr: float = 40.0
    preserve_residual_power: bool = False

    def __post_init__(self):
        for name in ('frequency_bin_count', 'sample_rate_hz', 'band_count', 'high_level_attack_frames', 'high_level_release_frames'):
            positive_integer(getattr(self, name), name)
        if not isinstance(self.preserve_residual_power, bool):
            raise ValueError("preserve_residual_power must be boolean")
        if self.frequency_bin_count < 2 or self.band_count < 12:
            raise ValueError('at least two frequency bins and twelve bands are required')
        if self.method not in ('critical_band', 'spectral', 'mask'):
            raise ValueError('method must be critical_band, spectral or mask')
        for name in ('coherence_smoothing', 'posterior_weight', 'energy_smoothing'):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        for name in ('coherence_scale', 'minimum_prior_ratio', 'residual_energy_ratio',
                     'double_talk_energy_threshold', 'mean_gain_scale', 'high_level_threshold'):
            finite_scalar(getattr(self, name), name, minimum=0)
        finite_scalar(self.noise_floor, 'noise_floor', minimum=np.finfo(float).eps)
        finite_scalar(self.maximum_snr, 'maximum_snr', minimum=np.finfo(float).eps)
        if any(not 0 <= start < stop <= self.frequency_bin_count for start, stop in self.energy_intervals):
            raise ValueError('energy intervals must fit inside the spectrum')


class ResidualLevelDetector:
    def __init__(self, threshold=.0005, attack_frames=10, release_frames=200):
        finite_scalar(threshold, 'threshold', minimum=0)
        positive_integer(attack_frames, 'attack_frames')
        positive_integer(release_frames, 'release_frames')
        self.threshold, self.attack_frames, self.release_frames = threshold, attack_frames, release_frames
        self.reset()

    def reset(self):
        self.smoothed_power = 0.0
        self.is_high_level = False
        self.high_frames = self.low_frames = 0

    def process_frame(self, residual_echo_power):
        self.smoothed_power = .8*self.smoothed_power+.2*np.mean(residual_echo_power)
        if not self.is_high_level:
            self.high_frames = self.high_frames+1 if self.smoothed_power > self.threshold else 0
            if self.high_frames >= self.attack_frames:
                self.is_high_level, self.high_frames = True, 0
        else:
            self.low_frames = self.low_frames+1 if self.smoothed_power < self.threshold else 0
            if self.low_frames >= self.release_frames:
                self.is_high_level, self.low_frames = False, 0
        return self.is_high_level


class ResidualEchoPostfilter:
    def __init__(self, config=ResidualEchoConfig()):
        self.config = config
        self.level_detector = ResidualLevelDetector(config.high_level_threshold, config.high_level_attack_frames,
                                                    config.high_level_release_frames)
        frequency = np.linspace(0, config.sample_rate_hz/2, config.frequency_bin_count)
        scale = 13.1*np.arctan(.00074*frequency)+2.24*np.arctan(frequency**2*1.85e-8)+1e-4*frequency
        position = scale/(scale[-1]/(config.band_count-1))
        self.left_band = np.minimum(np.floor(position).astype(int), config.band_count-2)
        self.right_fraction = np.minimum(position-self.left_band, 1)
        self.band_weights = np.zeros((config.band_count, config.frequency_bin_count))
        indices = np.arange(config.frequency_bin_count)
        self.band_weights[self.left_band, indices] = 1-self.right_fraction
        self.band_weights[self.left_band+1, indices] = self.right_fraction
        self.reset()

    def reset(self):
        bins = self.config.frequency_bin_count
        self.frame_count = 0
        self.frame_probability = .1
        self.cross_power = np.zeros(bins, complex)
        self.echo_power = np.zeros(bins)
        self.residual_power = np.zeros(bins)
        size = bins+self.config.band_count if self.config.method == 'critical_band' else bins
        self.previous_output_power = np.zeros(size)
        self.held_echo_power = np.zeros(size)
        self.smoothed_prior = np.zeros(size)
        self.reference_energy, self.residual_energy = .1, 0.0
        self.gains = np.zeros(bins)
        self.level_detector.reset()

    def _estimate_residual_echo(self, residual, estimated, power, echo):
        smoothing = self.config.coherence_smoothing if self.frame_count else 0
        self.cross_power = smoothing*self.cross_power+(1-smoothing)*estimated*residual.conj()
        self.echo_power = smoothing*self.echo_power+(1-smoothing)*echo
        self.residual_power = smoothing*self.residual_power+(1-smoothing)*power
        ratio = np.minimum(1, np.divide(self.residual_power, self.echo_power, out=np.ones_like(power), where=self.echo_power>0))
        denominator = np.sqrt(self.residual_power*self.echo_power)
        coherence = np.divide(np.abs(self.cross_power), denominator, out=np.zeros_like(power), where=denominator>0)
        return echo*np.minimum(1, self.config.coherence_scale*ratio*coherence)+self.config.noise_floor

    def process_frame(self, residual_spectrum, echo_spectrum, *, speech_mask=None):
        config = self.config
        bins = config.frequency_bin_count
        residual = spectral_frame(residual_spectrum, bins, 1)[:, 0]
        estimated = spectral_frame(echo_spectrum, bins, 1)[:, 0]
        power, echo = np.abs(residual)**2, np.abs(estimated)**2
        residual_echo = self._estimate_residual_echo(residual, estimated, power, echo)
        if config.enable_level_detection:
            self.level_detector.process_frame(residual_echo)
        if config.method == 'mask':
            if speech_mask is None:
                raise ValueError('mask mode requires speech_mask for every active frame')
            gains = unit_interval_array(speech_mask, 'speech_mask', (bins,)).copy()
        elif config.method == 'spectral':
            posterior = np.minimum(power/residual_echo, 40)
            prior = np.maximum(config.posterior_weight*np.maximum(posterior-1, 0)
                               +(1-config.posterior_weight)*self.previous_output_power/residual_echo, config.minimum_prior_ratio)
            prior_ratio = prior/(1+prior)
            gains = np.minimum(1, prior_ratio*spectral_amplitude_gain(prior_ratio*posterior))
        else:
            self.held_echo_power[:bins] = np.maximum(.6*self.held_echo_power[:bins], residual_echo)
            self.held_echo_power[bins:] = self.band_weights @ self.held_echo_power[:bins]
            combined_power = np.r_[power, self.band_weights @ power]
            if self.frame_count == 0:
                self.previous_output_power = combined_power.copy()
            noise = config.noise_floor+self.held_echo_power
            posterior = np.minimum(combined_power/noise-1, config.maximum_snr)
            weight = .1+.89*(self.previous_output_power/(self.previous_output_power+noise))**2
            prior = np.minimum(weight*np.maximum(posterior, 0)+(1-weight)*self.previous_output_power/noise, config.maximum_snr)
            self.smoothed_prior[bins-1:] = .7*self.smoothed_prior[bins-1:]+.3*prior[bins-1:]
            mean_prior = self.smoothed_prior[bins+6:bins+config.band_count-5].sum()/(config.band_count-11)
            frame_probability = .1+.899*mean_prior/(mean_prior+.15)
            self.frame_probability = frame_probability
            attenuation = (1-frame_probability)*-40+frame_probability*-15
            echo_floor = np.exp(.2302585*np.trunc(attenuation))
            band_floor = np.sqrt(echo_floor*self.held_echo_power[bins:])/np.sqrt(1+self.held_echo_power[bins:])
            ratio = prior/(prior+1)
            argument = ratio*(1+posterior)
            base_gain = np.minimum(1, ratio*spectral_amplitude_gain(np.maximum(argument, 0)))
            self.previous_output_power[bins:] = .2*self.previous_output_power[bins:]+.8*base_gain[bins:]**2*combined_power[bins:]
            smooth = self.smoothed_prior[bins:]
            band_probability = .199+.8*smooth/(smooth+.15)
            absence = 1-frame_probability*band_probability
            probability = 1/(1+absence/(1-absence)*(1+prior[bins:])*np.exp(-argument[bins:]))
            band_presence = probability.copy()
            interpolated = self.band_weights.T @ base_gain[bins:]
            gains = np.where(.333*base_gain[:bins] > interpolated, 3*interpolated, base_gain[:bins])
            self.previous_output_power[:bins] = .2*self.previous_output_power[:bins]+.8*gains**2*power
            floor = self.band_weights.T @ band_floor
            gains = np.maximum(gains, floor)
            probability = self.band_weights.T @ probability
            gains = (probability*np.sqrt(gains)+(1-probability)*np.sqrt(floor))**2
        if config.enable_energy_gate:
            self.reference_energy = .8*self.reference_energy+.2*echo[1:].mean()
            output_power = power*gains**2
            energy = sum(output_power[start:stop].sum() for start, stop in config.energy_intervals)
            self.residual_energy = config.energy_smoothing*self.residual_energy+(1-config.energy_smoothing)*energy
            if self.residual_energy < min(config.residual_energy_ratio*self.reference_energy, config.double_talk_energy_threshold):
                mean_gain = np.r_[gains, band_presence].mean() if config.method == 'critical_band' else gains.mean()
                gains = np.minimum(gains, config.mean_gain_scale*mean_gain)
        if config.preserve_residual_power and config.method != 'mask':
            nuisance = np.maximum(residual_echo-config.noise_floor,0)
            fraction = np.divide(nuisance,power,out=np.ones_like(power),where=power>0)
            gains = np.maximum(gains, 1-np.sqrt(np.minimum(fraction,1)))
        output = residual*gains
        if config.method == 'spectral':
            self.previous_output_power = np.abs(output)**2
        self.frame_count += 1
        self.gains = gains.copy()
        return output[:, None]
