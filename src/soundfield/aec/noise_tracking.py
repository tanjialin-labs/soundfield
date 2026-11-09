"""Two-stage minimum tracking and speech-presence spectral gains."""

from dataclasses import dataclass

import numpy as np

from ..contracts import positive_integer, finite_scalar, real_array
from ..numerics import exponential_integral_approximation


@dataclass(frozen=True)
class EchoNoiseConfig:
    frequency_bin_count: int = 513
    minimum_window_count: int = 10
    minimum_window_frames: int = 15
    temporal_smoothing: float = 0.9
    local_power_threshold: float = 1.67
    minimum_bias: float = 1.66
    instantaneous_power_threshold: float = 4.6
    conditional_power_threshold: float = 3.0
    noise_smoothing: float = 0.85
    prior_smoothing: float = 0.95
    minimum_prior_ratio: float = 10**(-55/10)
    speech_bins: tuple = (7, 137)
    speech_probability_threshold: float = 0.3
    absence_bias: float = 0.1
    gain_floor: float = 0.01
    noise_estimate_scale: float = 2.0

    def __post_init__(self):
        for name in ('frequency_bin_count', 'minimum_window_count', 'minimum_window_frames'):
            positive_integer(getattr(self, name), name)
        for name in ('temporal_smoothing', 'noise_smoothing', 'prior_smoothing',
                     'speech_probability_threshold', 'absence_bias', 'gain_floor'):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        for name in ('local_power_threshold', 'minimum_bias', 'instantaneous_power_threshold',
                     'minimum_prior_ratio', 'noise_estimate_scale'):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).eps)
        finite_scalar(self.conditional_power_threshold, 'conditional_power_threshold', minimum=1+np.finfo(float).eps)
        if not 0 <= self.speech_bins[0] < self.speech_bins[1] <= self.frequency_bin_count:
            raise ValueError('speech_bins must be a nonempty interval inside the spectrum')


class ConditionalNoiseTracker:
    def __init__(self, config=EchoNoiseConfig()):
        self.config = config
        self.reset()

    def reset(self):
        bins = self.config.frequency_bin_count
        self.frame_count = self.window_position = 0
        for name in ('smoothed_power', 'conditional_power', 'minimum_power', 'active_minimum',
                     'conditional_minimum', 'conditional_active_minimum', 'noise_power', 'previous_echo_power'):
            setattr(self, name, np.zeros(bins))
        self.minimum_history = np.zeros((bins, self.config.minimum_window_count))
        self.conditional_history = self.minimum_history.copy()

    def process_frame(self, residual_power, echo_power, previous_prior_term, *, previous_noise_power=None):
        config = self.config
        values = [real_array(v, name, 1) for v, name in ((residual_power, 'residual_power'),
                  (echo_power, 'echo_power'), (previous_prior_term, 'previous_prior_term'))]
        if any(v.shape != (config.frequency_bin_count,) or np.any(v < 0) for v in values):
            raise ValueError('power and prior arrays must be nonnegative vectors matching the spectrum')
        power, echo, prior_term = values
        self.frame_count += 1
        if self.frame_count == 1:
            noise = power
        elif previous_noise_power is None:
            noise = 2*self.noise_power+self.previous_echo_power
            self.previous_echo_power = echo.copy()
        else:
            noise = previous_noise_power
        posterior = power/np.maximum(noise, 1e-10)
        prior = np.maximum(config.prior_smoothing*prior_term+(1-config.prior_smoothing)*np.maximum(posterior-1, 0),
                           config.minimum_prior_ratio)
        argument = posterior*prior/(1+prior)
        smooth = np.convolve(power, [.25, .5, .25], mode='full')[1:len(power)+1]
        if self.frame_count == 1:
            self.smoothed_power, self.conditional_power, self.noise_power = smooth.copy(), smooth.copy(), power.copy()
        else:
            self.smoothed_power = config.temporal_smoothing*self.smoothed_power+(1-config.temporal_smoothing)*smooth
        initializing = self.frame_count == 1 or self.frame_count < config.minimum_window_frames
        if initializing:
            self.minimum_power = self.smoothed_power.copy()
            self.active_minimum = self.smoothed_power.copy()
        else:
            self.minimum_power = np.minimum(self.minimum_power, self.smoothed_power)
            self.active_minimum = np.minimum(self.active_minimum, self.smoothed_power)
        inactive = ((power < config.instantaneous_power_threshold*config.minimum_bias*self.minimum_power)
                    & (self.smoothed_power < config.local_power_threshold*config.minimum_bias*self.minimum_power)).astype(float)
        weight = np.convolve(inactive, [.25, .5, .25], mode='full')[1:len(power)+1]
        weighted = np.convolve(inactive*power, [.25, .5, .25], mode='full')[1:len(power)+1]
        conditional = np.divide(weighted, weight, out=self.conditional_power.copy(), where=weight>0)
        if initializing:
            self.conditional_power = self.smoothed_power.copy()
            self.conditional_minimum = self.conditional_power.copy()
            self.conditional_active_minimum = self.conditional_power.copy()
        else:
            self.conditional_power = config.temporal_smoothing*self.conditional_power+(1-config.temporal_smoothing)*conditional
            self.conditional_minimum = np.minimum(self.conditional_minimum, self.conditional_power)
            self.conditional_active_minimum = np.minimum(self.conditional_active_minimum, self.conditional_power)
        instantaneous_ratio = power/config.minimum_bias/np.maximum(self.conditional_minimum, 1e-10)
        smoothed_ratio = self.smoothed_power/config.minimum_bias/np.maximum(self.conditional_minimum, 1e-10)
        probability = np.zeros_like(power)
        selected = ((instantaneous_ratio > 1) & (instantaneous_ratio < config.conditional_power_threshold)
                    & (smoothed_ratio < config.local_power_threshold))
        absence = (config.conditional_power_threshold-instantaneous_ratio[selected])/(config.conditional_power_threshold-1)
        probability[selected] = 1/(1+absence/(1-absence)*(1+prior[selected])*np.exp(-argument[selected]))
        probability[(instantaneous_ratio >= config.conditional_power_threshold)
                    | (smoothed_ratio >= config.local_power_threshold)] = 1
        smoothing = config.noise_smoothing+(1-config.noise_smoothing)*probability
        self.noise_power = smoothing*self.noise_power+(1-smoothing)*power
        self.window_position += 1
        if self.window_position == config.minimum_window_frames:
            self.window_position = 0
            if self.frame_count == config.minimum_window_frames:
                self.minimum_history[:] = self.smoothed_power[:, None]
                self.conditional_history[:] = self.conditional_power[:, None]
            else:
                self.minimum_history[:, :-1] = self.minimum_history[:, 1:]
                self.minimum_history[:, -1] = self.active_minimum
                self.minimum_power = self.minimum_history.min(axis=1)
                self.active_minimum = self.smoothed_power.copy()
                self.conditional_history[:, :-1] = self.conditional_history[:, 1:]
                self.conditional_history[:, -1] = self.conditional_active_minimum
                self.conditional_minimum = self.conditional_history.min(axis=1)
                self.conditional_active_minimum = self.conditional_power.copy()
        return self.noise_power.copy()


class SpeechPresenceGain:
    def __init__(self, config=EchoNoiseConfig()):
        self.config = config
        bins = config.frequency_bin_count
        self.absence_gain = config.gain_floor*np.maximum(.25+np.arange(bins)*.25/max(bins-1, 1), .3)
        self.reset()

    def reset(self):
        self.previous_prior_term = np.ones(self.config.frequency_bin_count)
        self.posterior_ratio = np.zeros(self.config.frequency_bin_count)
        self.noise_power = np.zeros(self.config.frequency_bin_count)

    def process_frame(self, residual_power, noise_power, echo_power):
        config = self.config
        self.noise_power = config.noise_estimate_scale*noise_power+echo_power
        posterior = residual_power/np.maximum(self.noise_power, 1e-10)
        prior = np.maximum(config.prior_smoothing*self.previous_prior_term
                           +(1-config.prior_smoothing)*np.maximum(posterior-1, 0), config.minimum_prior_ratio)
        argument = posterior*prior/(1+prior)
        absence = np.clip((4.6-posterior)/3.6, 0, 1)
        start, stop = config.speech_bins
        if np.mean(1-absence[start:stop]) < config.speech_probability_threshold:
            absence = np.clip(np.maximum(absence, absence.mean())+config.absence_bias, 0, 1)
        probability = np.zeros_like(posterior)
        selected = absence < .9
        probability[selected] = 1/(1+absence[selected]/(1-absence[selected])*(1+prior[selected])*np.exp(-argument[selected]))
        present_gain = np.ones_like(posterior)
        selected = argument > 5
        present_gain[selected] = prior[selected]/(1+prior[selected])
        selected = (argument <= 5) & (argument > 0)
        present_gain[selected] = prior[selected]/(1+prior[selected])*np.exp(.5*exponential_integral_approximation(argument[selected]))
        gain = present_gain**probability*self.absence_gain**(1-probability)
        self.previous_prior_term = present_gain**2*posterior
        self.posterior_ratio = posterior
        return gain
