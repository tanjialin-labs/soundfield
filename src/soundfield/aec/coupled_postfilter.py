"""Coupled noise and residual-echo suppression with shared gain history."""

from dataclasses import dataclass

import numpy as np

from ..contracts import audio_array, finite_scalar, positive_integer
from ..numerics import spectral_amplitude_gain
from .audio_output import EchoAudioSynthesizer


@dataclass(frozen=True)
class CoupledPostfilterConfig:
    hop_length_samples: int = 512
    coherence_memory: float = .984
    leakage_scale: float = 1.7
    noise_floor: float = 1e-7
    posterior_weight: float = .02
    minimum_prior_ratio: float = .0032
    enable_noise_suppression: bool = True
    minimum_window_frames: int = 2

    def __post_init__(self):
        positive_integer(self.hop_length_samples, 'hop_length_samples')
        positive_integer(self.minimum_window_frames, 'minimum_window_frames')
        for name in ('coherence_memory', 'posterior_weight'):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        finite_scalar(self.leakage_scale, 'leakage_scale', minimum=0)
        for name in ('noise_floor', 'minimum_prior_ratio'):
            finite_scalar(getattr(self, name), name, minimum=np.finfo(float).eps)


class CoupledEchoPostfilter:
    def __init__(self, config=CoupledPostfilterConfig()):
        self.config = config
        self.synthesis = EchoAudioSynthesizer(config.hop_length_samples, enable_deemphasis=False)
        self.reset()

    def reset(self):
        hop = self.config.hop_length_samples
        self.frame_count = 0
        self.residual_history = np.zeros((2*hop, 1))
        self.echo_history = np.zeros_like(self.residual_history)
        self.cross_power = np.zeros(hop+1, complex)
        for name in ('echo_power', 'residual_power', 'previous_output_power', 'tracked_power',
                     'minimum_power', 'recent_minimum', 'noise_power', 'presence', 'gains', 'estimated_echo_power'):
            setattr(self, name, np.zeros(hop+1))
        self.output_spectrum = np.zeros((hop+1, 1), complex)
        self.synthesis.reset()

    def _remove_noise(self, spectrum, power):
        if not self.frame_count:
            self.tracked_power = power.copy()
            self.minimum_power = power.copy()
            self.recent_minimum = power.copy()
            self.noise_power = power.copy()
        else:
            self.tracked_power = .6*self.tracked_power+.4*power
            if (self.frame_count+1) % self.config.minimum_window_frames == 0:
                self.minimum_power = np.minimum(self.recent_minimum, self.tracked_power)
                self.recent_minimum = self.tracked_power.copy()
            else:
                self.minimum_power = np.minimum(self.minimum_power, self.tracked_power)
                self.recent_minimum = np.minimum(self.recent_minimum, self.tracked_power)
            self.presence = .1*self.presence+.9*(self.tracked_power > 5*self.minimum_power)
            memory = .8+.2*self.presence
            self.noise_power = memory*self.noise_power+(1-memory)*power
        noise = np.maximum(self.noise_power, np.finfo(float).tiny)
        posterior = np.minimum(power/noise, 40)
        prior = (.3*np.maximum(posterior-1, 0)+.7 if not self.frame_count else
                 np.maximum(.3*np.maximum(posterior-1, 0)+.7*self.previous_output_power/noise, 10**(-2.5)))
        ratio = prior/(1+prior)
        gains = np.maximum(10**(-1.2), ratio*spectral_amplitude_gain(ratio*posterior))
        return spectrum*gains

    def process_block(self, residual_audio, echo_audio):
        residual, estimated = audio_array(residual_audio), audio_array(echo_audio)
        config, hop = self.config, self.config.hop_length_samples
        if residual.shape != (hop, 1) or estimated.shape != residual.shape:
            raise ValueError('residual and echo audio must contain one hop and one channel')
        for history, audio in ((self.residual_history, residual), (self.echo_history, estimated)):
            history[:hop] = history[hop:]
            history[hop:] = audio
        residual = np.fft.rfft(self.residual_history[:, 0]*self.synthesis.window)
        estimated = np.fft.rfft(self.echo_history[:, 0]*self.synthesis.window)
        power, echo = np.abs(residual)**2, np.abs(estimated)**2
        if config.enable_noise_suppression:
            residual = self._remove_noise(residual, power)
        memory = config.coherence_memory
        self.cross_power = memory*self.cross_power+(1-memory)*estimated*residual.conj()
        self.echo_power = memory*self.echo_power+(1-memory)*echo
        self.residual_power = memory*self.residual_power+(1-memory)*np.abs(residual)**2
        ratio = np.minimum(1, np.divide(self.residual_power, self.echo_power,
                                      out=np.zeros_like(power), where=self.echo_power > 0))
        denominator = np.sqrt(self.residual_power*self.echo_power)
        coherence = np.divide(np.abs(self.cross_power), denominator, out=np.zeros_like(power), where=denominator > 0)
        noise = echo*np.minimum(1, config.leakage_scale*ratio*coherence)+config.noise_floor
        posterior = np.minimum(power/noise, 40)
        prior = np.maximum(config.posterior_weight*np.maximum(posterior-1, 0)
                           +(1-config.posterior_weight)*self.previous_output_power/noise, config.minimum_prior_ratio)
        ratio = prior/(1+prior)
        self.gains = np.minimum(1, ratio*spectral_amplitude_gain(ratio*posterior))
        self.output_spectrum = (residual*self.gains)[:, None]
        self.previous_output_power = np.abs(self.output_spectrum[:, 0])**2
        self.estimated_echo_power = echo
        self.frame_count += 1
        return self.synthesis.process_frame(self.output_spectrum)

    def flush(self):
        return self.synthesis.flush()
