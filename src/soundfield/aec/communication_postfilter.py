"""Residual echo suppression with speech-presence gains and noise reinjection."""

from dataclasses import dataclass, field

import numpy as np

from ..contracts import positive_integer, nonnegative_integer, finite_scalar, spectral_frame, unit_interval_array
from .noise_tracking import EchoNoiseConfig, ConditionalNoiseTracker, SpeechPresenceGain


@dataclass(frozen=True)
class CommunicationPostfilterConfig:
    noise: EchoNoiseConfig = field(default_factory=EchoNoiseConfig)
    coherence_smoothing: float = 0.96
    leakage_scale: float = 2.0
    minimum_leakage: float = 0.1
    leakage_regularization: float = 1e-9
    activity_history_frames: int = 25
    activity_threshold: float = 0.03
    reference_energy_threshold: float = 0.03
    reference_energy_smoothing: float = 0.8
    initial_reference_energy: float = 0.1
    enable_residual_gate: bool = True
    residual_energy_smoothing: float = 0.9
    residual_energy_ratio: float = 0.3
    double_talk_energy_threshold: float = 0.03
    residual_energy_intervals: tuple = ((1, 2), (2, 512))
    mean_gain_scale: float = 0.1
    reference_threshold_start: float = 20.0
    reference_threshold_end: float = 6.0
    reference_threshold_bin_count: int = 67
    highpass_bin_count: int = 1
    reference_channel_index: int = 0
    inactive_gain_floor: float = 0.31
    noise_output_scale: float = 1.4
    reference_amplitude_threshold: float = 200.0
    noise_feedback: str = 'lagged_echo'
    inactive_gain_policy: str = 'wiener'
    limit_noise_reinjection: bool = False

    def __post_init__(self):
        if not isinstance(self.limit_noise_reinjection, bool):
            raise ValueError("limit_noise_reinjection must be boolean")
        positive_integer(self.activity_history_frames, 'activity_history_frames')
        positive_integer(self.reference_threshold_bin_count, 'reference_threshold_bin_count')
        nonnegative_integer(self.reference_channel_index, 'reference_channel_index')
        for name in ('coherence_smoothing', 'activity_threshold', 'reference_energy_smoothing',
                     'residual_energy_smoothing', 'inactive_gain_floor'):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        for name in ('leakage_scale', 'minimum_leakage', 'reference_energy_threshold', 'initial_reference_energy',
                     'residual_energy_ratio', 'double_talk_energy_threshold', 'mean_gain_scale',
                     'reference_threshold_start', 'reference_threshold_end', 'noise_output_scale', 'reference_amplitude_threshold'):
            finite_scalar(getattr(self, name), name, minimum=0)
        finite_scalar(self.leakage_regularization, 'leakage_regularization', minimum=np.finfo(float).eps)
        bins = self.noise.frequency_bin_count
        if self.reference_threshold_bin_count >= bins or not 0 <= self.highpass_bin_count <= bins:
            raise ValueError('reference and highpass intervals must fit inside the spectrum')
        if any(not 0 <= start < stop <= bins for start, stop in self.residual_energy_intervals):
            raise ValueError('residual energy intervals must fit inside the spectrum')
        if self.noise_feedback not in ('lagged_echo', 'combined') or self.inactive_gain_policy not in ('wiener', 'bypass', 'preserve_nearend'):
            raise ValueError('unsupported noise feedback or inactive gain policy')


class CommunicationEchoPostfilter:
    def __init__(self, config=CommunicationPostfilterConfig()):
        self.config = config
        self.noise_tracker = ConditionalNoiseTracker(config.noise)
        self.gain_estimator = SpeechPresenceGain(config.noise)
        self.reference_threshold = np.r_[np.linspace(config.reference_threshold_start, config.reference_threshold_end,
                                                     config.reference_threshold_bin_count+1),
                                          np.full(config.noise.frequency_bin_count-config.reference_threshold_bin_count-1,
                                                  config.reference_threshold_end)]
        self.reset()

    def reset(self):
        bins = self.config.noise.frequency_bin_count
        self.cross_power = np.zeros(bins, complex)
        self.echo_power = np.zeros(bins)
        self.activity_history = np.zeros(self.config.activity_history_frames)
        self.reference_energy = self.config.initial_reference_energy
        self.residual_energy = 0.0
        self.gains = np.ones(bins)
        self.residual_echo_power = np.zeros(bins)
        self.noise_tracker.reset()
        self.gain_estimator.reset()

    def process_frame(self, residual_spectrum, echo_spectrum, reference_spectrum, *, speech_mask=None,
                      reference_activity_energy=None, playback_active=True):
        config = self.config
        bins = config.noise.frequency_bin_count
        residual = spectral_frame(residual_spectrum, bins, 1)[:, 0]
        estimated = spectral_frame(echo_spectrum, bins, 1)[:, 0]
        reference = np.asarray(reference_spectrum, dtype=complex)
        if (reference.ndim != 2 or reference.shape[0] != bins
                or reference.shape[1] <= config.reference_channel_index or not np.isfinite(reference).all()):
            raise ValueError('reference spectrum does not contain the configured channel')
        estimated_power, residual_power = np.abs(estimated)**2, np.abs(residual)**2
        energy = estimated_power[1:].mean() if bins > 1 else estimated_power[0]
        if reference_activity_energy is not None:
            finite_scalar(reference_activity_energy, 'reference_activity_energy', minimum=0)
            energy = reference_activity_energy
        smoothing = config.reference_energy_smoothing
        self.reference_energy = smoothing*self.reference_energy+(1-smoothing)*energy
        self.activity_history[:-1] = self.activity_history[1:]
        self.activity_history[-1] = self.reference_energy > config.reference_energy_threshold
        activity = self.activity_history.mean()
        self.cross_power = config.coherence_smoothing*self.cross_power+(1-config.coherence_smoothing)*residual*estimated.conj()
        self.echo_power = config.coherence_smoothing*self.echo_power+(1-config.coherence_smoothing)*estimated_power
        leakage = np.minimum(10, np.abs(self.cross_power)/(np.abs(self.echo_power)+config.leakage_regularization))
        self.residual_echo_power = np.maximum(config.minimum_leakage, config.leakage_scale*leakage)*estimated_power
        previous_noise = self.gain_estimator.noise_power if config.noise_feedback == 'combined' else None
        noise = self.noise_tracker.process_frame(residual_power, self.residual_echo_power,
                                                 self.gain_estimator.previous_prior_term, previous_noise_power=previous_noise)
        if speech_mask is None:
            gain = self.gain_estimator.process_frame(residual_power, noise, self.residual_echo_power)
        else:
            gain = unit_interval_array(speech_mask, 'speech_mask', (bins,)).copy()
            posterior = residual_power/np.maximum(noise+self.residual_echo_power, 1e-10)
            self.gain_estimator.previous_prior_term = gain**2*posterior
            if activity < config.activity_threshold:
                gain.fill(1)
        if config.enable_residual_gate and not (speech_mask is not None and activity < config.activity_threshold):
            power = np.abs(gain*residual)**2
            total = sum(power[start:stop].sum() for start, stop in config.residual_energy_intervals)
            self.residual_energy = config.residual_energy_smoothing*self.residual_energy+(1-config.residual_energy_smoothing)*total
            if (self.residual_energy < min(config.residual_energy_ratio*self.reference_energy,
                                          config.double_talk_energy_threshold) and activity > config.activity_threshold):
                gain = np.minimum(gain, config.mean_gain_scale*gain.mean())
        if speech_mask is None and activity < config.activity_threshold and self.reference_energy < config.reference_energy_threshold:
            if config.inactive_gain_policy == 'bypass':
                gain = np.ones(bins)
            else:
                posterior = self.gain_estimator.posterior_ratio
                reciprocal = np.divide(1, posterior, out=np.full(bins, np.inf), where=posterior>0)
                gain = np.maximum(config.inactive_gain_floor, 1-reciprocal)
                selected_reference = np.abs(reference[:, config.reference_channel_index])
                if selected_reference.sum() > config.reference_amplitude_threshold:
                    gain[selected_reference >= self.reference_threshold] = 0
        if speech_mask is None and not playback_active and config.inactive_gain_policy == 'preserve_nearend':
            self.gains.fill(1.)
            return residual[:, None].copy()
        amplitude_squared = np.abs(gain*residual)**2+config.noise_output_scale*(1-gain)*noise
        if config.limit_noise_reinjection:
            amplitude_squared = np.minimum(amplitude_squared, residual_power)
        output = np.sqrt(np.maximum(amplitude_squared, 0))*residual/np.maximum(np.abs(residual), 1e-10)
        output[:config.highpass_bin_count] = 0
        self.gains = gain.copy()
        return output[:, None]


class SpectralEchoPostfilter(CommunicationEchoPostfilter):
    """Independent spectral echo suppressor with instantaneous activity and combined noise feedback."""

    def __init__(self, config=None):
        if config is None:
            config = CommunicationPostfilterConfig(
                noise=EchoNoiseConfig(frequency_bin_count=257, speech_bins=(16, 137),
                                      speech_probability_threshold=.4, absence_bias=.5),
                leakage_scale=4, minimum_leakage=0, leakage_regularization=1e-5,
                activity_history_frames=20, reference_energy_threshold=.028,
                reference_energy_smoothing=0, initial_reference_energy=0,
                enable_residual_gate=False, residual_energy_intervals=((1, 257),),
                reference_threshold_start=3, reference_threshold_end=.2,
                reference_threshold_bin_count=66, reference_amplitude_threshold=10, noise_feedback='combined')
        super().__init__(config)
