"""Separate echo paths for playback modes with covariance-normalized adaptation."""

from dataclasses import dataclass

import numpy as np

from ..contracts import positive_integer, finite_scalar, audio_array


@dataclass(frozen=True)
class PlaybackEchoConfig:
    hop_length_samples: int = 512
    reference_count: int = 2
    partition_count: int = 8
    adaptation_rate: float = 1.0
    covariance_smoothing: float = .98
    covariance_loading: float = .15
    cross_power_scale: float = .99
    determinant_floor: float = 1e-6
    normalization_floor: float = .007340032
    update_loading_scale: float = .01
    minimum_reference_power: float = 1e-6
    mode_count: int = 2

    def __post_init__(self):
        for name in ('hop_length_samples', 'reference_count', 'partition_count', 'mode_count'):
            positive_integer(getattr(self, name), name)
        if self.reference_count > 3:
            raise ValueError('playback covariance supports one, two or three references')
        for name in ('covariance_smoothing', 'cross_power_scale'):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)
        for name in ('adaptation_rate', 'covariance_loading', 'determinant_floor', 'normalization_floor',
                     'update_loading_scale', 'minimum_reference_power'):
            finite_scalar(getattr(self, name), name, minimum=0)
        if self.normalization_floor*self.update_loading_scale <= 0:
            raise ValueError('positive update regularization is required')


@dataclass(frozen=True)
class PlaybackEchoResult:
    residual_audio: np.ndarray
    estimated_echo_audio: np.ndarray


class PlaybackAdaptiveEchoFilter:
    """Select the coefficient bank explicitly with a zero-based playback_mode."""

    def __init__(self, config=PlaybackEchoConfig()):
        self.config = config
        bins = config.hop_length_samples+1
        self.update_floor = np.maximum(1-np.arange(bins)/bins, 2/config.partition_count)*config.partition_count*config.normalization_floor
        self.reset()

    def reset(self):
        config = self.config
        hop, references = config.hop_length_samples, config.reference_count
        self.reference_history = np.zeros((2*hop, references))
        self.partition_spectra = np.zeros((hop+1, references*config.partition_count), complex)
        self.coefficient_banks = np.zeros((config.mode_count, *self.partition_spectra.shape), complex)
        self.covariance = np.zeros((hop+1, references, references), complex)
        self.inverse_covariance = np.zeros_like(self.covariance)

    def process_block(self, microphone_audio, reference_audio, *, playback_mode=0, is_reference_active=True, is_clipped=False):
        microphone, reference = audio_array(microphone_audio), audio_array(reference_audio)
        config, hop = self.config, self.config.hop_length_samples
        if microphone.shape != (hop,1) or reference.shape != (hop,config.reference_count):
            raise ValueError('audio blocks must match the configured microphone and references')
        if not isinstance(playback_mode, (int,np.integer)) or not 0 <= playback_mode < config.mode_count:
            raise ValueError('playback_mode must identify a coefficient bank')
        self.reference_history[:hop], self.reference_history[hop:] = self.reference_history[hop:].copy(), reference
        if not is_reference_active:
            return PlaybackEchoResult(microphone.copy(), np.zeros_like(microphone))
        spectrum = np.fft.rfft(self.reference_history, axis=0)
        count = config.reference_count
        self.partition_spectra[:, count:] = self.partition_spectra[:, :-count]
        self.partition_spectra[:, :count] = spectrum
        self.covariance = config.covariance_smoothing*self.covariance+(1-config.covariance_smoothing)*(spectrum.conj()[:,:,None]*spectrum[:,None,:])
        diagonal = np.diagonal(self.covariance, axis1=1,axis2=2).real
        if count == 1:
            self.inverse_covariance[:,0,0] = 1/(diagonal[:,0]+self.update_floor*config.update_loading_scale)
        else:
            cross_scale = .99 if count == 2 else config.cross_power_scale
            regularized = cross_scale*self.covariance
            index = np.arange(count)
            regularized[:,index,index] = np.maximum(diagonal+config.covariance_loading*diagonal.mean(axis=0),
                                                     self.update_floor[:,None]*config.update_loading_scale)
            determinant = np.linalg.det(regularized)
            self.inverse_covariance = np.linalg.inv(regularized)*(determinant/(determinant+config.determinant_floor))[:,None,None]
        weights = self.coefficient_banks[playback_mode]
        estimate = np.fft.irfft(np.sum(self.partition_spectra*weights,axis=1),n=2*hop)[hop:]
        residual = microphone[:,0]-estimate
        error_spectrum = np.fft.rfft(np.r_[np.zeros(hop),residual])
        cross = (self.partition_spectra.conj()*error_spectrum[:,None]).reshape(hop+1,config.partition_count,count)
        update = np.einsum('bij,btj->bti',self.inverse_covariance,cross).reshape(weights.shape)
        impulse_update = np.fft.irfft(update,n=2*hop,axis=0)
        impulse_update[hop:] = 0
        update = np.fft.rfft(impulse_update,axis=0)
        if not is_clipped and diagonal.sum(axis=1).mean()/(count*config.partition_count) > config.minimum_reference_power:
            weights += config.adaptation_rate*(1-config.covariance_smoothing)*update
        return PlaybackEchoResult(residual[:,None],estimate[:,None])
