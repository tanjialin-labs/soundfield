"""Channel sensitivity balance and microphone energy acceptance."""

from dataclasses import dataclass

import numpy as np

from ..contracts import audio_array, finite_scalar, positive_integer
from ..transform import FourierConfig, ShortTimeFourierTransform


@dataclass(frozen=True)
class ChannelCalibration:
    amplitude_gains: np.ndarray
    segment_energy_db: np.ndarray
    selected_segment_indices: np.ndarray
    trimmed_spread_db: float
    is_accepted: bool


def calibrate_channel_levels(audio, sample_rate_hz, *, segment_start_samples=None, segment_length_samples=None,
                             maximum_spread_db=6., gain_bounds=(.7, 1.4), energy_floor=.0001):
    """Estimate gains from segments ranked by their inter-channel energy spread."""
    samples = audio_array(audio)
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    starts = ((np.arange(5)*4+3)*sample_rate_hz-1 if segment_start_samples is None else np.asarray(segment_start_samples))
    length = 2*sample_rate_hz+1 if segment_length_samples is None else segment_length_samples
    positive_integer(length, 'segment_length_samples')
    finite_scalar(maximum_spread_db, 'maximum_spread_db', minimum=0)
    finite_scalar(energy_floor, 'energy_floor', minimum=np.finfo(float).tiny)
    minimum_gain, maximum_gain = gain_bounds
    finite_scalar(minimum_gain, 'minimum_gain', minimum=0)
    finite_scalar(maximum_gain, 'maximum_gain', minimum=minimum_gain)
    if starts.ndim != 1 or len(starts) < 3 or starts.dtype.kind not in 'iu' or np.any(starts < 0) or np.any(starts+length > len(samples)):
        raise ValueError('at least three complete nonnegative calibration segments are required')
    energy = np.stack([10*np.log10(np.maximum(np.sum(samples[start:start+length]**2, axis=0), energy_floor)) for start in starts], axis=1)
    spread = np.ptp(energy, axis=0)
    selected = np.argsort(spread, kind='stable')[1:-1]
    trimmed = float(np.mean(spread[selected]))
    accepted = trimmed < maximum_spread_db
    gains = np.ones(samples.shape[1])
    if accepted:
        mean_energy = np.mean(energy[:, selected], axis=1)
        gains = np.clip(10**((np.mean(mean_energy)-mean_energy)/20), minimum_gain, maximum_gain)
    return ChannelCalibration(gains, energy, selected, trimmed, accepted)


@dataclass(frozen=True)
class MicrophoneEnergyResult:
    band_level_db: np.ndarray
    accepted_channels: np.ndarray
    smoothed_spectral_power: np.ndarray


def inspect_microphone_energy(audio, *, frame_length_samples=512, start_frame_index=0, frequency_interval=(1, 128),
                              current_frame_weight=.1, lower_level_db=-80., upper_level_db=-10.):
    """Check the mean per-bin decibel power after exponential frame smoothing."""
    samples = audio_array(audio)
    positive_integer(frame_length_samples, 'frame_length_samples')
    if frame_length_samples % 2:
        raise ValueError('frame length must be even')
    if not isinstance(start_frame_index, (int, np.integer)) or start_frame_index < 0:
        raise ValueError('start_frame_index must be nonnegative')
    finite_scalar(current_frame_weight, 'current_frame_weight', minimum=0, maximum=1)
    finite_scalar(lower_level_db, 'lower_level_db')
    finite_scalar(upper_level_db, 'upper_level_db', minimum=lower_level_db)
    hop = frame_length_samples//2
    low, high = frequency_interval
    if not 0 <= low < high <= hop+1:
        raise ValueError('frequency interval must fit the one-sided spectrum')
    spectrum = ShortTimeFourierTransform(FourierConfig(frame_length_samples, hop, 'hann')).analyze(samples)
    if start_frame_index >= spectrum.shape[1]:
        raise ValueError('recording does not contain the requested inspection frames')
    power = np.zeros((hop+1, samples.shape[1]))
    for frame in range(start_frame_index, spectrum.shape[1]):
        power = current_frame_weight*(np.abs(spectrum[:, frame])/hop)**2+(1-current_frame_weight)*power
    with np.errstate(divide='ignore'):
        levels = np.mean(10*np.log10(power[low:high]), axis=0)
    accepted = (levels > lower_level_db)&(levels < upper_level_db)
    return MicrophoneEnergyResult(levels, accepted, power)
