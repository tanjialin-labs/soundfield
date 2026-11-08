"""Auditory modulation-energy ratios with explicit preprocessing and filter modes."""

from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter, hilbert, sosfilt, sosfreqz
from ..contracts import real_array, positive_integer


def trim_speech_intervals(audio, sample_rate_hz):
    """Select samples above a peak-relative 50 dB threshold with 50 ms gaps."""
    samples = real_array(audio, 'audio', 1)
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    indices = np.flatnonzero(samples**2 > np.max(np.abs(samples), initial=0)**2/1e5)
    if not len(indices):
        raise ValueError('speech preprocessing requires non-silent input')
    gaps = np.flatnonzero(np.diff(indices) > .05*sample_rate_hz)
    if not len(gaps):
        return samples[indices[0]:indices[-1]+1].copy()
    boundaries = np.column_stack((indices[gaps], indices[gaps+1])).ravel()
    parts = [samples[indices[0]:boundaries[0]+1]]
    parts.extend(samples[boundaries[index]:boundaries[index+1]+1] for index in range(1, len(boundaries)-1, 2))
    parts.append(samples[boundaries[-1]:indices[-1]+1])
    return np.concatenate(parts)


def active_speech_level_db(audio, sample_rate_hz):
    """Measure a double-smoothed envelope with 200 ms hangover and 15.9 dB margin."""
    samples = real_array(audio, 'audio', 1)
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    if not len(samples) or not np.any(samples):
        raise ValueError('active speech level requires non-silent input')
    if samples.max() > 1:
        samples = samples/32768
    memory = np.exp(-1/(.03*sample_rate_hz))
    envelope = lfilter([1-memory], [1, -memory], np.abs(samples))
    envelope = lfilter([1-memory], [1, -memory], envelope)
    hangover = int(np.floor(.2*sample_rate_hz+.5))
    positions = np.arange(len(samples))
    thresholds = .5**np.arange(15, 0, -1)
    counts = []
    for threshold in thresholds:
        latest = np.maximum.accumulate(np.where(envelope >= threshold, positions, -len(samples)-hangover))
        counts.append(np.sum(positions-latest <= hangover))
    counts = np.asarray(counts)
    levels = np.full(15, -100.)
    levels[counts>0] = 10*np.log10(np.sum(samples**2)/counts[counts>0])
    threshold_db = 20*np.log10(thresholds)
    candidates = np.flatnonzero(levels-threshold_db <= 15.9)
    if not len(candidates):
        return -100.
    index = candidates[0]
    if index == 0:
        return float(levels[index])
    upper, lower = levels[index], levels[index-1]
    upper_threshold, lower_threshold = threshold_db[index], threshold_db[index-1]
    if abs(upper-upper_threshold-15.9)<.1:
        return float(upper)
    if abs(lower-lower_threshold-15.9)<.1:
        return float(lower)
    middle, middle_threshold = (upper+lower)/2, (upper_threshold+lower_threshold)/2
    tolerance, iteration = .1, 1
    while abs(middle-middle_threshold-15.9) > tolerance:
        iteration += 1
        if iteration > 20:
            tolerance *= 1.1
        endpoint, endpoint_threshold = (upper, upper_threshold) if middle-middle_threshold-15.9>tolerance else (lower, lower_threshold)
        middle, middle_threshold = (endpoint+middle)/2, (endpoint_threshold+middle_threshold)/2
    return float(middle)


def _auditory_filters(sample_rate_hz, channel_count, lowest_frequency_hz, implant):
    offset = 9.26449*24.7
    centers = -offset+np.exp(np.arange(1, channel_count+1)*(-np.log(sample_rate_hz/2+offset)+np.log(lowest_frequency_hz+offset))/channel_count)*(sample_rate_hz/2+offset)
    widths = np.array([1000, 875, 750, 625, 625, 500, 500, 375, 375, 250, 250, 250, 250, 125, 125, 125, 125, 125, 125, 125, 125, 125]) if implant else centers/9.26449+24.7
    sections = []
    period = 1/sample_rate_hz
    for center, width in zip(centers, widths):
        decay = np.exp(-1.019*2*np.pi*width*period)
        phase = 2*np.pi*center*period
        coefficients = []
        for factor in (np.sqrt(3+2**1.5), -np.sqrt(3+2**1.5), np.sqrt(3-2**1.5), -np.sqrt(3-2**1.5)):
            coefficients.append([period, -period*decay*(np.cos(phase)+factor*np.sin(phase)), 0,
                                 1, -2*np.cos(phase)*decay, decay**2])
        coefficients = np.asarray(coefficients)
        normalization = abs(sosfreqz(coefficients, worN=np.array([phase]))[1][0])
        coefficients[0, :3] /= normalization
        sections.append(coefficients)
    return np.asarray(sections), widths


@dataclass(frozen=True)
class ModulationEnergyResult:
    ratio: float
    energies: np.ndarray
    upper_modulation_band_count: int
    active_level_db: float
    analyzed_sample_count: int


def modulation_energy_ratio(audio, sample_rate_hz, *, fast=False, normalized=False, implant=False,
                             minimum_modulation_hz=4., maximum_modulation_hz=None):
    """Compute SRMR using 23 auditory channels, or 22 implant analysis channels.

    Energies have shape (auditory channels, modulation bands, frames). The fast
    mode uses spectral filter weights and a 400 Hz envelope grid.
    """
    if sample_rate_hz not in (8000, 16000):
        raise ValueError('modulation-energy analysis supports 8000 and 16000 Hz')
    if fast and implant:
        raise ValueError('implant analysis uses time-domain auditory filters')
    samples = trim_speech_intervals(audio, sample_rate_hz)
    active_level = active_speech_level_db(samples, sample_rate_hz)
    samples = samples/10**((active_level+26)/20)
    count, minimum_frequency = (22, 150) if implant else (23, 125)
    sections, bandwidths = _auditory_filters(sample_rate_hz, count, minimum_frequency, implant)
    envelope_rate = 400 if fast else sample_rate_hz
    maximum_modulation_hz = maximum_modulation_hz if maximum_modulation_hz is not None else (30. if normalized else 64. if implant else 128.)
    if not 0 < minimum_modulation_hz < maximum_modulation_hz < envelope_rate/2:
        raise ValueError('modulation frequencies must fit below the envelope Nyquist frequency')
    if fast:
        window_length, hop = round(.01*sample_rate_hz), round(.0025*sample_rate_hz)
        if len(samples) < window_length:
            raise ValueError('speech interval is too short for spectral auditory filtering')
        fft_length = 2**int(np.ceil(np.log2(2*.01*sample_rate_hz)))
        frames = np.lib.stride_tricks.sliding_window_view(samples, window_length)[::hop]
        spectrum = np.abs(np.fft.rfft(frames*np.hanning(window_length), n=fft_length, axis=1)).T
        frequency = np.arange(fft_length//2+1)*2*np.pi/fft_length
        weights = np.stack([np.abs(sosfreqz(section, worN=frequency)[1]) for section in sections])
        envelopes = weights@spectrum/fft_length
    else:
        auditory = np.stack([sosfilt(section, samples) for section in sections])
        envelopes = np.abs(hilbert(auditory, axis=1))
    centers = np.geomspace(minimum_modulation_hz, maximum_modulation_hz, 8)
    length, hop = int(np.ceil(.256*envelope_rate)), int(np.ceil(.064*envelope_rate))
    frame_count = int(np.ceil(envelopes.shape[1]/hop))
    energies = np.empty((count, 8, frame_count))
    window = np.hamming(length)
    for band, center in enumerate(centers):
        frequency = np.tan(np.pi*center/envelope_rate)
        width = frequency/2
        denominator = np.array([1+width+frequency**2, 2*frequency**2-2, 1-width+frequency**2])
        numerator = np.array([width, 0, -width])/denominator[0]
        denominator /= denominator[0]
        modulated = lfilter(numerator, denominator, envelopes, axis=1)
        padded = np.pad(modulated, ((0, 0), (length-hop, frame_count*hop-envelopes.shape[1])))
        frames = np.lib.stride_tricks.sliding_window_view(padded, length, axis=1)[:, ::hop]
        energies[:, band] = np.sum((frames*window)**2, axis=2)
    if normalized:
        peak = np.max(np.mean(energies, axis=0))
        energies = np.clip(energies, peak*.001, peak)
    average = energies.mean(axis=2)
    total = average.sum()
    if total <= 0:
        raise ValueError('modulation energy is zero')
    cumulative = np.cumsum(average.sum(axis=1)[::-1])*100/total
    width = bandwidths[::-1][np.flatnonzero(cumulative>90)[0]]
    cutoffs = centers-np.tan(np.pi*centers/envelope_rate)/2*envelope_rate/(2*np.pi)
    upper_count = max(5, min(8, int(np.searchsorted(cutoffs, width, side='left'))))
    denominator = average[:, 4:upper_count].sum()
    if denominator == 0:
        raise ValueError('high modulation-band energy is zero')
    return ModulationEnergyResult(float(average[:, :4].sum()/denominator), energies, upper_count, active_level, len(samples))
