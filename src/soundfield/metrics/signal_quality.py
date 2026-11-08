"""Explicit spectral-distance, speech-quality and intelligibility measures."""

from dataclasses import dataclass
import numpy as np
from ..contracts import real_array, positive_integer


def _mono_pair(reference, processed, sample_rate_hz):
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    reference = real_array(reference, 'reference', 1)
    processed = real_array(processed, 'processed', 1)
    if not len(reference) or not len(processed):
        raise ValueError('signals must not be empty')
    return reference, processed


def _windowed_spectrum(samples, length, hop):
    if len(samples) < length:
        raise ValueError('signal is shorter than one analysis frame')
    frames = np.lib.stride_tricks.sliding_window_view(samples, length)[::hop]
    return np.fft.rfft(frames*np.hamming(length), axis=1).T


def _remove_inactive_segments(reference, processed, rate):
    length, hop = int(np.floor(.025*rate+.5)), int(.010*rate)
    initial = int((.25*rate-length)/hop+1)
    clean = _windowed_spectrum(reference, length, hop)
    altered = _windowed_spectrum(processed, length, hop)
    if initial < 1 or min(clean.shape[1], altered.shape[1]) < initial:
        raise ValueError('activity detection requires at least 0.25 seconds')
    magnitude = np.abs(clean)
    noise = magnitude[:, :initial].mean(axis=1)
    counter, accepted = 100, []
    for frame in range(initial, min(clean.shape[1], altered.shape[1])):
        distance = np.mean(np.maximum(20*np.log10(np.maximum(magnitude[:, frame], 1e-300)/np.maximum(noise, 1e-300)), 0))
        counter = counter+1 if distance < 2.5 else 0
        if counter <= int(.08*10000/hop):
            accepted.append(frame)
        else:
            noise = (15*noise+magnitude[:, frame])/16
    if not accepted:
        raise ValueError('no speech-active segments remain')
    def assemble(spectrum):
        output = np.zeros((len(accepted)-1)*hop+length)
        for index, frame in enumerate(accepted):
            output[index*hop:index*hop+length] += np.fft.irfft(spectrum[:, frame], n=length)
        return output
    return assemble(clean), assemble(altered)


def log_spectral_distance(reference, processed, sample_rate_hz=10000, *, frequency_range_hz=None,
                          remove_silence=False, magnitude_floor=None):
    """Mean frame RMS of natural-log magnitudes after independent L2 scaling.

    Analysis uses symmetric Hamming windows of 25 ms, a 40% hop, and only
    complete frames. With no floor, zero-energy signals have no defined score.
    """
    reference, processed = _mono_pair(reference, processed, sample_rate_hz)
    length = int(np.floor(.025*sample_rate_hz+.5))
    hop = int(.4*length)
    if length < 2 or hop < 1:
        raise ValueError('sample rate is too low for the analysis grid')
    if remove_silence:
        reference, processed = _remove_inactive_segments(reference, processed, sample_rate_hz)
    count = min(len(reference), len(processed))
    reference, processed = reference[:count], processed[:count]
    norms = np.linalg.norm(reference), np.linalg.norm(processed)
    if min(norms) == 0:
        raise ValueError('log spectral distance requires two nonzero-energy signals')
    clean = np.abs(_windowed_spectrum(reference/norms[0], length, hop))
    altered = np.abs(_windowed_spectrum(processed/norms[1], length, hop))
    lower, upper = frequency_range_hz or (0, sample_rate_hz/2)
    if not 0 <= lower <= upper <= sample_rate_hz/2:
        raise ValueError('frequency range must lie inside the Nyquist interval')
    start, stop = np.floor(np.array([lower, upper])/(sample_rate_hz/2)*(len(clean)-1)+.5).astype(int)
    clean, altered = clean[start:stop+1], altered[start:stop+1]
    if magnitude_floor is not None:
        if not np.isfinite(magnitude_floor) or magnitude_floor <= 0:
            raise ValueError('magnitude_floor must be positive and finite')
        clean, altered = np.maximum(clean, magnitude_floor), np.maximum(altered, magnitude_floor)
    elif np.any(clean == 0) or np.any(altered == 0):
        raise ValueError('zero spectral magnitudes require an explicit positive magnitude_floor')
    return float(np.mean(np.sqrt(np.mean((np.log(clean)-np.log(altered))**2, axis=0))))


@dataclass(frozen=True)
class PerceptualSpeechQuality:
    listening_quality_mos: float
    raw_mos: float | None
    mode: str


def perceptual_speech_quality(reference, processed, sample_rate_hz, *, mode='narrowband'):
    """Evaluate PESQ; raw MOS is available for narrowband, MOS-LQO for both modes."""
    reference, processed = _mono_pair(reference, processed, sample_rate_hz)
    if mode not in ('narrowband', 'wideband') or sample_rate_hz not in (8000, 16000) or mode == 'wideband' and sample_rate_hz != 16000:
        raise ValueError('PESQ needs 8/16 kHz narrowband or 16 kHz wideband input')
    if min(len(reference), len(processed)) < sample_rate_hz//4 or min(np.max(np.abs(reference)), np.max(np.abs(processed))) == 0:
        raise ValueError('PESQ requires at least 0.25 seconds of non-silent input')
    try:
        from pesq import pesq
    except ImportError as exception:
        raise ImportError('Speech-quality measurement requires soundfield[evaluation]') from exception
    score = float(pesq(sample_rate_hz, reference, processed, 'nb' if mode == 'narrowband' else 'wb'))
    raw = float((4.6607-np.log(4/(score-.999)-1))/1.4945) if mode == 'narrowband' else None
    return PerceptualSpeechQuality(score, raw, mode)


def speech_intelligibility(reference, processed, sample_rate_hz, *, extended=False, window='hann'):
    """Evaluate STOI on a 10 kHz grid with an explicit Hann window convention.

    'hann' includes zero endpoints; 'hann_inset' removes the endpoints of a
    window two samples longer. Both analysis and activity selection use it.
    """
    reference, processed = _mono_pair(reference, processed, sample_rate_hz)
    if len(reference) != len(processed):
        raise ValueError('intelligibility signals must have equal sample counts')
    if len(reference)/sample_rate_hz < .4 or min(np.max(np.abs(reference)), np.max(np.abs(processed))) == 0:
        raise ValueError('intelligibility needs at least 0.4 seconds of non-silent audio')
    try:
        from pystoi import utils
    except ImportError as exception:
        raise ImportError('Intelligibility measurement requires soundfield[evaluation]') from exception
    if window not in ('hann', 'hann_inset'):
        raise ValueError('window must be hann or hann_inset')
    if sample_rate_hz != 10000:
        reference = utils.resample_oct(reference, 10000, sample_rate_hz)
        processed = utils.resample_oct(processed, 10000, sample_rate_hz)
    taper = np.hanning(256) if window == 'hann' else np.hanning(258)[1:-1]
    def frames(samples):
        return np.array([samples[start:start+256]*taper for start in range(0, len(samples)-256, 128)])
    clean_frames, processed_frames = frames(reference), frames(processed)
    levels = 20*np.log10(np.maximum(np.linalg.norm(clean_frames, axis=1), 1e-300))
    active = levels > levels.max()-40
    def compact(values):
        chosen = values[active]
        output = np.zeros((len(chosen)-1)*128+256)
        for index, current in enumerate(chosen):
            output[index*128:index*128+256] += current
        return output
    octave_bands = utils.thirdoct(10000, 512, 15, 150)[0]
    def bands(values):
        spectrum = np.fft.rfft(frames(compact(values)), n=512, axis=1).T
        return np.sqrt(octave_bands@np.abs(spectrum)**2)
    clean_bands, processed_bands = bands(clean_frames), bands(processed_frames)
    if clean_bands.shape[1] < 30:
        raise ValueError('at least 30 speech-active analysis frames are required')
    clean_segments = np.array([clean_bands[:, end-30:end] for end in range(30, clean_bands.shape[1]+1)])
    processed_segments = np.array([processed_bands[:, end-30:end] for end in range(30, processed_bands.shape[1]+1)])
    epsilon = np.finfo(float).eps
    if extended:
        def normalize(values):
            values = values-values.mean(axis=2, keepdims=True)
            values /= np.maximum(np.linalg.norm(values, axis=2, keepdims=True), epsilon)
            values -= values.mean(axis=1, keepdims=True)
            return values/np.maximum(np.linalg.norm(values, axis=1, keepdims=True), epsilon)
        return float(np.sum(normalize(clean_segments)*normalize(processed_segments))/(30*len(clean_segments)))
    scale = np.linalg.norm(clean_segments, axis=2, keepdims=True)/np.maximum(np.linalg.norm(processed_segments, axis=2, keepdims=True), epsilon)
    clipped = np.minimum(processed_segments*scale, clean_segments*(1+10**(.75)))
    clean_segments -= clean_segments.mean(axis=2, keepdims=True)
    clipped -= clipped.mean(axis=2, keepdims=True)
    clean_segments /= np.maximum(np.linalg.norm(clean_segments, axis=2, keepdims=True), epsilon)
    clipped /= np.maximum(np.linalg.norm(clipped, axis=2, keepdims=True), epsilon)
    return float(np.mean(np.sum(clean_segments*clipped, axis=2)))
