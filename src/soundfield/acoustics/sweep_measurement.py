"""Synchronized exponential-sweep measurement and independent probe validation."""

from dataclasses import dataclass, asdict
from math import gcd

import numpy as np
from scipy import signal

from ..contracts import audio_array, positive_integer, finite_scalar
from . import _sweep_math as math_ops


@dataclass(frozen=True)
class SweepConfig:
    sample_rate_hz: int = 48000
    lower_frequency_hz: float = 100.0
    upper_frequency_hz: float = 8000.0
    sweep_seconds: float = 8.0
    repeat_count: int = 3
    probe_seconds: float = 5.0
    marker_seconds: float = 0.35
    lead_seconds: float = 0.85
    marker_gap_seconds: float = 0.65
    sweep_gap_seconds: float = 1.75
    probe_tail_seconds: float = 1.6
    post_seconds: float = 0.85

    def __post_init__(self):
        positive_integer(self.sample_rate_hz, 'sample_rate_hz')
        positive_integer(self.repeat_count, 'repeat_count')
        if self.sample_rate_hz < 8000:
            raise ValueError('sample_rate_hz must be at least 8000')
        if not 0 < self.lower_frequency_hz < self.upper_frequency_hz < .46*self.sample_rate_hz:
            raise ValueError('sweep frequencies must lie between zero and 0.46 times the sample rate')
        for name, value in asdict(self).items():
            if name.endswith('_seconds'):
                finite_scalar(value, name, minimum=0)
        if min(self.sweep_seconds, self.probe_seconds, self.marker_seconds) < .05 or self.sweep_gap_seconds <= .3:
            raise ValueError('excitation durations must be at least 0.05 s and sweep gap greater than 0.3 s')


@dataclass(frozen=True)
class SweepStimulus:
    audio: np.ndarray
    config: SweepConfig
    segments: tuple[dict, ...]

    def __post_init__(self):
        audio = audio_array(self.audio)
        if audio.shape[1] != 1:
            raise ValueError('stimulus must be mono')
        position = 0
        for segment in self.segments:
            if (segment['start'] != position or segment['end'] <= position
                    or segment['end'] > len(audio) or segment['length'] != segment['end']-position):
                raise ValueError('segments must partition the stimulus timeline')
            position = segment['end']
        kinds = [segment['type'] for segment in self.segments]
        if position != len(audio) or any(kinds.count(kind) != 1 for kind in ('start_marker', 'end_marker', 'probe')) or kinds.count('sweep') != self.config.repeat_count:
            raise ValueError('stimulus must contain both markers, a probe and the configured sweeps')
        object.__setattr__(self, 'audio', audio.copy())


def create_sweep_stimulus(config=SweepConfig(), *, probe_audio=None, probe_sample_rate_hz=None):
    """Create mono excitation; supplied probe audio is averaged, resampled and faded."""
    rate = config.sample_rate_hz
    sweep = math_ops._make_sweep(rate, config.lower_frequency_hz, config.upper_frequency_hz, config.sweep_seconds)
    if probe_audio is None:
        if probe_sample_rate_hz is not None:
            raise ValueError('probe_sample_rate_hz requires probe_audio')
        probe = math_ops._make_probe(rate, config.probe_seconds)
    else:
        positive_integer(probe_sample_rate_hz, 'probe_sample_rate_hz')
        probe = audio_array(probe_audio).mean(axis=1)
        divisor = gcd(rate, probe_sample_rate_hz)
        probe = signal.resample_poly(probe, rate//divisor, probe_sample_rate_hz//divisor)
        probe = math_ops._padded_segment(probe, 0, round(rate*config.probe_seconds))
        if not np.any(probe):
            raise ValueError('probe audio must contain nonzero samples')
        probe = math_ops._fade_edges(math_ops._normalize_peak(probe, .27), rate, 50)
    parts, segments = [], []
    position = 0

    def append(kind, samples):
        nonlocal position
        if len(samples):
            segments.append(dict(type=kind, start=position, end=position+len(samples), length=len(samples)))
            parts.append(samples)
            position += len(samples)

    def silence(seconds):
        append('silence', np.zeros(round(rate*seconds)))

    silence(config.lead_seconds)
    append('start_marker', math_ops._make_sync_marker(rate, config.marker_seconds, 483917))
    silence(config.marker_gap_seconds)
    for _ in range(config.repeat_count):
        append('sweep', sweep)
        silence(config.sweep_gap_seconds)
    append('probe', probe)
    silence(config.probe_tail_seconds)
    append('end_marker', math_ops._make_sync_marker(rate, config.marker_seconds, 825601))
    silence(config.post_seconds)
    return SweepStimulus(np.concatenate(parts)[:, None], config, tuple(segments))


@dataclass(frozen=True)
class SweepMeasurement:
    impulse_response: np.ndarray
    repeated_responses: np.ndarray
    aligned_recording: np.ndarray
    predicted_probe: np.ndarray
    recorded_probe: np.ndarray
    alignment: dict
    channel_quality: tuple[dict, ...]
    response_offset_samples: int
    sample_rate_hz: int


def measure_impulse_response(recording, recording_sample_rate_hz, stimulus, *, response_seconds=1.15,
                             preroll_seconds=.15, regularization=1e-8,
                             maximum_clock_drift_ppm=1800.0, minimum_marker_correlation=.12):
    """Estimate channels on a shared clock and retain their relative arrival delays.

    Responses have shape (samples, channels); repetitions have shape
    (repeats, samples, channels). Marker alignment removes common latency.
    """
    samples = audio_array(recording)
    positive_integer(recording_sample_rate_hz, 'recording_sample_rate_hz')
    config, rate = stimulus.config, stimulus.config.sample_rate_hz
    for name, value in (('response_seconds', response_seconds), ('regularization', regularization),
                        ('maximum_clock_drift_ppm', maximum_clock_drift_ppm)):
        finite_scalar(value, name, minimum=0)
    finite_scalar(minimum_marker_correlation, 'minimum_marker_correlation', minimum=0, maximum=1)
    if not np.isfinite(preroll_seconds) or preroll_seconds <= .08 or response_seconds <= 0 or response_seconds > config.sweep_gap_seconds-.30:
        raise ValueError('preroll must exceed 0.08 s and response must fit the sweep gap with 0.30 s clearance')
    if len(samples) < recording_sample_rate_hz:
        raise ValueError('recording must contain at least one second')
    clipping = np.mean(np.abs(samples) >= .999, axis=0)
    divisor = gcd(rate, recording_sample_rate_hz)
    if rate != recording_sample_rate_hz:
        samples = signal.resample_poly(samples, rate//divisor, recording_sample_rate_hz//divisor, axis=0)
    excitation = stimulus.audio[:, 0]
    manifest = {'segments': stimulus.segments}
    aligned, alignment = math_ops._align_channels(samples, rate, excitation, manifest,
                                                 maximum_clock_drift_ppm, minimum_marker_correlation)
    preroll, length, guard = round(preroll_seconds*rate), round(response_seconds*rate), round(.18*rate)
    responses = []
    for segment in stimulus.segments:
        if segment['type'] == 'sweep':
            sweep = excitation[segment['start']:segment['end']]
            responses.append([math_ops._deconvolve_sweep(aligned[:, channel], sweep, segment['start'], rate,
                              preroll, length, regularization, guard) for channel in range(samples.shape[1])])
    responses = np.asarray(responses)
    peak = math_ops._arrival_peak(responses, preroll, rate)
    start = max(0, peak-round(.005*rate))
    if start+length > responses.shape[-1]:
        raise ValueError('response crop exceeds the deconvolution range')
    repeated = responses[:, :, start:start+length]
    offset = start-preroll
    probe = math_ops._segment_of(manifest, 'probe')
    probe_audio = excitation[probe['start']:probe['end']]
    combined, predicted, captured, quality = [], [], [], []
    for channel in range(samples.shape[1]):
        consistency = math_ops._repeat_consistency(repeated[:, channel], rate)
        response, policy, selected = math_ops._combine_repeats(repeated[:, channel])
        prediction = signal.fftconvolve(probe_audio, response)
        actual = math_ops._padded_segment(aligned[:, channel], probe['start']+offset, len(prediction))
        accuracy = math_ops._probe_accuracy(prediction, actual, rate, config.lower_frequency_hz, config.upper_frequency_hz)
        status, flags = math_ops._measurement_quality(alignment, float(clipping[channel]), consistency, accuracy)
        quality.append(dict(status=status, flags=flags, clipping_fraction=float(clipping[channel]),
                            repeat_consistency=consistency, probe_accuracy=accuracy,
                            selection_policy=policy, selected_repeat=selected))
        combined.append(response)
        predicted.append(prediction)
        captured.append(actual)
    return SweepMeasurement(np.stack(combined, axis=1), repeated.transpose(0, 2, 1), aligned,
                            np.stack(predicted, axis=1), np.stack(captured, axis=1), alignment,
                            tuple(quality), offset, rate)


def resample_impulse_response(impulse_response, sample_rate_hz, output_sample_rate_hz):
    """Resample taps while retaining discrete convolution gain."""
    samples = audio_array(impulse_response)
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    positive_integer(output_sample_rate_hz, 'output_sample_rate_hz')
    divisor = gcd(sample_rate_hz, output_sample_rate_hz)
    return signal.resample_poly(samples, output_sample_rate_hz//divisor, sample_rate_hz//divisor, axis=0)*sample_rate_hz/output_sample_rate_hz
