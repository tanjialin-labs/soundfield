"""Source separation followed by neural evidence and delayed channel routing."""

from dataclasses import dataclass
import numpy as np
from ..contracts import audio_array
from ..transform import ShortTimeFourierTransform
from ..selection import ChannelRoutingConfig, HysteresisChannelSelector, WakeWordBeamSelector
from ..selection.event_beam import EventBeamConfig, select_event_beam
from ..contracts import positive_integer
from ..numerics.spectral_functions import rescale_upper_range
from .source_separation import separate_sources
from .farfield_capture import estimate_aligned_masks


@dataclass(frozen=True)
class NeuralRoutingResult:
    audio: np.ndarray
    separated_audio: np.ndarray
    speech_gains: np.ndarray
    selected_channels: np.ndarray
    output_spectrum: np.ndarray
    latency_samples: int


def separate_and_route(audio, mask_estimator, separation_config, *, routing_config=None, compensate_delay=True):
    """Use source-dependent masks as both routing probability and energy evidence."""
    samples = audio_array(audio)
    separated = separate_sources(samples, separation_config)
    transform = ShortTimeFourierTransform()
    spectrum = transform.analyze(separated)
    features = transform.analyze(np.round(separated*32767)/32768)
    masks, _ = estimate_aligned_masks(features, mask_estimator)
    gains = rescale_upper_range(masks, .4, .8)
    selector = HysteresisChannelSelector(routing_config or ChannelRoutingConfig(channel_count=samples.shape[1]))
    output = np.zeros((*spectrum.shape[:2], selector.config.output_channel_count), complex)
    selected = np.zeros(spectrum.shape[1], int)
    for frame in range(spectrum.shape[1]):
        output[:, frame] = selector.select(spectrum[:, frame], spectrum[:, frame]*gains[:, frame], gains[:, frame])
        selected[frame] = selector.selected_channel
    if compensate_delay and spectrum.shape[1]:
        output = np.concatenate((output, selector.flush()), axis=1)[:, selector.latency_frames:selector.latency_frames+spectrum.shape[1]]
    return NeuralRoutingResult(transform.synthesize(output, length_samples=len(samples)), separated, gains, selected,
        output, 256*(mask_estimator.config.right_context_frames+selector.latency_frames))


def select_wakeword_events(audio, sample_rate_hz, event_seconds, beam_activity, *, history_seconds=5.,
                           config=None, strategy='energy'):
    """Select a beam for each event using the preceding recording interval."""
    samples = audio_array(audio)
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    if strategy not in ('energy', 'noise_contrast'):
        raise ValueError('strategy must be energy or noise_contrast')
    times = np.asarray(event_seconds, dtype=float)
    activity = np.asarray(beam_activity)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times<=0) or np.any(times>len(samples)/sample_rate_hz):
        raise ValueError('event times must be positive and fall within the recording')
    if activity.shape != (len(times), samples.shape[1]) or np.any((activity!=0)&(activity!=1)):
        raise ValueError('beam_activity must contain one binary channel row per event')
    if not np.isfinite(history_seconds) or history_seconds <= 0:
        raise ValueError('history_seconds must be positive')
    transform = ShortTimeFourierTransform()
    selector = WakeWordBeamSelector(config) if strategy == 'noise_contrast' else None
    choices = np.empty(len(times), int)
    for index, time in enumerate(times):
        start, stop = int(np.floor(max(0, time-history_seconds)*sample_rate_hz)), int(np.floor(time*sample_rate_hz))
        spectrum = transform.analyze(samples[start:stop])
        choices[index] = (selector.select(activity[index], spectrum) if selector is not None
                          else select_event_beam(spectrum, activity[index], config or EventBeamConfig()))
    return choices
