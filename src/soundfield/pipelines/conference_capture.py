"""Two-stage conference capture with explicit neural and separation resources."""

from dataclasses import dataclass

import numpy as np

from ..contracts import audio_array
from ..neural import SpeechActivityDetector
from ..metrics.prediction_kurtosis import PredictionResidualKurtosis
from ..selection.conference_channels import ConferenceSelectionConfig, KurtosisChannelSelector, gate_separated_channels
from ..separation import DemixingConfig, OnlineSourceSeparator


@dataclass(frozen=True)
class ConferenceCaptureResult:
    audio: np.ndarray
    selected_audio: np.ndarray
    separated_audio: np.ndarray
    selected_channel_indices: np.ndarray
    speech_probability: np.ndarray
    residual_kurtosis: np.ndarray
    retained_channels: np.ndarray


def capture_conference(audio, speech_model, *, selection_config=None, separation_config=None):
    """Select microphones in 160-sample hops, separate sources, then gate aligned outputs."""
    samples = audio_array(audio)
    selection_config = selection_config or ConferenceSelectionConfig(channel_count=samples.shape[1])
    count = selection_config.output_channel_count
    separation_config = separation_config or DemixingConfig(channel_count=count, weighting_strategy='blended_energy',
                                                            output_mode='channel_mask')
    if samples.shape[1] != selection_config.channel_count or separation_config.channel_count != count:
        raise ValueError('audio, selection and separation channel counts must agree')
    if separation_config.nearfield_control is not None:
        raise ValueError('nearfield control requires explicit per-frame spatial evidence')
    frame_count = (len(samples)+159)//160
    padded = np.pad(samples, ((0, frame_count*160-len(samples)), (0, 0)))
    first_detector = SpeechActivityDetector(speech_model, samples.shape[1])
    second_detector = SpeechActivityDetector(speech_model, count)
    kurtosis = PredictionResidualKurtosis(samples.shape[1])
    selector = KurtosisChannelSelector(selection_config)
    first_history = np.zeros((2, samples.shape[1]))
    second_history = np.zeros((2, count))
    selected_audio = np.zeros((len(padded), count))
    selections = []
    probabilities, scores = [], []
    for frame in range(frame_count):
        block = padded[frame*160:(frame+1)*160]
        first_history[0], first_history[1] = first_history[1].copy(), first_detector.process_block(block)
        speech = first_history.mean(axis=0)
        residual_score, _ = kurtosis.process_block(block, speech)
        selection = selector.select(speech, residual_score)
        selected_audio[frame*160:(frame+1)*160] = block[:, selection.channel_indices]
        selections.append(selection)
        probabilities.append(speech)
        scores.append(residual_score)
    separator = OnlineSourceSeparator(separation_config)
    frame_length = 2*(separation_config.frequency_bin_count-1)
    hop = frame_length//2
    window = np.sqrt(np.hanning(frame_length))
    overlap = np.zeros((hop, count))
    separated = np.zeros_like(selected_audio)
    for start in range(0, len(padded), hop):
        block = selected_audio[start:start+frame_length]
        block = np.pad(block, ((0, frame_length-len(block)), (0, 0)))
        spectrum = np.fft.rfft(block*window[:, None], axis=0)
        output = separator.process_frame(spectrum)
        synthesized = np.fft.irfft(output, n=frame_length, axis=0)*window[:, None]
        length = min(hop, len(padded)-start)
        separated[start:start+length] = (overlap+synthesized[:hop])[:length]
        overlap = synthesized[hop:].copy()
    output = np.zeros_like(separated)
    retained = np.zeros((frame_count, count), bool)
    for frame, selection in enumerate(selections):
        block = separated[frame*160:(frame+1)*160]
        second_history[0], second_history[1] = second_history[1].copy(), second_detector.process_block(block)
        output[frame*160:(frame+1)*160], retained[frame] = gate_separated_channels(block, selection, second_history.mean(axis=0))
    return ConferenceCaptureResult(output[:len(samples)], selected_audio[:len(samples)], separated[:len(samples)],
        np.array([s.channel_indices for s in selections], dtype=int).reshape(frame_count, count),
        np.asarray(probabilities).reshape(frame_count, samples.shape[1]), np.asarray(scores).reshape(frame_count, samples.shape[1]), retained)
