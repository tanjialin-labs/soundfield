"""Kurtosis-ranked channel retention and speech-aware output gating."""

from dataclasses import dataclass

import numpy as np

from ..contracts import positive_integer, finite_scalar, real_array, unit_interval_array, audio_array


@dataclass(frozen=True)
class ConferenceSelectionConfig:
    channel_count: int = 8
    output_channel_count: int = 3
    speech_threshold: float = 0.2
    history_frames: int = 10
    prefer_active_speakers: bool = False

    def __post_init__(self):
        if not isinstance(self.prefer_active_speakers, bool):
            raise ValueError('prefer_active_speakers must be boolean')
        for name in ('channel_count', 'output_channel_count', 'history_frames'):
            positive_integer(getattr(self, name), name)
        finite_scalar(self.speech_threshold, 'speech_threshold', minimum=0, maximum=1)
        if self.output_channel_count > self.channel_count or self.history_frames < 2:
            raise ValueError('selection requires enough input channels and at least two history frames')


@dataclass(frozen=True)
class ConferenceSelection:
    channel_indices: np.ndarray
    ranked_channel_indices: np.ndarray
    low_kurtosis_mean: float
    speech_probability: np.ndarray
    kurtosis: np.ndarray


class KurtosisChannelSelector:
    def __init__(self, config=ConferenceSelectionConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.channel_history = np.tile(np.arange(self.config.output_channel_count), (self.config.history_frames, 1))
        self.frame_count = 0

    def select(self, speech_probability, kurtosis):
        config = self.config
        speech = unit_interval_array(speech_probability, 'speech_probability', (config.channel_count,))
        scores = real_array(kurtosis, 'kurtosis', 1)
        if scores.shape != speech.shape:
            raise ValueError('kurtosis must contain one value per channel')
        rank = np.argsort(scores, kind='stable')
        speech_rank = np.argsort(speech, kind='stable')
        low_count = min(4, config.channel_count)
        common = np.intersect1d(rank[:low_count], speech_rank[:low_count])
        baseline = scores[common if common.size else rank[:low_count]].mean()
        priority = np.lexsort((scores, speech > config.speech_threshold)) if config.prefer_active_speakers else rank
        top = priority[-config.output_channel_count:]
        previous = self.channel_history[-1].copy()
        selected = previous.copy()
        retained = np.isin(selected, top)
        incoming = top[~np.isin(top, selected)]
        selected[~retained] = incoming
        if self.frame_count:
            if not np.any(speech > config.speech_threshold):
                selected = previous
            else:
                reject = (selected != previous) & (speech[selected] < config.speech_threshold)
                selected[reject] = previous[reject]
        self.channel_history[:-1] = self.channel_history[1:]
        self.channel_history[-1] = selected
        self.frame_count += 1
        return ConferenceSelection(selected.copy(), top.copy(), float(baseline), speech.copy(), scores.copy())


def gate_separated_channels(audio_block, selection: ConferenceSelection, separated_speech_probability):
    """Apply channel decisions to a sample-aligned separated audio block."""
    audio = audio_array(audio_block)
    selected = selection.channel_indices
    after = unit_interval_array(separated_speech_probability, 'separated_speech_probability', (len(selected),))
    if audio.shape[1] != len(selected):
        raise ValueError('audio channels must match the selection')
    before = selection.speech_probability[selected]
    highest = selection.ranked_channel_indices[-1]
    second_highest = selection.ranked_channel_indices[-2] if len(selected) > 1 else highest
    clear = (after < 0.1) & (selected != highest)
    clear[selection.kurtosis[selected]-selection.low_kurtosis_mean > 1.0] = False
    clear |= ((after < 0.2) & (before > 0.2) & (before-after > 0.2)
              & (selected != highest) & (selected != second_highest))
    clear |= (after < 0.2) & (before-after > 0.5)
    output = audio.copy()
    output[:, clear] = 0
    return output, ~clear
