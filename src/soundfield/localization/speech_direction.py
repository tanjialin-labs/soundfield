"""Speech-gated direction persistence within channel-dependent search regions."""

from dataclasses import dataclass

import numpy as np

from ..contracts import real_array, positive_integer, finite_scalar, nonnegative_integer


@dataclass(frozen=True)
class SpeechDirectionConfig:
    smoothing: float = 0.9
    peak_distance_bins: int = 2
    silence_hold_frames: int = 20
    silence_reset_frames: int = 50
    speech_threshold: float = 10.0
    history_frames: int = 5

    def __post_init__(self):
        finite_scalar(self.smoothing, 'smoothing', minimum=0, maximum=1)
        finite_scalar(self.speech_threshold, 'speech_threshold', minimum=0)
        for name in ('peak_distance_bins', 'silence_hold_frames', 'silence_reset_frames', 'history_frames'):
            positive_integer(getattr(self, name), name)
        if self.silence_reset_frames < self.silence_hold_frames:
            raise ValueError('silence_reset_frames must be at least silence_hold_frames')


class SpeechGatedDirectionTracker:
    """Return zero-based direction indices, or None after sustained inactivity."""

    def __init__(self, config=SpeechDirectionConfig(), *, search_regions=((0, 15), (7, 20), (13, 27)),
                 direction_count=27, circular_search=False):
        positive_integer(direction_count, 'direction_count')
        self.config, self.direction_count = config, direction_count
        self.search_regions = tuple(search_regions)
        if not isinstance(circular_search, bool):
            raise ValueError("circular_search must be boolean")
        self.circular_search = circular_search
        self._regions = []
        for start, stop in self.search_regions:
            nonnegative_integer(start, "region start")
            nonnegative_integer(stop, "region stop")
            if start >= direction_count or stop > direction_count or start == stop:
                raise ValueError("search regions must be nonempty half-open direction intervals")
            if start < stop:
                region = np.arange(start, stop)
            elif circular_search:
                region = np.r_[start:direction_count, 0:stop]
            else:
                raise ValueError("wrapped search regions require circular_search")
            self._regions.append(region)
        if not self._regions:
            raise ValueError("at least one search region is required")
        self.reset()

    def reset(self):
        self.smoothed_scores = np.zeros(self.direction_count)
        self.direction_history = np.full(self.config.history_frames, -2-self.config.peak_distance_bins, int)
        self.silent_frames = 0

    def process_frame(self, direction_scores, speech_evidence, selected_channel):
        scores = real_array(direction_scores, 'direction_scores', 1)
        finite_scalar(speech_evidence, 'speech_evidence', minimum=0)
        if scores.shape != (self.direction_count,) or np.any(scores < 0):
            raise ValueError('direction_scores must be nonnegative and match direction_count')
        if isinstance(selected_channel, bool) or not isinstance(selected_channel, (int, np.integer)) or not 0 <= selected_channel < len(self.search_regions):
            raise ValueError('selected_channel is outside the search regions')
        index = self.direction_history[-1]
        region = self._regions[selected_channel]
        if index >= 0 and index not in region:
            index = -2-self.config.peak_distance_bins
        if speech_evidence > self.config.speech_threshold and np.any(scores[region] > 0):
            self.silent_frames = 0
            self.smoothed_scores *= self.config.smoothing
            self.smoothed_scores += (1-self.config.smoothing)*scores
            levels = self.smoothed_scores[region]
            candidates = region[levels > levels.max()*0.5]
            distances = np.abs(candidates-index)
            if self.circular_search and index >= 0:
                distances = np.minimum(distances, self.direction_count-distances)
            if candidates.size and (index < 0 or distances.min() > self.config.peak_distance_bins):
                index = int(region[np.argmax(levels)])
        else:
            self.silent_frames += 1
        if self.silent_frames >= self.config.silence_hold_frames:
            index = -2-self.config.peak_distance_bins
        self.direction_history[:-1] = self.direction_history[1:]
        self.direction_history[-1] = index
        if self.silent_frames >= self.config.silence_reset_frames:
            self.smoothed_scores.fill(0)
        index = int(self.direction_history[-1])
        return None if index < 0 else index
