"""Channel routing with speech evidence, temporal voting, and hysteresis."""

from dataclasses import dataclass

import numpy as np

from ..contracts import finite_scalar, nonnegative_integer, positive_integer, spectral_frame, unit_interval_array


@dataclass(frozen=True)
class ChannelRoutingConfig:
    frequency_bin_count: int = 257
    channel_count: int = 3
    output_channel_count: int = 1
    energy_start_bin: int = 2
    energy_stop_bin: int = 256
    comparison_start_bin: int = 31
    comparison_stop_bin: int = 256
    speech_start_bin: int = 31
    speech_stop_bin: int = 256
    energy_history_frames: int = 20
    output_delay_frames: int = 20
    activity_history_frames: int = 10
    adjacent_channel_comparison: bool = True
    activity_energy_ratio: float = 4.0
    speech_probability_threshold: float = .25
    switching_energy_ratio: float = 1.5
    adjacent_energy_ratio: float = 1.5
    winning_frame_fraction: float = .75
    adjacent_winning_fraction: float = .6

    def __post_init__(self):
        for name in ("frequency_bin_count", "channel_count", "output_channel_count", "energy_history_frames", "activity_history_frames"):
            positive_integer(getattr(self, name), name)
        nonnegative_integer(self.output_delay_frames, "output_delay_frames")
        if self.output_channel_count not in (1, self.channel_count):
            raise ValueError("output_channel_count must be one or equal channel_count")
        if self.output_delay_frames and self.activity_history_frames > self.output_delay_frames:
            raise ValueError("activity_history_frames cannot exceed output_delay_frames")
        for band in ("energy", "comparison", "speech"):
            start, stop = getattr(self, band + "_start_bin"), getattr(self, band + "_stop_bin")
            nonnegative_integer(start, band + "_start_bin")
            positive_integer(stop, band + "_stop_bin")
            if not start < stop <= self.frequency_bin_count:
                raise ValueError(f"invalid {band} frequency interval")
        for name in ("activity_energy_ratio", "switching_energy_ratio", "adjacent_energy_ratio"):
            finite_scalar(getattr(self, name), name, minimum=0)
        for name in ("speech_probability_threshold", "winning_frame_fraction", "adjacent_winning_fraction"):
            finite_scalar(getattr(self, name), name, minimum=0, maximum=1)


class HysteresisChannelSelector:
    def __init__(self, config: ChannelRoutingConfig | None = None):
        self.config = config if config is not None else ChannelRoutingConfig()
        self.reset()

    def reset(self):
        config = self.config
        length = config.energy_history_frames + config.output_delay_frames
        self._energy = np.zeros((length, config.channel_count))
        self._comparison_energy = np.zeros_like(self._energy)
        self._speech = np.zeros_like(self._energy)
        self._audio = np.zeros((config.frequency_bin_count, length, config.channel_count), complex)
        self._selected_channel = 0
        self._frame_index = 0
        self._closed = False

    @property
    def selected_channel(self):
        return self._selected_channel

    @property
    def latency_frames(self):
        return self.config.output_delay_frames

    def select(self, spectrum_frame, evidence_frame, speech_probability=None):
        if self._closed:
            raise RuntimeError("reset is required after flush")
        config = self.config
        spectrum = spectral_frame(spectrum_frame, config.frequency_bin_count, config.channel_count)
        evidence = spectral_frame(evidence_frame, config.frequency_bin_count, config.channel_count)
        probability = (np.ones(evidence.shape) if speech_probability is None else
                       unit_interval_array(speech_probability, "speech_probability", evidence.shape))
        speech = np.mean(probability[config.speech_start_bin:config.speech_stop_bin], axis=0)
        power = np.abs(evidence * probability) ** 2
        self._energy[:-1] = self._energy[1:]
        self._energy[-1] = np.mean(power[config.energy_start_bin:config.energy_stop_bin], axis=0)
        mean_energy = np.mean(self._energy[:config.energy_history_frames], axis=0)
        ordered = np.argsort(mean_energy, kind="stable")
        candidate, previous = int(ordered[-1]), self._selected_channel
        if not config.output_delay_frames:
            if (np.max(mean_energy) < config.activity_energy_ratio * np.min(mean_energy)
                    or np.max(speech) < config.speech_probability_threshold):
                candidate = previous
            selected = spectrum[:, candidate]
        else:
            self._speech[:-1] = self._speech[1:]
            self._speech[-1] = speech
            if config.adjacent_channel_comparison:
                self._comparison_energy[:-1] = self._comparison_energy[1:]
                self._comparison_energy[-1] = np.mean(power[config.comparison_start_bin:config.comparison_stop_bin], axis=0)
            position = config.energy_history_frames - 1
            if candidate != previous:
                activity = slice(position + 1, position + 1 + config.activity_history_frames)
                activity_energy = np.mean(self._energy[activity], axis=0)
                active_speech = np.mean(self._speech[activity], axis=0)
                if (np.max(activity_energy) < config.activity_energy_ratio * np.min(activity_energy)
                        or np.max(active_speech) < config.speech_probability_threshold):
                    candidate = previous
                else:
                    votes = np.count_nonzero(np.argmax(self._energy[position + 1:], axis=1) == candidate)
                    fraction = config.winning_frame_fraction
                    hold = mean_energy[candidate] < mean_energy[previous] * config.switching_energy_ratio
                    if config.adjacent_channel_comparison and config.channel_count > 1:
                        runner_up = int(ordered[-2])
                        if abs(candidate - runner_up) in (1, config.channel_count - 1):
                            fraction = config.adjacent_winning_fraction
                            if runner_up == previous:
                                comparison = np.mean(self._comparison_energy[:config.energy_history_frames], axis=0)
                                hold = comparison[candidate] < comparison[previous] * config.adjacent_energy_ratio
                    if votes < np.floor(fraction * config.output_delay_frames) or hold:
                        candidate = previous
            self._audio[:, :-1] = self._audio[:, 1:]
            self._audio[:, -1] = spectrum
            selected = self._audio[:, position, candidate]
        self._selected_channel = candidate
        self._frame_index += 1
        output = np.zeros((config.frequency_bin_count, config.output_channel_count), complex)
        output[:, 0 if config.output_channel_count == 1 else candidate] = selected
        return output

    def flush(self):
        config = self.config
        if self._closed or not self._frame_index or not self.latency_frames:
            self._closed = True
            return np.empty((config.frequency_bin_count, 0, config.output_channel_count), complex)
        silence = np.zeros((config.frequency_bin_count, config.channel_count), complex)
        output = np.stack([self.select(silence, silence) for _ in range(self.latency_frames)], axis=1)
        self._closed = True
        return output
