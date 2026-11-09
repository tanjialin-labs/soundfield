"""Synchronized audio chunks with explicit input and output sample positions."""

from dataclasses import dataclass

import numpy as np

from ..contracts import audio_array, nonnegative_integer, positive_integer
from .reference_alignment import PlaybackReferenceAligner
from .echo_cancellation import AcousticEchoCanceller, EchoCancellationConfig


@dataclass(frozen=True)
class TimedEchoAudio:
    """Audio on [start_sample, stop_sample), with an input availability boundary.

    available_after_sample is an exclusive input sample count, not wall time.
    released_on_flush means availability also requires an end-of-stream signal.
    """

    audio: np.ndarray
    communication_audio: np.ndarray
    start_sample: int
    available_after_sample: int
    sample_rate_hz: int
    released_on_flush: bool

    @property
    def stop_sample(self):
        return self.start_sample+len(self.audio)


class EchoCancellationStream:
    """Process contiguous, synchronized microphone and playback chunks at 16 kHz.

    Input chunks may have any length. Outputs have at most one hop each and
    retain the recording's sample positions. Optional reference alignment uses the first microphone
    and a bounded playback history. One instance owns
    one stream; flush closes it until reset.
    """

    def __init__(self, config=None, *, sample_rate_hz=16000, reference_alignment=None, maximum_gap_samples=0):
        nonnegative_integer(maximum_gap_samples, 'maximum_gap_samples')
        self.maximum_gap_samples = maximum_gap_samples
        positive_integer(sample_rate_hz, 'sample_rate_hz')
        if sample_rate_hz != 16000:
            raise ValueError('echo streaming requires a 16000 Hz sample grid')
        self.config = config if config is not None else EchoCancellationConfig()
        if self.config.residual_filter and self.config.residual_filter.sample_rate_hz != sample_rate_hz:
            raise ValueError('residual filtering and stream sample rates must agree')
        if self.config.residual_filter and self.config.residual_filter.method == 'mask':
            raise ValueError('external masks require an explicitly scheduled block interface')
        self.sample_rate_hz = int(sample_rate_hz)
        self._processor = AcousticEchoCanceller(self.config)
        self._microphone = np.zeros((self.latency_samples, self.config.input.microphone_count))
        self._reference = np.zeros((self.latency_samples, self.config.input.reference_count))
        self._aligner = (PlaybackReferenceAligner(reference_alignment, self.latency_samples, self.config.input.reference_count)
                         if reference_alignment is not None else None)
        self.reset()

    @property
    def latency_samples(self):
        """Input lookahead after each complete output interval's end."""
        return self.config.input.hop_length_samples

    @property
    def startup_samples(self):
        """Input samples needed for the first complete output hop before flush."""
        return 2*self.latency_samples

    @property
    def received_samples(self):
        return self._received

    @property
    def pending_samples(self):
        """Accepted input samples whose corresponding output has not been emitted."""
        return self._received-self._emitted

    def reset(self):
        """Clear processing state and restart the sample timeline at zero."""
        self._processor.reset()
        if self._aligner is not None:
            self._aligner.reset()
        self._microphone.fill(0)
        self._reference.fill(0)
        self._received = self._emitted = self._filled = self._blocks = 0
        self._closed = False

    def _process_hop(self, released_on_flush):
        reference = self._aligner.process_block(self._microphone, self._reference) if self._aligner is not None else self._reference
        result = self._processor.process_block(self._microphone, reference)
        self._blocks += 1
        self._filled = 0
        start = (self._blocks-2)*self.latency_samples
        if start < 0 or start >= self._received:
            return ()
        stop = min(start+self.latency_samples, self._received)
        self._emitted = stop
        return (TimedEchoAudio(result.audio[:stop-start].copy(),
                              result.communication_audio[:stop-start].copy(),
                              start, self._received, self.sample_rate_hz, released_on_flush),)

    def process_block(self, microphone_audio, reference_audio, *, start_sample):
        """Accept paired chunks at the next input sample; fill configured timestamp gaps with silence.

        Returned availability boundaries describe the earliest required input,
        independent of the caller's chunk size or wall-clock scheduling.
        """
        if self._closed:
            raise ValueError('stream is closed; reset before supplying more audio')
        nonnegative_integer(start_sample, 'start_sample')
        microphone, reference = audio_array(microphone_audio), audio_array(reference_audio)
        gap = start_sample-self._received
        if gap < 0 or gap > self.maximum_gap_samples:
            raise ValueError('start_sample must equal received_samples')
        if (len(microphone) != len(reference) or microphone.shape[1] != self.config.input.microphone_count
                or reference.shape[1] != self.config.input.reference_count):
            raise ValueError('paired chunks must have equal lengths and configured channel counts')
        outputs = []
        if gap:
            outputs.extend(self.process_block(np.zeros((gap, microphone.shape[1])), np.zeros((gap, reference.shape[1])),
                                              start_sample=self._received))
        offset = 0
        while offset < len(microphone):
            count = min(self.latency_samples-self._filled, len(microphone)-offset)
            stop = self._filled+count
            self._microphone[self._filled:stop] = microphone[offset:offset+count]
            self._reference[self._filled:stop] = reference[offset:offset+count]
            self._filled = stop
            self._received += count
            offset += count
            if self._filled == self.latency_samples:
                outputs.extend(self._process_hop(False))
        return tuple(outputs)

    def flush(self):
        """Declare input complete, emit all pending samples once, and close.

        A partial hop is zero-padded and one extra zero hop advances analysis.
        Padding is excluded from the returned recording timeline. Empty streams
        and repeated calls return no chunks.
        """
        if self._closed:
            return ()
        outputs = []
        if self._received:
            if self._filled:
                self._microphone[self._filled:] = 0
                self._reference[self._filled:] = 0
                outputs.extend(self._process_hop(True))
            self._microphone.fill(0)
            self._reference.fill(0)
            outputs.extend(self._process_hop(True))
        self._closed = True
        return tuple(outputs)
