"""Block output control with windowed DC removal and continuous de-emphasis."""

import numpy as np

from ..contracts import audio_array
from .audio_output import EchoAudioSynthesizer


class PlaybackOutputController(EchoAudioSynthesizer):
    def __init__(self, hop_length_samples=512, *, preemphasis=.9, output_gain=1,
                 remove_inactive_dc=False):
        self.remove_inactive_dc = remove_inactive_dc
        super().__init__(hop_length_samples, preemphasis=preemphasis, output_gain=output_gain)

    def reset(self):
        super().reset()
        self.microphone_history = np.zeros((2*self.hop_length_samples, 1))

    def process_block(self, microphone_audio, residual_spectrum=None, *, is_reference_active=True,
                      is_muted=False, linear_audio=None):
        microphone = audio_array(microphone_audio)
        hop = self.hop_length_samples
        if microphone.shape != (hop, 1):
            raise ValueError('microphone_audio must contain one hop and one channel')
        if linear_audio is not None:
            linear = audio_array(linear_audio)
            if linear.shape != microphone.shape:
                raise ValueError('linear_audio must match microphone_audio')
        self.microphone_history[:hop] = self.microphone_history[hop:]
        self.microphone_history[hop:] = microphone
        if not is_reference_active:
            windowed = self.microphone_history*self.window[:, None]
            if self.remove_inactive_dc:
                windowed -= windowed.mean(axis=0)
            return super().process_frame(np.fft.rfft(windowed, axis=0))
        if is_muted:
            return self._finish(np.zeros_like(microphone))
        if linear_audio is not None:
            return self._finish(linear)
        return super().process_frame(residual_spectrum)
