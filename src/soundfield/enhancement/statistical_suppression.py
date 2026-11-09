"""Single-channel suppression with conditional noise tracking and bounded attenuation."""

import numpy as np

from ..aec.noise_tracking import ConditionalNoiseTracker, EchoNoiseConfig, SpeechPresenceGain
from ..contracts import finite_scalar, spectral_frame


class StatisticalNoiseSuppressor:
    """Estimate noise from the microphone alone and preserve bins above the estimated background."""

    def __init__(self, config=EchoNoiseConfig(), *, minimum_gain=.16):
        if not isinstance(config, EchoNoiseConfig):
            raise TypeError("config must be an EchoNoiseConfig")
        finite_scalar(minimum_gain, "minimum_gain", minimum=0, maximum=1)
        self.config = config
        self.minimum_gain = minimum_gain
        self._tracker = ConditionalNoiseTracker(config)
        self._estimator = SpeechPresenceGain(config)
        self._zero_echo = np.zeros(config.frequency_bin_count)
        self.reset()

    @property
    def gain(self):
        return self._gain.copy()

    def reset(self):
        self._tracker.reset()
        self._estimator.reset()
        self._gain = np.ones(self.config.frequency_bin_count)

    def process_frame(self, spectrum_frame):
        spectrum = spectral_frame(spectrum_frame, self.config.frequency_bin_count, 1)
        power = np.abs(spectrum[:, 0])**2
        noise = self._tracker.process_frame(power, self._zero_echo, self._estimator.previous_prior_term)
        gain = self._estimator.process_frame(power, noise, self._zero_echo)
        noise_fraction = np.divide(noise, power, out=np.ones_like(power), where=power > 0)
        self._gain = np.clip(np.maximum(gain, 1 - noise_fraction), self.minimum_gain, 1.)
        return spectrum * self._gain[:, None]
