"""Frequency-selective attenuation relative to a persistent off-axis arrival."""
import numpy as np

from ..localization.geometry import plane_wave_responses


class InterferenceSelectiveGain:
    """Redistribute a closed spatial gate's attenuation using a two-microphone phase contrast."""

    def __init__(self, matching_config):
        if matching_config.geometry.microphone_count != 2:
            raise ValueError("interference-selective attenuation requires two microphones")
        frequencies = np.arange(matching_config.frequency_bin_count) * (
            matching_config.sample_rate_hz / matching_config.frame_length_samples)
        self.angles = np.asarray(matching_config.scan_angles_degrees)
        responses = plane_wave_responses(matching_config.geometry, frequencies, self.angles,
                                         polar_angle_radians=matching_config.polar_angle_radians,
                                         propagation_sign=matching_config.propagation_sign)
        self.relative_phase = responses[:, 1] * responses[:, 0].conj()
        self.histogram = np.zeros(len(self.angles))

    def reset(self):
        self.histogram.fill(0)

    def process_frame(self, microphones, votes, target_index, broadband_gain):
        self.histogram *= .985
        self.histogram += votes
        offset = np.abs((self.angles - self.angles[target_index] + 180) % 360 - 180)
        eligible = offset >= 30
        rival = np.argmax(np.where(eligible, self.histogram, 0.))
        support = self.histogram[rival] if eligible[rival] else 0.
        if broadband_gain >= 1 or support == 0:
            return np.full(len(microphones), broadband_gain)
        target_phase = self.relative_phase[:, target_index]
        rival_phase = self.relative_phase[:, rival]
        contrast = np.abs(target_phase-rival_phase)**2
        confidence = np.clip((contrast-.05)/.5, 0, 1) * min(1., 2*support/max(1., self.histogram.max()))
        target_error = np.abs(microphones[:, 1] - target_phase*microphones[:, 0])**2
        rival_error = np.abs(microphones[:, 1] - rival_phase*microphones[:, 0])**2
        target_error = np.maximum(target_error, 1e-15)
        scale = np.maximum(rival_error, target_error)
        rival_power = (rival_error/scale)**2
        share = rival_power/(rival_power+(target_error/scale)**2)
        spectral_gain = .16 + .84*share
        return broadband_gain + .35*confidence*(spectral_gain-broadband_gain)
