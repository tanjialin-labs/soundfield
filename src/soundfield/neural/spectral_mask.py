"""Per-channel spectral gain and speech probability estimation."""

from dataclasses import dataclass

import numpy as np

from ..contracts import positive_integer, spectral_frame
from .band_features import (BandFeatureConfig, TemporalFeatureNormalizer,
                            triangular_band_energy, interpolate_band_gains, round_away_from_zero)


@dataclass(frozen=True)
class SpectralMaskResult:
    gains: np.ndarray
    speech_probability: np.ndarray
    target_frame_index: int


class SpectralMaskEstimator:
    """Input spectra use Q21 units; reported gains and probabilities are linear."""

    def __init__(self, model, offset, scale, config=BandFeatureConfig(), *, channel_count=1):
        positive_integer(channel_count, 'channel_count')
        if model.input_count != config.feature_count or model.output_count < config.band_count:
            raise ValueError('network dimensions do not match the band configuration')
        self.config, self.model, self.channel_count = config, model, channel_count
        self.normalizers = [TemporalFeatureNormalizer(config, offset, scale) for _ in range(channel_count)]
        self.network_states = [model.create_state() for _ in range(channel_count)]
        self.reset()

    def reset(self):
        for processor in (*self.normalizers, *self.network_states):
            processor.reset()
        self.frame_count = 0
        self.band_energies = np.zeros((self.config.band_count, self.channel_count))
        self.features = np.zeros_like(self.band_energies)

    def process_frame(self, spectrum_frame):
        spectrum = spectral_frame(spectrum_frame, self.config.frequency_bin_count, self.channel_count)
        gains = np.zeros(spectrum.shape, dtype=float)
        probabilities = np.zeros(self.channel_count)
        for channel, (normalizer, network) in enumerate(zip(self.normalizers, self.network_states)):
            energy = triangular_band_energy(spectrum[:, channel], self.config.band_edges) / 32
            features = round_away_from_zero(np.log10(energy + 1/32) * 2**24)
            output = network.process_frame(normalizer.process_frame(features)) / 2**24
            gains[:, channel] = interpolate_band_gains(output[:self.config.band_count], self.config.band_edges,
                                                       self.config.frequency_bin_count)
            probabilities[channel] = output[-1]
            self.band_energies[:, channel], self.features[:, channel] = energy, features
        index = self.frame_count - self.config.right_context_frames
        self.frame_count += 1
        return SpectralMaskResult(gains, probabilities, index)
