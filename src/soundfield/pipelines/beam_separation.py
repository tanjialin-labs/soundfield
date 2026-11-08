"""Adaptive beam enhancement followed by source separation on one frame grid."""

from dataclasses import dataclass

import numpy as np

from ..beamforming import GeneralizedSidelobeCanceller
from ..contracts import audio_array
from ..separation import OnlineSourceSeparator
from ..transform import FourierConfig, ShortTimeFourierTransform


@dataclass(frozen=True)
class BeamSeparationResult:
    audio: np.ndarray
    input_spectrum: np.ndarray
    fixed_beam_spectrum: np.ndarray
    blocking_spectrum: np.ndarray
    enhanced_beam_spectrum: np.ndarray
    separated_spectrum: np.ndarray
    separation_gains: np.ndarray


def separate_beams(audio, beam_bank, enhancement_config, separation_config, *, transform_config=None, speech_mask=None):
    """Return all signal stages; spectral frames keep their analysis time indices."""
    samples = audio_array(audio)
    bins = beam_bank.frequency_bin_count
    transform_config = transform_config or FourierConfig(2*(bins-1), bins-1)
    transform = ShortTimeFourierTransform(transform_config)
    if (samples.shape[1] != beam_bank.microphone_count or transform_config.frequency_bin_count != bins
            or separation_config.frequency_bin_count != bins or separation_config.channel_count != beam_bank.beam_count):
        raise ValueError('audio, beam bank, transform and separation dimensions must agree')
    if separation_config.nearfield_control is not None:
        raise ValueError('near-field separation requires explicit spatial evidence')
    spectrum = transform.analyze(samples)
    if speech_mask is not None and np.shape(speech_mask) != spectrum.shape[:2]:
        raise ValueError('speech_mask must match the complete analysis frame grid')
    processors = [GeneralizedSidelobeCanceller(beam_bank, beam, enhancement_config, speech_mask=speech_mask)
                  for beam in range(beam_bank.beam_count)]
    separator = OnlineSourceSeparator(separation_config)
    shape = (bins, spectrum.shape[1], beam_bank.beam_count)
    fixed, enhanced, separated = (np.empty(shape, complex) for _ in range(3))
    gains = np.empty(shape)
    blocked = np.empty((*shape[:2], beam_bank.reference_count, beam_bank.beam_count), complex)
    for frame in range(spectrum.shape[1]):
        fixed[:, frame], blocked[:, frame] = beam_bank.project(spectrum[:, frame])
        for beam, processor in enumerate(processors):
            enhanced[:, frame, beam:beam+1] = processor.process_frame(spectrum[:, frame])
        separated[:, frame] = separator.process_frame(enhanced[:, frame])
        gains[:, frame] = separator.separation_gain
    output = transform.synthesize(separated, length_samples=len(samples))
    return BeamSeparationResult(output, spectrum, fixed, blocked, enhanced, separated, gains)
