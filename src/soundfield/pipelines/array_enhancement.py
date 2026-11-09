"""Whole-recording array enhancement with independent beam states."""

import numpy as np

from ..beamforming import GeneralizedSidelobeCanceller
from ..contracts import audio_array
from ..transform import FourierConfig, ShortTimeFourierTransform


def enhance_microphone_array(audio, bank, config, *, transform_config=None, speech_mask=None):
    """Return enhanced audio (samples, beams), processing every analysis frame.

    A speech mask, when used, has shape (bins, frames) and is shared by beams.
    Its frame count must equal the analysis frame count.
    """
    samples = audio_array(audio)
    hop = bank.frequency_bin_count - 1
    transform = ShortTimeFourierTransform(transform_config or FourierConfig(2 * hop, hop))
    if samples.shape[1] != bank.microphone_count or transform.config.frequency_bin_count != bank.frequency_bin_count:
        raise ValueError("audio, transform, and beam bank dimensions must agree")
    spectrum = transform.analyze(samples)
    if speech_mask is not None and np.shape(speech_mask) != spectrum.shape[:2]:
        raise ValueError("speech_mask must match the complete analysis frame grid")
    engines = [GeneralizedSidelobeCanceller(bank, index, config, speech_mask=speech_mask) for index in range(bank.beam_count)]
    output = np.empty((*spectrum.shape[:2], bank.beam_count), dtype=np.complex128)
    for frame_index in range(spectrum.shape[1]):
        for beam_index, engine in enumerate(engines):
            output[:, frame_index, beam_index:beam_index + 1] = engine.process_frame(spectrum[:, frame_index])
    return transform.synthesize(output, length_samples=len(samples))
