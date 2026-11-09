"""Prepare a sample-aligned microphone segment and its enhancement for recognition."""
import numpy as np
from ..contracts import audio_array


def prepare_recognition_segment(microphone, enhanced):
    """Retain microphone detail and normalize one complete mono utterance with limited gain."""
    original = audio_array(microphone)
    processed = audio_array(enhanced)
    if original.shape[1] != 1 or original.shape != processed.shape:
        raise ValueError("recognition inputs must be aligned mono segments")
    segment = .8*original + .2*processed
    if segment.size:
        rms = np.sqrt(np.mean(segment**2))
        gain = min(12., .06/max(rms, 1e-10), .8/max(np.max(abs(segment)), 1e-10))
        segment *= gain
    return segment
