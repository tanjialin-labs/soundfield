"""Align neural speech evidence with directional scores and channel decisions."""

import numpy as np

from ..contracts import spectral_array, real_array
from ..localization.speech_direction import SpeechGatedDirectionTracker, SpeechDirectionConfig
from ..neural.band_features import round_away_from_zero


def track_speech_directions(spectrum, direction_scores, selected_channels, mask_estimator, *,
                            config=SpeechDirectionConfig(), speech_bins=(15,64)):
    """Return one direction index per normalized spectral frame; -1 denotes inactivity."""
    spectrum=spectral_array(spectrum)
    scores=real_array(direction_scores,'direction_scores',2)
    selected=np.asarray(selected_channels)
    bins,frames,channels=spectrum.shape
    if (bins!=mask_estimator.config.frequency_bin_count or channels!=mask_estimator.channel_count
            or scores.shape!=(27,frames) or selected.shape!=(frames,) or selected.dtype.kind not in 'iu'
            or np.any(selected<0) or np.any(selected>=min(3,channels)) or np.any(scores<0)):
        raise ValueError('spectra, directional scores and selected channels must have aligned frames')
    start,stop=speech_bins
    if not 0<=start<stop<=bins:raise ValueError('speech_bins must fit inside the spectrum')
    tracker=SpeechGatedDirectionTracker(config)
    mask_estimator.reset()
    output=np.full(frames,-1,int)
    for frame in range(frames+mask_estimator.config.right_context_frames):
        current=spectrum[:,frame] if frame<frames else np.zeros((bins,channels),complex)
        quantized=round_away_from_zero(current.real*2**21)+1j*round_away_from_zero(current.imag*2**21)
        mask=mask_estimator.process_frame(quantized)
        target=mask.target_frame_index
        if 0<=target<frames:
            channel=int(selected[target])
            index=tracker.process_frame(scores[:,target],float(mask.gains[start:stop,channel].sum()),channel)
            output[target]=-1 if index is None else index
    return output
