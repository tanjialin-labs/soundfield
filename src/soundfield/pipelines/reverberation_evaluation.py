"""Generate reverberant cases, reduce late energy and score declared timelines."""

from dataclasses import dataclass
import numpy as np
from ..acoustics.room_simulation import convolve_room_response, align_delayed_recordings
from ..dereverb import WpeConfig
from ..metrics.coherent_diffuse import CoherentDiffuseConfig, coherent_diffuse_ratio
from ..metrics.signal_quality import log_spectral_distance, perceptual_speech_quality, speech_intelligibility
from ..metrics.modulation_energy import modulation_energy_ratio
from .spatial_enhancement import reduce_reverberation


@dataclass(frozen=True)
class ReverberationEvaluation:
    reverberant_audio: np.ndarray
    enhanced_audio: np.ndarray
    delay_samples: int
    input_scores: dict
    output_scores: dict


def evaluate_reverberation(reference, impulse_response, sample_rate_hz, *, prediction_config=None,
                            response_gain=4., metrics=('spectral_distance', 'coherent_diffuse', 'modulation', 'perceptual')):
    """Score complete WPE output after explicit delay removal; metrics use channel zero."""
    reverberant = convolve_room_response(reference, impulse_response, response_gain=response_gain)
    config = prediction_config or WpeConfig(channel_count=reverberant.shape[1])
    enhanced = reduce_reverberation(reverberant, config)
    delay = config.output_delay_frames*(config.frame_length_samples//2)
    clean, before, after = align_delayed_recordings(reference, reverberant, enhanced, delay)
    def score(values):
        results = {}
        for metric in metrics:
            if metric == 'spectral_distance':
                results[metric] = log_spectral_distance(clean[:, 0], values[:, 0], sample_rate_hz, magnitude_floor=1e-12)
            elif metric == 'coherent_diffuse':
                history = coherent_diffuse_ratio(values, CoherentDiffuseConfig(sample_rate_hz=sample_rate_hz, block_samples=round(.02*sample_rate_hz), hop_samples=round(.02*sample_rate_hz)))
                results[metric] = float(np.mean(history[max(0, int(np.ceil(len(history)/10))-1):]))
            elif metric == 'modulation':
                results[metric] = modulation_energy_ratio(values[:, 0], sample_rate_hz).ratio
            elif metric == 'perceptual':
                results[metric] = vars(perceptual_speech_quality(clean[:, 0], values[:, 0], sample_rate_hz))
            elif metric == 'intelligibility':
                results[metric] = speech_intelligibility(clean[:, 0], values[:, 0], sample_rate_hz)
            else:
                raise ValueError(f'unknown evaluation metric: {metric}')
        return results
    return ReverberationEvaluation(reverberant, enhanced, delay, score(before), score(after))
