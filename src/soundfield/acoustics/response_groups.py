"""Compare impulse responses by spectral shape and early-to-late energy."""

from dataclasses import dataclass
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from ..contracts import real_array, positive_integer, finite_scalar
from ._sweep_math import _response_features


@dataclass(frozen=True)
class ResponseGroups:
    labels: np.ndarray
    distances: np.ndarray
    spectral_shapes_db: np.ndarray
    clarity_db: np.ndarray
    energy_db: np.ndarray


def group_impulse_responses(responses, sample_rate_hz, *, distance_threshold=3.0):
    """Group mono responses; distance combines shape RMS and 0.35 times C50."""
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    finite_scalar(distance_threshold, 'distance_threshold', minimum=0)
    if sample_rate_hz < 1000 or not len(responses):
        raise ValueError('provide responses sampled at 1000 Hz or higher')
    shapes, clarity, energy = [], [], []
    for response in responses:
        values = real_array(response, 'response', 1)
        if not len(values) or not np.any(values):
            raise ValueError('each response must contain nonzero samples')
        shape, summary = _response_features(values, sample_rate_hz)
        shapes.append(shape)
        clarity.append(summary['clarity_c50_db'])
        energy.append(summary['rir_energy_db'])
    shapes, clarity = np.asarray(shapes), np.asarray(clarity)
    distances = np.sqrt(np.mean((shapes[:, None]-shapes[None, :])**2, axis=-1)+(.35*(clarity[:, None]-clarity[None, :]))**2)
    labels = fcluster(linkage(squareform(distances), method='average'), distance_threshold, criterion='distance') if len(shapes)>1 else np.ones(1, int)
    return ResponseGroups(labels, distances, shapes, clarity, np.asarray(energy))
