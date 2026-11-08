"""Load explicitly named numeric impulse-response arrays and metadata."""

from pathlib import Path
import numpy as np
from scipy.io import loadmat
from ..contracts import audio_array, positive_integer


def load_impulse_array(path, *, array_key, sample_rate_hz=None, sample_rate_key=None, sample_axis=0):
    """Read NPZ or MAT numeric arrays; dotted metadata keys address nested fields."""
    path = Path(path)
    if (sample_rate_hz is None) == (sample_rate_key is None):
        raise ValueError('provide exactly one of sample_rate_hz and sample_rate_key')
    if sample_axis not in (0, 1):
        raise ValueError('sample_axis must be zero or one')
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as resource:
            data = {key: resource[key] for key in resource.files}
    elif path.suffix.lower() == '.mat':
        data = loadmat(path, simplify_cells=True)
    else:
        raise ValueError('numeric response resources must be NPZ or MAT files')
    def field(name):
        value = data
        for component in name.split('.'):
            value = value[component]
        return value
    values = np.asarray(field(array_key))
    if np.iscomplexobj(values):
        raise ValueError('impulse-response samples must be real')
    if values.ndim == 1:
        values = values[:, None]
    elif sample_axis == 1:
        values = values.T
    rate = sample_rate_hz if sample_rate_key is None else np.asarray(field(sample_rate_key)).item()
    if not isinstance(rate, (int, np.integer)) and isinstance(rate, (float, np.floating)) and rate.is_integer():
        rate = int(rate)
    positive_integer(rate, 'sample_rate_hz')
    return rate, audio_array(values)
