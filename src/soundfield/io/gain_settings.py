"""JSON resources for automatic-gain parameters."""

import json
from pathlib import Path
from dataclasses import asdict
from ..enhancement import GainControlConfig, GainCurveConfig, GainSmoothingConfig


def load_gain_settings(path):
    values = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(values, dict):
        raise ValueError('gain settings must be a JSON object')
    values = dict(values)
    values['curve'] = GainCurveConfig(**values.get('curve', {}))
    values['smoothing'] = GainSmoothingConfig(**values.get('smoothing', {}))
    return GainControlConfig(**values)


def save_gain_settings(path, config):
    if not isinstance(config, GainControlConfig):
        raise TypeError('config must be a GainControlConfig')
    Path(path).write_text(json.dumps(asdict(config), indent=2, allow_nan=False)+'\n', encoding='utf-8')
