"""Versioned sweep stimuli with content-checked audio resources."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from ..acoustics.sweep_measurement import SweepConfig, SweepStimulus
from .audio import read_audio, write_audio


def save_sweep_stimulus(manifest_path, audio_path, stimulus):
    """Store float64 excitation and a portable manifest beside it."""
    manifest_path, audio_path = Path(manifest_path), Path(audio_path)
    if manifest_path.resolve() == audio_path.resolve() or manifest_path.resolve().parent != audio_path.resolve().parent:
        raise ValueError('manifest and audio must be distinct files in the same directory')
    write_audio(audio_path, stimulus.audio, stimulus.config.sample_rate_hz, encoding='float64')
    manifest = dict(schema='soundfield.sweep.v1', configuration=asdict(stimulus.config),
                    audio_file=audio_path.name, audio_sha256=sha256(audio_path.read_bytes()).hexdigest(),
                    segments=stimulus.segments)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_sweep_stimulus(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'soundfield.sweep.v1':
        raise ValueError('unsupported sweep stimulus schema')
    filename = manifest['audio_file']
    if Path(filename).name != filename:
        raise ValueError('audio_file must name a file beside the manifest')
    path = manifest_path.parent/filename
    if sha256(path.read_bytes()).hexdigest() != manifest['audio_sha256']:
        raise ValueError('stimulus audio content does not match the manifest')
    rate, audio = read_audio(path)
    config = SweepConfig(**manifest['configuration'])
    if rate != config.sample_rate_hz:
        raise ValueError('stimulus sample rate does not match the manifest')
    return SweepStimulus(audio, config, tuple(manifest['segments']))
