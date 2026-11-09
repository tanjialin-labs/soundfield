"""Portable storage for fixed beams and blocking matrices."""
from hashlib import sha256

import numpy as np

from ..beamforming.fixed_beams import FixedBeamBank
from ..contracts import positive_integer, real_array


def _positions(values, count):
    positions = real_array(values, 'positions_meters', 2)
    if positions.shape != (count, 3):
        raise ValueError('positions_meters must describe every microphone in order')
    return positions


def _digest(weights, blocking, rate, positions):
    checksum = sha256()
    for array in (weights, blocking, np.array(rate), positions):
        checksum.update(np.ascontiguousarray(array).tobytes())
    return checksum.hexdigest()


def load_beam_bank(path, *, sample_rate_hz=None, positions_meters=None, require_metadata=False) -> FixedBeamBank:
    with np.load(path, allow_pickle=False) as resource:
        fixed, blocking = resource["fixed_beam_weights"], resource["blocking_matrix"]
        bank = FixedBeamBank(fixed, blocking)
        if 'format_version' not in resource:
            if require_metadata or positions_meters is not None:
                raise ValueError('beam resource must carry array metadata')
            return bank
        if resource['format_version'].item() != 1:
            raise ValueError('unsupported beam resource version')
        rate = resource['sample_rate_hz'].item()
        positive_integer(rate, 'sample_rate_hz')
        positions = _positions(resource['positions_meters'], bank.microphone_count)
        if sample_rate_hz is not None and rate != sample_rate_hz:
            raise ValueError('beam resource sample rate differs from audio')
        if positions_meters is not None and not np.allclose(positions, _positions(positions_meters,bank.microphone_count), rtol=0, atol=1e-8):
            raise ValueError('beam resource microphone geometry or order differs')
        if _digest(fixed, blocking, rate, positions) != resource['sha256'].item():
            raise ValueError('beam resource digest differs')
        return bank


def save_beam_bank(path, bank: FixedBeamBank, *, sample_rate_hz=None, positions_meters=None) -> None:
    fixed, blocking = bank.fixed_beam_weights, bank.blocking_matrix
    metadata = {}
    if sample_rate_hz is not None or positions_meters is not None:
        positive_integer(sample_rate_hz, 'sample_rate_hz')
        positions = _positions(positions_meters, bank.microphone_count)
        metadata = dict(format_version=1, sample_rate_hz=sample_rate_hz, positions_meters=positions,
                        sha256=_digest(fixed, blocking, sample_rate_hz, positions))
    with open(path, "wb") as stream:
        np.savez_compressed(stream, fixed_beam_weights=fixed, blocking_matrix=blocking, **metadata)
