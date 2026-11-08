"""Portable storage for fixed beams and blocking matrices."""

import numpy as np

from ..beamforming.fixed_beams import FixedBeamBank


def load_beam_bank(path) -> FixedBeamBank:
    with np.load(path, allow_pickle=False) as resource:
        return FixedBeamBank(resource["fixed_beam_weights"], resource["blocking_matrix"])


def save_beam_bank(path, bank: FixedBeamBank) -> None:
    with open(path, "wb") as stream:
        np.savez_compressed(stream, fixed_beam_weights=bank.fixed_beam_weights,
                            blocking_matrix=bank.blocking_matrix)
