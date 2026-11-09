"""Save fixed beam spectra, blocking references, and adaptation strengths."""

import argparse
from pathlib import Path

import numpy as np

from soundfield import FourierConfig, ShortTimeFourierTransform
from soundfield.beamforming import AdaptationGate, AdaptationGateConfig
from soundfield.io import load_beam_bank, read_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--beam-bank", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.resolve() in (arguments.input.resolve(), arguments.beam_bank.resolve()):
        parser.error("output must differ from input resources")
    bank = load_beam_bank(arguments.beam_bank)
    _, audio = read_audio(arguments.input)
    hop = bank.frequency_bin_count - 1
    spectrum = ShortTimeFourierTransform(FourierConfig(2 * hop, hop)).analyze(audio)
    fixed = np.empty((*spectrum.shape[:2], bank.beam_count), dtype=complex)
    blocked = np.empty((*spectrum.shape[:2], bank.reference_count, bank.beam_count), dtype=complex)
    strength = np.empty(fixed.shape)
    gate = AdaptationGateConfig("spatial_amplitude" if bank.beam_count > 1 else "diffuse_amplitude")
    gates = [AdaptationGate(bank, index, gate) for index in range(bank.beam_count)]
    for frame in range(spectrum.shape[1]):
        fixed[:, frame], blocked[:, frame] = bank.project(spectrum[:, frame])
        for index, gate in enumerate(gates):
            strength[:, frame, index:index + 1] = gate.process_frame(spectrum[:, frame])
    with arguments.output.open("wb") as stream:
        np.savez_compressed(stream, fixed_beam_spectrum=fixed, blocking_reference_spectrum=blocked, update_strength=strength)


if __name__ == "__main__":
    main()
