"""Enhance microphone audio using an explicit beam bank."""

import argparse
from pathlib import Path

import numpy as np

from soundfield.adaptive import NlmsConfig
from soundfield.beamforming import AdaptationGateConfig, SidelobeConfig
from soundfield.enhancement import ReferencePostfilterConfig
from soundfield.io import load_beam_bank, read_audio, write_audio
from soundfield.pipelines import enhance_microphone_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--beam-bank", type=Path, required=True)
    parser.add_argument("--strategy", choices=("continuous", "spatial_power", "spatial_amplitude", "diffuse_amplitude", "speech_mask"), default="spatial_amplitude")
    parser.add_argument("--speech-mask", type=Path)
    parser.add_argument("--postfilter", action="store_true")
    parser.add_argument("--disable-gate-attenuation", action="store_true")
    parser.add_argument("--protect-target-bins", action="store_true",
                        help="reduce adaptive cancellation where the gate favors the target")
    arguments = parser.parse_args()
    resources = [arguments.input, arguments.beam_bank]
    if arguments.speech_mask:
        resources.append(arguments.speech_mask)
    if arguments.output.resolve() in [path.resolve() for path in resources]:
        parser.error("output must differ from every input resource")
    bank = load_beam_bank(arguments.beam_bank)
    sample_rate, audio = read_audio(arguments.input)
    config = SidelobeConfig(
        NlmsConfig(sample_rate_hz=sample_rate, reference_count=bank.reference_count),
        AdaptationGateConfig(arguments.strategy),
        enable_gate_attenuation=not arguments.disable_gate_attenuation,
        protect_target_bins=arguments.protect_target_bins,
        postfilter=ReferencePostfilterConfig() if arguments.postfilter else None,
    )
    mask = np.load(arguments.speech_mask, allow_pickle=False) if arguments.speech_mask else None
    enhanced = enhance_microphone_array(audio, bank, config, speech_mask=mask)
    write_audio(arguments.output, enhanced, sample_rate)


if __name__ == "__main__":
    main()
