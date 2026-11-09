"""Enhance fixed beams, separate sources and optionally save intermediate spectra."""

import argparse
from pathlib import Path

import numpy as np

from soundfield.adaptive import NlmsConfig
from soundfield.beamforming import SidelobeConfig
from soundfield.enhancement import ReferencePostfilterConfig, SeparationPostfilterConfig
from soundfield.io import load_beam_bank, read_audio, write_audio
from soundfield.pipelines.beam_separation import separate_beams
from soundfield.separation import DemixingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--beam-bank', type=Path, required=True)
    parser.add_argument('--stages', type=Path)
    parser.add_argument('--postfilter', action='store_true')
    args = parser.parse_args()
    inputs = {args.input.resolve(), args.beam_bank.resolve()}
    outputs = [args.output.resolve(), *([args.stages.resolve()] if args.stages else [])]
    if any(path in inputs for path in outputs) or len(set(outputs)) != len(outputs):
        parser.error('output paths must be distinct from each other and from input resources')
    rate, audio = read_audio(args.input)
    bank = load_beam_bank(args.beam_bank)
    bins = bank.frequency_bin_count
    postfilter = ReferencePostfilterConfig.for_grid(bins, rate) if args.postfilter else None
    enhancement = SidelobeConfig(adaptive_filter=NlmsConfig(frame_length_samples=2*(bins-1),
        sample_rate_hz=rate, reference_count=bank.reference_count, limit_stop_bin=min(257, bins)),
        postfilter=postfilter)
    separation = DemixingConfig(frequency_bin_count=bins, channel_count=bank.beam_count,
        energy_stop_bin=min(257, bins), weighting_strategy='blended_energy',
        output_mode='postfilter' if args.postfilter else 'reference_mask',
        postfilter=SeparationPostfilterConfig(frequency_bin_count=bins, channel_count=bank.beam_count,
            speech_start_bin=postfilter.speech_start_bin, speech_stop_bin=postfilter.speech_stop_bin)
        if args.postfilter else None)
    result = separate_beams(audio, bank, enhancement, separation)
    write_audio(args.output, result.audio, rate)
    if args.stages:
        np.savez_compressed(args.stages, **vars(result), sample_rate_hz=rate, hop_length_samples=bins-1)


if __name__ == '__main__':
    main()
