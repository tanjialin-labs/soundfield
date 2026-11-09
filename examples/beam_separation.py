"""Enhance fixed beams and separate their sources."""

import argparse
from pathlib import Path

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
    parser.add_argument('--postfilter', action='store_true')
    parser.add_argument('--postfilter-mask-exponent', type=float, default=1.,
                        help='output amplitude exponent for the separation mask when postfiltering')
    args = parser.parse_args()
    if not args.postfilter and args.postfilter_mask_exponent != 1.:
        parser.error('--postfilter-mask-exponent requires --postfilter')
    inputs = {args.input.resolve(), args.beam_bank.resolve()}
    if args.output.resolve() in inputs:
        parser.error('output must differ from input resources')
    rate, audio = read_audio(args.input)
    bank = load_beam_bank(args.beam_bank, sample_rate_hz=rate)
    bins = bank.frequency_bin_count
    postfilter = ReferencePostfilterConfig.for_grid(bins, rate) if args.postfilter else None
    enhancement = SidelobeConfig(adaptive_filter=NlmsConfig(frame_length_samples=2*(bins-1),
        sample_rate_hz=rate, reference_count=bank.reference_count, limit_stop_bin=min(257, bins)),
        postfilter=postfilter)
    separation = DemixingConfig(frequency_bin_count=bins, channel_count=bank.beam_count,
        energy_stop_bin=min(257, bins), weighting_strategy='blended_energy',
        output_mode='postfilter' if args.postfilter else 'reference_mask',
        postfilter=SeparationPostfilterConfig(frequency_bin_count=bins, channel_count=bank.beam_count,
            speech_start_bin=postfilter.speech_start_bin, speech_stop_bin=postfilter.speech_stop_bin,
            mask_exponent=args.postfilter_mask_exponent)
        if args.postfilter else None)
    result = separate_beams(audio, bank, enhancement, separation)
    write_audio(args.output, result.audio, rate)


if __name__ == '__main__':
    main()
