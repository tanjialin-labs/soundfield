"""Gate a separated speech source using an external direction timeline."""
import argparse
from pathlib import Path
import numpy as np
from soundfield.adaptive import NlmsConfig
from soundfield.beamforming import SidelobeConfig
from soundfield.enhancement import SpatialActivityConfig
from soundfield.io import read_audio, write_audio, load_beam_bank
from soundfield.separation import DemixingConfig
from soundfield.localization import DirectionalMatchingConfig, linear_array
from soundfield.pipelines.directional_capture import capture_directed_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--beam-bank', type=Path, required=True)
    parser.add_argument('--directions', type=Path, required=True, help='NPY containing one scan index per frame')
    parser.add_argument('--spacing-meters', type=float, default=.03)
    parser.add_argument('--target-source', type=int, default=1)
    parser.add_argument("--competing-vote-fraction", type=float, default=0.)
    parser.add_argument("--inactive-gain", type=float, default=0.)
    parser.add_argument("--selective-attenuation", action="store_true", help="Use inter-microphone phase to protect speech outside the target sector")
    args = parser.parse_args()
    inputs = {path.resolve() for path in (args.input, args.beam_bank, args.directions)}
    if args.output.resolve() in inputs:
        parser.error('output must differ from input resources')
    rate, audio = read_audio(args.input)
    bank = load_beam_bank(args.beam_bank, sample_rate_hz=rate)
    bins = bank.frequency_bin_count
    enhancement = SidelobeConfig(adaptive_filter=NlmsConfig(frame_length_samples=2*(bins-1), reference_count=bank.reference_count))
    separation = DemixingConfig(frequency_bin_count=bins, channel_count=bank.beam_count, energy_stop_bin=min(257,bins), weighting_strategy='blended_energy')
    matching = DirectionalMatchingConfig(geometry=linear_array(audio.shape[1], args.spacing_meters), frame_length_samples=2*(bins-1),
        sample_rate_hz=rate, block_frames=1, event_delay_frames=0, search_all_directions=True)
    result = capture_directed_region(audio, bank, enhancement, separation, matching,
        selective_attenuation=args.selective_attenuation,
        activity_config=SpatialActivityConfig(competing_vote_fraction=args.competing_vote_fraction,inactive_gain=args.inactive_gain),
        target_source_index=args.target_source, target_direction_indices=np.load(args.directions, allow_pickle=False))
    write_audio(args.output, result.audio, rate)


if __name__ == '__main__':
    main()
