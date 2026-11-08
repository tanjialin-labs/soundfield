"""Gate a separated speech source using an external direction timeline."""
import argparse
from pathlib import Path
import numpy as np
from soundfield.adaptive import NlmsConfig
from soundfield.beamforming import SidelobeConfig
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
    parser.add_argument('--stages', type=Path)
    args = parser.parse_args()
    inputs = {path.resolve() for path in (args.input, args.beam_bank, args.directions)}
    outputs = [path.resolve() for path in (args.output, args.stages) if path]
    if any(path in inputs for path in outputs) or len(set(outputs)) != len(outputs):
        parser.error('outputs must be distinct from input resources and each other')
    rate, audio = read_audio(args.input)
    bank = load_beam_bank(args.beam_bank)
    bins = bank.frequency_bin_count
    enhancement = SidelobeConfig(adaptive_filter=NlmsConfig(frame_length_samples=2*(bins-1), reference_count=bank.reference_count))
    separation = DemixingConfig(frequency_bin_count=bins, channel_count=bank.beam_count, energy_stop_bin=min(257,bins), weighting_strategy='blended_energy')
    matching = DirectionalMatchingConfig(geometry=linear_array(audio.shape[1], args.spacing_meters), frame_length_samples=2*(bins-1),
        sample_rate_hz=rate, block_frames=1, event_delay_frames=0, search_all_directions=True)
    result = capture_directed_region(audio, bank, enhancement, separation, matching,
        target_source_index=args.target_source, target_direction_indices=np.load(args.directions, allow_pickle=False))
    write_audio(args.output, result.audio, rate)
    if args.stages:
        np.savez_compressed(args.stages, **vars(result), sample_rate_hz=rate)


if __name__ == '__main__':
    main()
