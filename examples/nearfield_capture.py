"""Separate close speech using a symmetric array and explicit fixed beams."""
import argparse
from pathlib import Path
from soundfield.adaptive import NlmsConfig
from soundfield.beamforming import SidelobeConfig
from soundfield.io import read_audio, write_audio, load_beam_bank
from soundfield.separation import DemixingConfig, NearfieldControlConfig
from soundfield.pipelines.nearfield_capture import capture_nearfield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--beam-bank', type=Path, required=True)
    args = parser.parse_args()
    inputs = {args.input.resolve(), args.beam_bank.resolve()}
    if args.output.resolve() in inputs:
        parser.error('output must differ from input resources')
    rate, audio = read_audio(args.input)
    bank = load_beam_bank(args.beam_bank, sample_rate_hz=rate)
    if rate != 16000 or audio.shape[1] != 8 or bank.beam_count != 5:
        parser.error('this configuration uses 16 kHz, eight microphones and five beams')
    enhancement = SidelobeConfig(adaptive_filter=NlmsConfig(frame_length_samples=2*(bank.frequency_bin_count-1), reference_count=bank.reference_count))
    separation = DemixingConfig(frequency_bin_count=2049, channel_count=5, covariance_memory=.9,
        energy_start_bin=255, energy_stop_bin=1000, weighting_strategy='blended_energy', output_mode='channel_mask',
        nearfield_control=NearfieldControlConfig())
    result = capture_nearfield(audio, bank, enhancement, separation)
    write_audio(args.output, result.audio, rate)


if __name__ == '__main__':
    main()
