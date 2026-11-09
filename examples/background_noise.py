"""Suppress background noise using statistical gains or a recurrent spectrum model."""
import argparse
from pathlib import Path

from soundfield.io import read_audio, write_audio
from soundfield.pipelines import suppress_background_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--minimum-gain", type=float)
    mode.add_argument("--recurrent-model", type=Path)
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.input, args.recurrent_model) if p is not None}:
        parser.error("input and output must differ")
    rate, audio = read_audio(args.input)
    if args.recurrent_model:
        from soundfield.neural.recurrent_spectrum import RecurrentSpectrumEnhancer
        output = RecurrentSpectrumEnhancer(args.recurrent_model).enhance_audio(audio, rate)
    else:
        output = suppress_background_noise(audio, rate, minimum_gain=.16 if args.minimum_gain is None else args.minimum_gain)
    write_audio(args.output, output, rate)


if __name__ == "__main__":
    main()
