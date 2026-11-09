"""Suppress background noise in a recording without external resources."""
import argparse
from pathlib import Path

from soundfield.io import read_audio, write_audio
from soundfield.pipelines import suppress_background_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-gain", type=float, default=.16)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ")
    rate, audio = read_audio(args.input)
    write_audio(args.output, suppress_background_noise(audio, rate, minimum_gain=args.minimum_gain), rate)


if __name__ == "__main__":
    main()
