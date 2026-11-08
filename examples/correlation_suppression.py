"""Enhance every audio channel using cross-channel noise correlation."""

import argparse
from pathlib import Path

from soundfield.io import read_audio, write_audio
from soundfield.pipelines import suppress_correlated_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.resolve() == arguments.input.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    write_audio(arguments.output, suppress_correlated_noise(audio), rate)


if __name__ == "__main__":
    main()
