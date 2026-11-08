"""Apply near-field spatial gains to a symmetric eight-microphone recording."""

import argparse
from pathlib import Path

from soundfield.io import read_audio, write_audio
from soundfield.pipelines import classify_nearfield_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    if rate != 16000:
        parser.error("this example uses 16 kHz and the documented eight-microphone geometry")
    write_audio(arguments.output, classify_nearfield_audio(audio), rate)


if __name__ == "__main__":
    main()
