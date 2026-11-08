"""Analyze and reconstruct a WAV recording."""

import argparse
from pathlib import Path

from soundfield import FourierConfig, ShortTimeFourierTransform
from soundfield.io import read_audio, write_audio
from soundfield.transform.short_time_fourier import WINDOW_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame-length", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--window", choices=WINDOW_NAMES, default="sqrt_hann")
    parser.add_argument("--encoding", choices=("pcm16", "float64"), default="pcm16")
    arguments = parser.parse_args()
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("input and output must be different files")
    sample_rate_hz, audio = read_audio(arguments.input)
    transform = ShortTimeFourierTransform(FourierConfig(
        arguments.frame_length, arguments.hop_length, arguments.window,
    ))
    output = transform.synthesize(transform.analyze(audio), length_samples=len(audio))
    write_audio(arguments.output, output, sample_rate_hz, encoding=arguments.encoding)


if __name__ == "__main__":
    main()
