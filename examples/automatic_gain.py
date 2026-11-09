"""Apply speech-aware gain control to a WAV recording."""

import argparse
from pathlib import Path

import numpy as np

from soundfield.enhancement import GainControlConfig
from soundfield.io import read_audio, write_audio
from soundfield.pipelines import apply_automatic_gain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speech-gains", required=True, type=Path,
                        help="NumPy array with shape (257, frames, channels)")
    parser.add_argument("--speech-level", action="store_true", help="track speech RMS with bounded noise and overlap-add peaks")
    arguments = parser.parse_args()
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("input and output must be different files")
    sample_rate_hz, audio = read_audio(arguments.input)
    speech_gains = np.load(arguments.speech_gains, allow_pickle=False)
    config = GainControlConfig.speech_leveling(sample_rate_hz=sample_rate_hz) if arguments.speech_level else None
    output = apply_automatic_gain(audio, speech_gains, config=config)
    write_audio(arguments.output, output, sample_rate_hz)


if __name__ == "__main__":
    main()
